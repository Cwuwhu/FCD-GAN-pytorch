import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as trans
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models

import os
from osgeo import gdal
from osgeo import ogr
from osgeo import osr
import numpy as np
import cv2
from tqdm import tqdm
import gc
from PIL import Image
import time

from Module import *
from data_utils import *
from metrics import Evaluator
from CommonFunc import *
from Loss import *

from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    init_num_epochs_G = 50  # initial training epochs for generator
    num_epochs = 100  # initial training epochs
    learning_rate = 0.00005 # learning rate for RMSprob
    init_batch_size = 20 # batch size for generator
    batch_size = 12 # batch size for model training

    patch_size = (200, 200) # slice patch size
    overlap_padding = (10, 10)   # overlap padding size of the patch
    gt_map = [1, 2]    # non-change/change label for ground truth
    pre_map = [0, 1]       # non-change/change label for predition map

    prob_thresh = 0.5          # threshold to segmentation binary change map
    tips = ''            # tips record in the txt file

    perception_weight = 0.1 # parameters for generator
    ssim_weight = 0
    perception_perBand = True
    perception_layer = 1

    l1_weight = 0.02    # weights for l1-loss
    g_weight = 0.5  # weights for generator loss
    d_weight = 1    # weights for GAN loss
    r_weight = 2    # weights for region loss

    write_color = True  # write a colorful change map, where TP/TN/FP/FN are shown with different colors
    modelG_reuse = True # model of generator can be directly reused to save time in repeated experiments

    discriminator_continuous = True # switch the masking of change map to be soft or hard, True (soft) is DEFAULT

    imgDir = r'/OSCD-10m-Dataset/'  # input dir for OSCD dataset
    OutGModelDir = r'/GModel'  # output dir for generator model, reuse is available
    txtName = 'train.txt'   # txt file to record training data in OSCD dataset
    text_txtName = 'test.txt'   # txt file to record testing data in OSCD dataset
    outName_density = 'density'
    outName_binary = 'color'
    extName = '_l1002_r2_d1_g05_github'  # expansion name for the output result, which is convenient to record experiments

    writer = SummaryWriter(comment='RSSS_OSCD{}'.format(extName))   # tensorboard

    OutDir = os.path.join(imgDir, 'model{}'.format(extName))    # dir to save network model
    if os.path.exists(OutDir) == False:
        os.mkdir(OutDir)

    # obtain the parameter for normalization and pre-preocessing
    tmp_dataset = OSCD_Dataset_RSS(imgDir, txtName)
    pathlist = tmp_dataset.pathlist
    scaler_list = []
    transforms_list = []
    statsName = 'statsMS'
    for path in pathlist:
        # obtain the normalization parameter for each image in OSCD dataset
        ImgXPath = path[0]
        ImgYPath = path[1]
        cur_path, cur_ImgXName = os.path.split(ImgXPath)
        cur_path, cur_ImgYName = os.path.split(ImgYPath)
        cur_ImgXName, _ = os.path.splitext(cur_ImgXName)
        cur_ImgYName, _ = os.path.splitext(cur_ImgYName)
        dataset_tmp = GDALDataset(ImgXPath, ImgYPath, patch_size=patch_size, overlap_padding=(0, 0))
        statsPath1 = os.path.join(cur_path, '{}_{}.txt'.format(cur_ImgXName, statsName))
        statsPath2 = os.path.join(cur_path, '{}_{}.txt'.format(cur_ImgYName, statsName))
        # scale_list1, scale_list2 = Dataset_maxmin(statsPath1, statsPath2, dataset_tmp)
        meanX, stdX, meanY, stdY = Dataset_meanstd(statsPath1, statsPath2, dataset_tmp) # normalization is DEFAULT

        # scaler_list.append(SCALE(scale_list1=scale_list1, scale_list2=scale_list2))
        scaler_list.append(NORMALIZE(meanX, stdX, meanY, stdY))
        # transforms_list.append(RANDOM_ERASER_MULTI_REGION())  # data augmentation (not used in this experiment)
        transforms_list.append(None)

    # build training dataset and dataloader
    dataset = OSCD_Dataset_RSS(imgDir, txtName, scaler=scaler_list, transforms=transforms_list, patch_size=patch_size, overlap_padding=overlap_padding)
    total_dataset_size = dataset.__len__()
    train_dataloader = DataLoader(dataset, batch_size=init_batch_size, shuffle=True)

    # build testing dataset and dataloader
    # obtain paramaters for normalization and pre-processing
    tmp_dataset = OSCD_Dataset_RSS(imgDir, text_txtName)
    pathlist = tmp_dataset.pathlist
    scaler_list = []
    transforms_list = []
    statsName = 'statsMS'
    for path in pathlist:
        ImgXPath = path[0]
        ImgYPath = path[1]
        cur_path, cur_ImgXName = os.path.split(ImgXPath)
        cur_path, cur_ImgYName = os.path.split(ImgYPath)
        cur_ImgXName, _ = os.path.splitext(cur_ImgXName)
        cur_ImgYName, _ = os.path.splitext(cur_ImgYName)
        dataset_tmp = GDALDataset(ImgXPath, ImgYPath, patch_size=patch_size, overlap_padding=(0, 0))
        statsPath1 = os.path.join(cur_path, '{}_{}.txt'.format(cur_ImgXName, statsName))
        statsPath2 = os.path.join(cur_path, '{}_{}.txt'.format(cur_ImgYName, statsName))
        # scale_list1, scale_list2 = Dataset_maxmin(statsPath1, statsPath2, dataset_tmp)
        # scaler_list.append(SCALE(scale_list1=scale_list1, scale_list2=scale_list2))

        meanX, stdX, meanY, stdY = Dataset_meanstd(statsPath1, statsPath2, dataset_tmp)
        scaler_list.append(NORMALIZE(meanX, stdX, meanY, stdY))

        # scaler_list.append(SCALE(scale_list1=scale_list1, scale_list2=scale_list2))
        # transforms_list.append(RANDOM_ERASER())
        transforms_list.append(None)

    test_dataset = OSCD_Dataset_RSS(imgDir, text_txtName, scaler=scaler_list, transforms=None, patch_size=patch_size,
                                    overlap_padding=overlap_padding)
    total_test_dataset_size = test_dataset.__len__()
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # model training
    netD = Discriminator_SRGAN_simple(n_channels=4)
    netD.to(device)

    netS = Segmentor(n_channels=4, bilinear=True)
    netS.to(device)

    netG = Generator(n_channels=4)
    netG.to(device)

    netS.train()
    netG.train()
    netD.train()

    # for generator, choose Adam
    optimizerG = torch.optim.Adam(netG.parameters(), lr=learning_rate, betas=(0.9, 0.99))
    # optimizerS = torch.optim.Adam(netS.parameters(), lr=learning_rate, betas=(0.9, 0.99))
    # optimizerD = torch.optim.Adam(netD.parameters(), lr=learning_rate, betas=(0.9, 0.99))

    # for GAN optimization, choose RMSprop, according to WGAN
    # optimizerG = torch.optim.RMSprop(netG.parameters(), lr=learning_rate)
    optimizerS = torch.optim.RMSprop(netS.parameters(), lr=learning_rate)
    optimizerD = torch.optim.RMSprop(netD.parameters(), lr=learning_rate)

    acc = Evaluator(num_class=len(gt_map))
    acc_test = Evaluator(num_class=len(gt_map))

    g_criterion = CGeneratorLoss(channel=4, perception_layer=perception_layer, perception_perBand=perception_perBand)
    g_criterion.to(device)

    # reuse of generator model, for saving time
    if modelG_reuse == True:
        path = os.path.join(OutGModelDir, 'GModel.pkl')
        if os.path.exists(path) == True:
            init_num_epochs_G = 0
            netG.load_state_dict(torch.load(path))

    print('Start Generator Training')
    with torch.enable_grad():
        for i in range(init_num_epochs_G):
            g_loss_aver = 0
            generator_loss_aver = 0
            perception_loss_aver = 0
            ssim_loss_aver = 0
            adjust_learning_rate(optimizerG, i, lr_start=1e-5, lr_max=3e-4, lr_warm_up_epoch=10, lr_sustain_epochs=10)

            process_num = 0

            for data_array in train_dataloader:

                time_start = time.time()

                # Update G network:

                optimizerG.zero_grad()

                x = data_array[0]
                y = data_array[1]
                region = data_array[4]

                x = x.to(device)
                y = y.to(device)
                region = region.to(device)

                y_fake = netG(x)

                # prediction of multi-temporal images, with the mask of supervised regions
                generator_loss, ssim_loss, perception_loss = g_criterion(y, y_fake, region)

                g_loss = generator_loss + perception_weight * perception_loss + ssim_weight * ssim_loss

                g_loss.backward()
                optimizerG.step()

                g_loss_aver += g_loss.item() * x.size(0) / total_dataset_size
                generator_loss_aver += generator_loss.item() * x.size(0) / total_dataset_size
                perception_loss_aver += perception_loss.item() * x.size(0) / total_dataset_size
                ssim_loss_aver += ssim_loss.item() * x.size(0) / total_dataset_size

                process_num += x.size()[0]

                time_end = time.time()
                time_per_iter = (time_end - time_start) / x.size()[0] * total_dataset_size
                time_remaining = time_per_iter * (
                        (init_num_epochs_G - 1 - i) + (1 - process_num / total_dataset_size))
                time_desc_per = time_show(time_per_iter)
                time_desc = time_show(time_remaining)

                print('\rProcessing batch: {}/{}; Processing speed per iter: {}; Processing time remaining: {}'.format(
                    process_num, total_dataset_size, time_desc_per, time_desc), end='', flush=True)

                # print('\rProcessing batch: {}/{}'.format(process_num, total_dataset_size), end='', flush=True)

            print('\r', end='', flush=True)

            print(
                'Epochs: {}/{}, g_loss: {:.4f}, generator_loss: {:.4f}, perception_loss:{:.4f}, ssim_loss:{:.4f}'.format(
                    i + 1, init_num_epochs_G, g_loss_aver, generator_loss_aver, perception_loss_aver, ssim_loss_aver))

            writer.add_scalar('g_loss', g_loss_aver, i)
            writer.add_scalar('generator_loss', generator_loss_aver, i)
            writer.add_scalar('perception_loss', perception_loss_aver, i)
            writer.add_scalar('ssim_loss', ssim_loss_aver, i)

    netG.eval()

    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print('Start Adversarial Training')
    with torch.enable_grad():
        for i in range(num_epochs):
            g_loss_aver = 0
            s_loss_aver = 0
            s_d_loss_aver = 0
            d_loss_aver = 0
            l1_loss_aver = 0
            r_loss_aver = 0

            generator_loss_aver = 0
            perception_loss_aver = 0
            ssim_loss_aver = 0

            acc.reset()

            # in the adversarial process, discriminator is easier to converge, thus optimization with lower learning rate
            adjust_learning_rate(optimizerS, i, lr_start=1e-4, lr_max=1e-3, lr_warm_up_epoch=5)
            adjust_learning_rate(optimizerD, i, lr_start=5e-6, lr_max=5e-5, lr_min=5e-7, lr_warm_up_epoch=5)

            process_num = 0

            for data_array in train_dataloader:

                time_start = time.time()

                x = data_array[0]
                y = data_array[1]
                item = data_array[2]
                ref = data_array[3]
                region = data_array[4]

                x = x.to(device)
                y = y.to(device)
                region = region.to(device)

                ##################
                # Update D network
                ##################

                # Calculate loss of CHANGED images
                cmap = netS(x, y)
                if discriminator_continuous == True:
                    cmask = cmap
                else:
                    cmask = (torch.sign(cmap - 0.5) + 1) / 2
                x_mask = x * (1 - cmask.repeat((1, x.size()[1], 1, 1)))
                y_mask = y * (1 - cmask.repeat((1, y.size()[1], 1, 1)))

                c_out = netD(x_mask, y_mask)

                # generate a fake unchanged image with the mask of region
                x_unc = x
                y_unc = y * (1 - region) + x * region

                x_unc = x_unc * (1 - cmask.repeat((1, x.size()[1], 1, 1)))
                y_unc = y_unc * (1 - cmask.repeat((1, y.size()[1], 1, 1)))
                nc_out = netD(x_unc, y_unc)

                optimizerD.zero_grad()
                d_loss = 1 + nc_out.mean() - c_out.mean()
                d_loss.backward(retain_graph=True)

                optimizerD.step()

                # Clip weights of discriminator
                # for p in netD.parameters():
                #     p.data.clamp_(-1, 1)

                ##################
                # Update S network
                ##################

                c_out = netD(x_mask, y_mask)

                y_fake = netG(x)
                generator_loss, ssim_loss, perception_loss = g_criterion(y, y_fake, cmap)
                g_loss = generator_loss + perception_weight * perception_loss + ssim_weight * ssim_loss
                criterion = nn.L1Loss()
                l1_loss = region_loss(cmap, region, criterion)
                s_d_loss = c_out.mean()

                criterion = nn.MSELoss()
                r_loss = region_loss(cmap, 1 - region, criterion)
                s_loss = d_weight * s_d_loss + l1_weight * l1_loss + g_weight * g_loss + r_weight * r_loss

                optimizerS.zero_grad()
                s_loss.backward()
                optimizerS.step()

                d_loss_aver += d_loss.item() * x.size(0) / total_dataset_size
                s_d_loss_aver += s_d_loss.item() * x.size(0) / total_dataset_size
                g_loss_aver += g_loss.item() * x.size(0) / total_dataset_size
                s_loss_aver += s_loss.item() * x.size(0) / total_dataset_size
                l1_loss_aver += l1_loss.item() * x.size(0) / total_dataset_size
                r_loss_aver += r_loss.item() * x.size(0) / total_dataset_size

                generator_loss_aver += generator_loss.item() * x.size(0) / total_dataset_size
                ssim_loss_aver += ssim_loss.item() * x.size(0) / total_dataset_size
                perception_loss_aver += perception_loss.item() * x.size(0) / total_dataset_size

                cmask = torch.zeros_like(cmap)
                cmask[cmap > prob_thresh] = 1
                for ns in range(x.size(0)):
                    change_mask = cmask[ns][0]
                    change_mask = change_mask.cpu().numpy()
                    ref_mask = ref[ns][0].numpy()

                    # when evaluating the accuracy, only consider the centering region of image patch without overlapping padding, to avoid the problem in patch edge
                    acc_range = dataset.EffRange(item[ns].numpy())
                    acc.add_batch_map(ref_mask[acc_range[0]:acc_range[1], acc_range[2]:acc_range[3]].astype(np.int16), change_mask[acc_range[0]:acc_range[1], acc_range[2]:acc_range[3]].astype(np.int16), gt_map, pre_map)

                process_num += x.size()[0]

                time_end = time.time()
                time_per_iter = (time_end - time_start) / x.size()[0] * total_dataset_size
                time_remaining = time_per_iter * (
                        (num_epochs - 1 - i) + (1 - process_num / total_dataset_size))
                time_desc_per = time_show(time_per_iter)
                time_desc = time_show(time_remaining)

                print('\rProcessing batch: {}/{}; Processing speed per iter: {}; Processing time remaining: {}'.format(
                    process_num, total_dataset_size, time_desc_per, time_desc), end='', flush=True)

                # print('\rProcessing batch: {}/{}'.format(process_num, total_dataset_size), end='', flush=True)

            print('\r', end='', flush=True)

            print(
                'Epochs: {}/{}, d_loss: {:.4f}, g_loss: {:.4f}, s_loss: {:.4f}, l1_loss:{:.4f}, s_d_loss: {:.4f}, r_loss: {:.4f}'.format(
                    i + 1, num_epochs, d_loss_aver, g_loss_aver, s_loss_aver, l1_loss_aver, s_d_loss_aver, r_loss_aver))
            print(
                'Epochs: {}/{}, Overall Accuracy: {:.4f}, Kappa: {:.4f}, Precision Rate: {:.4f}, Recall Rate: {:.4f}, F1:{:.4f}, mIOU:{:.4f}, cIoU:{:.4f}'.format(
                    i + 1, num_epochs, acc.Pixel_Accuracy(), acc.Pixel_Kappa(), acc.Pixel_Precision_Rate(),
                    acc.Pixel_Recall_Rate(), acc.Pixel_F1_score(), acc.Mean_Intersection_over_Union()[0],
                    acc.Mean_Intersection_over_Union()[1]))

            writer.add_scalar('g_loss', g_loss_aver, i + init_num_epochs_G)
            writer.add_scalar('d_loss', d_loss_aver, i + init_num_epochs_G)
            writer.add_scalar('s_loss', s_loss_aver, i + init_num_epochs_G)
            writer.add_scalar('s_d_loss', s_d_loss_aver, i + init_num_epochs_G)
            writer.add_scalar('l1_loss', l1_loss_aver, i + init_num_epochs_G)
            writer.add_scalar('r_loss', r_loss_aver, i + init_num_epochs_G)
            writer.add_scalar('generator_loss', generator_loss_aver, i + init_num_epochs_G)
            writer.add_scalar('perception_loss', perception_loss_aver, i + init_num_epochs_G)
            writer.add_scalar('ssim_loss', ssim_loss_aver, i + init_num_epochs_G)

            writer.add_scalar('Overall Accuracy:', acc.Pixel_Accuracy(), i + init_num_epochs_G)
            writer.add_scalar('Precision Rate', acc.Pixel_Precision_Rate(), i + init_num_epochs_G)
            writer.add_scalar('Recall Rate', acc.Pixel_Recall_Rate(), i + init_num_epochs_G)
            writer.add_scalar('F1', acc.Pixel_F1_score(), i + init_num_epochs_G)
            writer.add_scalar('mIOU', acc.Mean_Intersection_over_Union()[0], i + init_num_epochs_G)
            writer.add_scalar('cIOU', acc.Mean_Intersection_over_Union()[1], i + init_num_epochs_G)


            # accuracy assessment for testing set
            process_num = 0
            acc.reset()

            for data_array in test_dataloader:

                x = data_array[0]
                y = data_array[1]
                item = data_array[2]
                ref = data_array[3]
                region = data_array[4]

                x = x.to(device)
                y = y.to(device)
                region = region.to(device)

                cmap = netS(x, y)

                cmask = torch.zeros_like(cmap)
                cmask[cmap > prob_thresh] = 1

                for ns in range(x.size(0)):

                    change_mask = cmask[ns][0]
                    change_mask = change_mask.cpu().numpy()
                    ref_mask = ref[ns][0].numpy()

                    acc_range = dataset.EffRange(item[ns].numpy())
                    acc.add_batch_map(ref_mask[acc_range[0]:acc_range[1], acc_range[2]:acc_range[3]].astype(np.int16),
                                      change_mask[acc_range[0]:acc_range[1], acc_range[2]:acc_range[3]].astype(np.int16), gt_map,
                                      pre_map)

                process_num += x.size()[0]
                print('\rProcessing batch: {}/{}'.format(process_num, total_test_dataset_size), end='', flush=True)

            print('\r', end='', flush=True)

            print(
                'Test Dataset: Overall Accuracy: {:.4f}, Kappa: {:.4f}, Precision Rate: {:.4f}, Recall Rate: {:.4f}, F1:{:.4f}, mIOU:{:.4f}, cIoU:{:.4f}'.format(
                    acc.Pixel_Accuracy(), acc.Pixel_Kappa(), acc.Pixel_Precision_Rate(),
                    acc.Pixel_Recall_Rate(), acc.Pixel_F1_score(), acc.Mean_Intersection_over_Union()[0],
                    acc.Mean_Intersection_over_Union()[1]))

            writer.add_scalar('Test Overall Accuracy:', acc.Pixel_Accuracy(), i + init_num_epochs_G)
            writer.add_scalar('Test Precision Rate', acc.Pixel_Precision_Rate(), i + init_num_epochs_G)
            writer.add_scalar('Test Recall Rate', acc.Pixel_Recall_Rate(), i + init_num_epochs_G)
            writer.add_scalar('Test F1', acc.Pixel_F1_score(), i + init_num_epochs_G)
            writer.add_scalar('Test mIOU', acc.Mean_Intersection_over_Union()[0], i + init_num_epochs_G)
            writer.add_scalar('Test cIOU', acc.Mean_Intersection_over_Union()[1], i + init_num_epochs_G)

    # finally, output a change map

    netS.eval()
    netD.eval()

    print("Saving Change Map and Model")

    print("Segmentation of Change")
    with torch.no_grad():

        process_num = 0
        acc.reset()

        for data_array in test_dataloader:

            x = data_array[0]
            y = data_array[1]
            item = data_array[2]
            ref = data_array[3]
            region = data_array[4]

            x = x.to(device)
            y = y.to(device)
            region = region.to(device)

            cmap = netS(x, y)

            cmask = torch.zeros_like(cmap)
            cmask[cmap > prob_thresh] = 1

            for ns in range(x.size(0)):
                write_cmap = cmap[ns].cpu().numpy()
                test_dataset.GDALwrite(write_cmap, item[ns].numpy(), filterName="{}{}".format(outName_density, extName))

                change_mask = cmask[ns]
                change_mask = change_mask.cpu().numpy()
                ref_mask = ref[ns].numpy()
                write_cmask = write_changemap_gdal(change_mask, ref_mask, write_color=write_color, ref_map=gt_map, dt_map=pre_map)
                test_dataset.GDALwrite(write_cmask, item[ns].numpy(), filterName="{}{}".format(outName_binary, extName))

                acc_range = test_dataset.EffRange(item[ns].numpy())
                acc.add_batch_map(ref_mask[0, acc_range[0]:acc_range[1], acc_range[2]:acc_range[3]].astype(np.int16),
                                  change_mask[0, acc_range[0]:acc_range[1], acc_range[2]:acc_range[3]].astype(np.int16), gt_map, pre_map)

            process_num += x.size()[0]
            print('\rProcessing batch: {}/{}'.format(process_num, total_test_dataset_size), end='', flush=True)

        print('\r', end='', flush=True)

        print(
            'Overall Accuracy: {:.4f}, Kappa: {:.4f}, Precision Rate: {:.4f}, Recall Rate: {:.4f}, F1:{:.4f}, mIOU:{:.4f}, cIoU:{:.4f}'.format(
                acc.Pixel_Accuracy(), acc.Pixel_Kappa(), acc.Pixel_Precision_Rate(),
                acc.Pixel_Recall_Rate(), acc.Pixel_F1_score(), acc.Mean_Intersection_over_Union()[0],
                acc.Mean_Intersection_over_Union()[1]))

    print('\r' + 'End of Saving', flush=True)

    # saving model
    path = os.path.join(OutDir, 'SModel.pkl')
    torch.save(netS.state_dict(), path)

    path = os.path.join(OutGModelDir, 'GModel.pkl')
    torch.save(netG.state_dict(), path)

    path = os.path.join(OutDir, 'DModel.pkl')
    torch.save(netD.state_dict(), path)

    writer.close()

    # saving the parameter settings with a txt file
    ParaTxtPath = os.path.join(OutDir, 'Para.txt')
    TxtFile = open(ParaTxtPath, 'w')
    TxtFile.write("perception_weight:{}\n".format(perception_weight))
    TxtFile.write("ssim_weight:{}\n".format(ssim_weight))
    TxtFile.write("perception_perBand:{}\n".format(perception_perBand))
    TxtFile.write("perception_layer:{}\n".format(perception_layer))
    TxtFile.write("l1_weight:{}\n".format(l1_weight))
    TxtFile.write("g_weight:{}\n".format(g_weight))
    TxtFile.write("d_weight:{}\n".format(d_weight))
    TxtFile.write("r_weight:{}\n".format(r_weight))
    TxtFile.write("discriminator_continuous:{}\n".format(discriminator_continuous))
    TxtFile.write("prob_thresh:{}\n".format(prob_thresh))
    TxtFile.write(
        "Segmentation, Overall Accuracy: {:.4f}, Kappa: {:.4f}, Precision Rate: {:.4f}, Recall Rate: {:.4f}, F1:{:.4f}, mIOU:{:.4f}, cIOU:{:.4f}\n".format(
            acc.Pixel_Accuracy(), acc.Pixel_Kappa(), acc.Pixel_Precision_Rate(),
            acc.Pixel_Recall_Rate(), acc.Pixel_F1_score(), acc.Mean_Intersection_over_Union()[0],
            acc.Mean_Intersection_over_Union()[1]))
    TxtFile.write("tips:{}\n".format(tips))

    TxtFile.close()

