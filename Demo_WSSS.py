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

    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

    init_num_epochs_G = 50  # initial epochs for generator training
    num_epochs = 50 # epochs for adversarial training
    learning_rate = 0.0005 # learning rate
    # learning_rate = 3e-4
    unc_batch_size = 50 # batch size for testing
    batch_size = 15 # batch size for training
    prob_thresh = 0.6   # probability to separate changes, mostly 0.5
    tips = 'train'  # tips to record txt file, not used in algorithm

    ################parameter settings

    # parameter for predictor
    perception_weight = 0.5
    ssim_weight = 0
    perception_perBand = False
    perception_layer = 1

    # parameter for GAN
    g_weight = 0.2  # weight for generator
    l1_weight = 1.6 # weight for l1-loss
    d_weight = 1    # weight for discriminator loss, mostly 1
    nc_weight = 1.5  # weight for change map in unchanged images

    write_grey = True   # switch to write grey-scale image, i.e. change probability map
    write_color = True  # different color to indicate TP/TN/FP/FN
    modelG_reuse = True # reuse of trained generator

    discriminator_continuous = True # soft or hard change map for optimization

    ImgDirX = r'/data/chen.wu/data/ChangeNet/Building/Building CD Slice Dataset/before'
    ImgDirY = r'/data/chen.wu/data/ChangeNet/Building/Building CD Slice Dataset/after'
    RefDir = r'/data/chen.wu/data/ChangeNet/Building/Building CD Slice Dataset/Label'
    LabelDir = r'/data/chen.wu/data/ChangeNet/Building/Building CD Slice Dataset'
    OutGModelDir = r'/data/chen.wu/data/ChangeNet/Building/Building CD Slice Dataset/GModel'
    extName = '_l1w05_nl1w15_norm_github'
    OutDir = r'/data/chen.wu/data/ChangeNet/Building/Building CD Slice Dataset/Detection_WSS{}'.format(extName)

    writer = SummaryWriter(comment='Building_WSSS{}'.format(extName))

    # calculate and record normalization parameter
    statsName = 'stats'
    dataset = WHU_Dataset(ImgDirX, ImgDirY, RefDir, LabelDir, label_selected='-1')
    # scale_list1, scale_list2 = Dataset_maxmin(statsPath1, statsPath2, dataset)
    statsPath1 = os.path.join(ImgDirX, '{}_meanstd.txt'.format(statsName))
    statsPath2 = os.path.join(ImgDirY, '{}_meanstd.txt'.format(statsName))
    meanX, stdX, meanY, stdY = Dataset_meanstd(statsPath1, statsPath2, dataset)
    # scaler = SCALE(scale_list1, scale_list2)
    scaler = NORMALIZE(meanX, stdX, meanY, stdY)

    # data augmentation, not used in this algorithm
    # eraser = RANDOM_ERASER()
    # mask_generate = RANDOM_ERASER_MULTI_REGION()

    dataset = WHU_Dataset_WSS(ImgDirX, ImgDirY, RefDir, LabelDir, scale=scaler, random_assign=False)
    total_dataset_size = dataset.__len__()
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # dataset for unchanged samples
    # used for training generator
    unc_dataset = WHU_Dataset(ImgDirX, ImgDirY, RefDir, LabelDir, scale=scaler, label_selected='0')
    total_unc_dataset_size = unc_dataset.__len__()
    unc_dataloader = DataLoader(unc_dataset, batch_size=unc_batch_size, shuffle=True)

    if os.path.exists(OutDir) == False:
        os.mkdir(OutDir)

    if write_grey == True:
        OutDensityDir = OutDir + "_Density"
        if os.path.exists(OutDensityDir) == False:
            os.mkdir(OutDensityDir)

    # model training
    netD = Discriminator_SRGAN_simple()
    netD.to(device)

    netS = Segmentor(n_channels=3, bilinear=True)
    netS.to(device)

    netG = Generator(n_channels=3)
    netG.to(device)

    netS.train()
    netD.train()
    netG.train()

    optimizerG = torch.optim.Adam(netG.parameters(), lr=learning_rate, betas=(0.9, 0.99))
    # optimizerS = torch.optim.Adam(netS.parameters(), lr=learning_rate, betas=(0.9, 0.99))
    # optimizerD = torch.optim.Adam(netD.parameters(), lr=learning_rate, betas=(0.9, 0.99))

    # RMSprop optimizer, according to WGAN
    optimizerS = torch.optim.RMSprop(netS.parameters(), lr=1e-3)
    optimizerD = torch.optim.RMSprop(netD.parameters(), lr=1e-5)

    # accuracy assessment
    acc = Evaluator(num_class=2)

    g_criterion = CGeneratorLoss(perception_layer=perception_layer, perception_perBand=False)
    g_criterion.to(device)

    # model reuse of generator
    if modelG_reuse == True:
        path = os.path.join(OutGModelDir, 'GModel.pkl')
        if os.path.exists(path) == True:
            init_num_epochs_G = 0
            netG.load_state_dict(torch.load(path))

    if g_weight == 0:
        init_num_epochs_G = 0

    print('Start Generator Training')
    with torch.enable_grad():
        acc.reset()
        for i in range(init_num_epochs_G):
            g_loss_aver = 0
            generator_loss_aver = 0
            perception_loss_aver = 0
            ssim_loss_aver = 0
            adjust_learning_rate(optimizerG, i, lr_start=1e-5, lr_max=3e-4, lr_warm_up_epoch=10, lr_sustain_epochs=10)

            process_num = 0

            for data_array in unc_dataloader:
                time_start = time.time()

                # Update G network:

                optimizerG.zero_grad()

                x = data_array[0]
                y = data_array[1]

                x = x.to(device)
                y = y.to(device)

                # gpu_tracker.track()

                y_fake = netG(x)
                cmap = torch.zeros((x.size()[0], 1, x.size()[2], x.size()[3]))
                cmap = cmap.to(device)

                generator_loss, ssim_loss, perception_loss = g_criterion(y, y_fake, cmap)

                g_loss = generator_loss + perception_weight * perception_loss + ssim_weight * ssim_loss

                g_loss.backward()
                optimizerG.step()

                g_loss_aver += g_loss.item() * x.size(0) / total_unc_dataset_size
                generator_loss_aver += generator_loss.item() * x.size(0) / total_unc_dataset_size
                perception_loss_aver += perception_loss.item() * x.size(0) / total_unc_dataset_size
                ssim_loss_aver += ssim_loss.item() * x.size(0) / total_unc_dataset_size

                process_num += x.size()[0]

                time_end = time.time()
                time_per_iter = (time_end - time_start) / x.size()[0] * total_unc_dataset_size
                time_remaining = time_per_iter * (
                            (init_num_epochs_G - 1 - i) + (1 - process_num / total_unc_dataset_size))
                time_desc_per = time_show(time_per_iter)
                time_desc = time_show(time_remaining)

                print('\rProcessing batch: {}/{}; Processing speed per iter: {}; Processing time remaining: {}'.format(
                    process_num, total_unc_dataset_size, time_desc_per, time_desc), end='', flush=True)

            print('\r', end='', flush=True)

            print(
                'Epochs: {}/{}, g_loss: {:.4f}, generator_loss: {:.4f}, perception_loss:{:.4f}, ssim_loss:{:.4f}'.format(
                    i + 1, init_num_epochs_G, g_loss_aver, generator_loss_aver, perception_loss_aver, ssim_loss_aver))

            writer.add_scalar('g_loss', g_loss_aver, i)
            writer.add_scalar('generator_loss', generator_loss_aver, i)
            writer.add_scalar('perception_loss', perception_loss_aver, i)
            writer.add_scalar('ssim_loss', ssim_loss_aver, i)

    netG.eval()

    print('Start Adversarial Training')
    with torch.enable_grad():

        for i in range(num_epochs):
            s_loss_aver = 0
            s_d_loss_aver = 0
            d_loss_aver = 0
            l1_loss_aver = 0
            nc_loss_aver = 0
            g_loss_aver = 0

            generator_loss_aver = 0
            perception_loss_aver = 0
            ssim_loss_aver = 0

            acc.reset()

            # warm-up strategy
            adjust_learning_rate(optimizerS, i, lr_start=1e-4, lr_max=1e-3, lr_warm_up_epoch=5)
            adjust_learning_rate(optimizerD, i, lr_start=1e-6, lr_max=1e-5, lr_min=1e-8, lr_warm_up_epoch=5)

            process_num = 0

            # since the changed pairs and unchanged pairs have unequal quantity, and have no specific orders
            # for random optimization, dataset.order_reset() is used to reorder the links between changed pairs and unchange pairs
            dataset.order_reset()

            for data_array in train_dataloader:

                time_start = time.time()

                # changed pairs and unchanged pairs
                cds_data = data_array[0]
                ncds_data = data_array[1]

                ##################
                # Update D network
                ##################

                # Calculate loss of CHANGED images

                x = cds_data[0]
                y = cds_data[1]
                ref = cds_data[2]

                x = x.to(device)
                y = y.to(device)

                cmap = netS(x, y)
                if discriminator_continuous == True:
                    cmask = cmap
                else:
                    cmask = (torch.sign(cmap - 0.5) + 1) / 2
                x_mask = x * (1 - cmask.repeat((1, x.size()[1], 1, 1)))
                y_mask = y * (1 - cmask.repeat((1, y.size()[1], 1, 1)))

                c_out = netD(x_mask, y_mask)

                # Calculate loss of UNCHANGED images
                x_nc = ncds_data[0]
                y_nc = ncds_data[1]

                x_nc = x_nc.to(device)
                y_nc = y_nc.to(device)

                ncmap = netS(x_nc, y_nc)

                # even for unchanged samples, they should be masked by change maps
                # we all know that, unchanged samples can be seen as unchanged with any masks
                # without this process, the adversarial process is hard to converge
                x_mask_nc = x_nc * (1 - cmask.repeat((1, x_nc.size()[1], 1, 1)))
                y_mask_nc = y_nc * (1 - cmask.repeat((1, y_nc.size()[1], 1, 1)))

                nc_out = netD(x_mask_nc, y_mask_nc)

                optimizerD.zero_grad()
                d_loss = 1 + nc_out.mean() - c_out.mean()
                d_loss.backward(retain_graph=True)
                # d_loss.backward()
                optimizerD.step()

                # Clip weights of discriminator
                # for p in netD.parameters():
                #     p.data.clamp_(-1, 1)

                ##################
                # Update S network
                ##################

                # when detected in unchanged pairs, the results should be all zero
                # thus we call it nc_loss,
                nc_loss = torch.mean(torch.pow(ncmap, 2))

                # rebuild the graph
                c_out = netD(x_mask, y_mask)

                # generator loss
                if g_weight != 0:
                    y_fake = netG(x)
                    generator_loss, ssim_loss, perception_loss = g_criterion(y, y_fake, cmap)
                else:
                    generator_loss = torch.Tensor([0]).to(device)
                    ssim_loss = torch.Tensor([0]).to(device)
                    perception_loss = torch.Tensor([0]).to(device)
                g_loss = generator_loss + perception_weight * perception_loss + ssim_weight * ssim_loss

                # l1-loss to avoid constant solution
                l1_loss = torch.mean(abs(cmap))

                s_d_loss = c_out.mean()

                s_loss = d_weight * s_d_loss + l1_weight * l1_loss + g_weight * g_loss + nc_weight * nc_loss

                optimizerS.zero_grad()
                s_loss.backward()
                optimizerS.step()

                d_loss_aver += d_loss.item() * x.size(0) / total_dataset_size
                s_d_loss_aver += s_d_loss.item() * x.size(0) / total_dataset_size
                g_loss_aver += g_loss.item() * x.size(0) / total_dataset_size
                s_loss_aver += s_loss.item() * x.size(0) / total_dataset_size
                l1_loss_aver += l1_loss.item() * x.size(0) / total_dataset_size
                nc_loss_aver += nc_loss.item() * x.size(0) / total_dataset_size

                generator_loss_aver += generator_loss.item() * x.size(0) / total_dataset_size
                ssim_loss_aver += ssim_loss.item() * x.size(0) / total_dataset_size
                perception_loss_aver += perception_loss.item() * x.size(0) / total_dataset_size

                # accuracy assessment during the optimization
                cmask = torch.zeros_like(cmap)
                cmask[cmap > prob_thresh] = 1
                for ns in range(x.size(0)):
                    change_mask = cmask[ns][0]
                    change_mask = change_mask.cpu().numpy()
                    ref_mask = ref[ns][0].numpy()
                    acc.add_batch(ref_mask.astype(np.int16), change_mask.astype(np.int16))

                process_num += x.size()[0]

                time_end = time.time()
                time_per_iter = (time_end - time_start) / x.size()[0] * total_dataset_size
                time_remaining = time_per_iter * (
                            (num_epochs - 1 - i) + (1 - process_num / total_dataset_size))
                time_desc_per = time_show(time_per_iter)
                time_desc = time_show(time_remaining)

                print('\rProcessing batch: {}/{}; Processing speed per iter: {}; Processing time remaining: {}'.format(
                    process_num, total_dataset_size, time_desc_per, time_desc), end='', flush=True)

            print('\r', end='', flush=True)

            print(
                'Epochs: {}/{}, d_loss: {:.4f}, g_loss: {:.4f}, s_loss: {:.4f}, l1_loss:{:.4f}, nc_loss:{:.4f}, s_d_loss: {:.4f}'.format(
                    i + 1, num_epochs, d_loss_aver, g_loss_aver, s_loss_aver, l1_loss_aver, nc_loss_aver,
                    s_d_loss_aver))
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
            writer.add_scalar('nc_loss', nc_loss_aver, i + init_num_epochs_G)
            writer.add_scalar('generator_loss', generator_loss_aver, i + init_num_epochs_G)
            writer.add_scalar('perception_loss', perception_loss_aver, i + init_num_epochs_G)
            writer.add_scalar('ssim_loss', ssim_loss_aver, i + init_num_epochs_G)

            writer.add_scalar('Overall Accuracy:', acc.Pixel_Accuracy(), i + init_num_epochs_G)
            writer.add_scalar('Kappa Coefficient:', acc.Pixel_Kappa(), i + init_num_epochs_G)
            writer.add_scalar('Precision Rate', acc.Pixel_Precision_Rate(), i + init_num_epochs_G)
            writer.add_scalar('Recall Rate', acc.Pixel_Recall_Rate(), i + init_num_epochs_G)
            writer.add_scalar('F1', acc.Pixel_F1_score(), i + init_num_epochs_G)
            writer.add_scalar('mIOU', acc.Mean_Intersection_over_Union()[0], i + init_num_epochs_G)
            writer.add_scalar('cIOU', acc.Mean_Intersection_over_Union()[1], i + init_num_epochs_G)

    # generate the final result

    # as we found, the train mode can get a better performance
    # netS.eval()
    # netD.eval()

    c_dataset = WHU_Dataset(ImgDirX, ImgDirY, RefDir, LabelDir, scale=scaler, label_selected='1')
    test_dataloader = DataLoader(c_dataset, batch_size=batch_size, shuffle=False)

    print("Saving Change Map and Model")

    print("Segmentation of Change")
    with torch.no_grad():

        process_num = 0

        acc.reset()

        for data_array in test_dataloader:

            x = data_array[0]
            y = data_array[1]
            ref = data_array[2]
            item = data_array[3]
            label = data_array[4]

            x = x.to(device)
            y = y.to(device)

            cmap = netS(x, y)

            cmask = torch.zeros_like(cmap)
            cmask[cmap > prob_thresh] = 1

            for ns in range(x.size(0)):
                change_mask = cmask[ns][0]
                change_mask = change_mask.cpu().numpy()
                ref_mask = ref[ns][0].numpy()
                outPath = os.path.join(OutDir, c_dataset.getFileName(item[ns].item()))
                change_write = write_changemap(change_mask, ref_mask, write_color=write_color)

                acc.add_batch(ref_mask.astype(np.int16), change_mask.astype(np.int16))

                if write_grey == True:
                    change_mask = cmap[ns][0]
                    change_mask = change_mask.cpu().numpy()
                    change_write_density = np.zeros((change_mask.shape[0], change_mask.shape[1]))
                    change_write_density = change_mask * 255
                    change_write_density = Image.fromarray(np.uint8(change_write_density))
                    OutDensityPath = os.path.join(OutDensityDir, c_dataset.getFileName(item[ns].item()))
                    change_write_density.save(OutDensityPath)

                if len(change_write.shape) == 3:
                    change_write = change_write.transpose((1, 2, 0))
                change_write = Image.fromarray(np.uint8(change_write))
                change_write.save(outPath)

            process_num += x.size()[0]
            print('\rProcessing batch: {}/{}'.format(process_num, total_dataset_size), end='', flush=True)

    print(
        '\rSegmentation, Overall Accuracy: {:.4f}, Kappa: {:.4f}, Precision Rate: {:.4f}, Recall Rate: {:.4f}, F1:{:.4f}, mIOU:{:.4f}, cIOU:{:.4f}'.format(
            acc.Pixel_Accuracy(), acc.Pixel_Kappa(), acc.Pixel_Precision_Rate(),
            acc.Pixel_Recall_Rate(), acc.Pixel_F1_score(), acc.Mean_Intersection_over_Union()[0], acc.Mean_Intersection_over_Union()[1]))

    print('\r' + 'End of Saving', flush=True)

    path = os.path.join(OutDir, 'SModel.pkl')
    torch.save(netS.state_dict(), path)

    path = os.path.join(OutGModelDir, 'GModel.pkl')
    torch.save(netG.state_dict(), path)

    path = os.path.join(OutDir, 'DModel.pkl')
    torch.save(netD.state_dict(), path)

    writer.close()

    ParaTxtPath = os.path.join(OutDir, 'Para.txt')
    TxtFile = open(ParaTxtPath, 'w')
    TxtFile.write("perception_weight:{}\n".format(perception_weight))
    TxtFile.write("ssim_weight:{}\n".format(ssim_weight))
    TxtFile.write("perception_perBand:{}\n".format(perception_perBand))
    TxtFile.write("perception_layer:{}\n".format(perception_layer))
    TxtFile.write("l1_weight:{}\n".format(l1_weight))
    TxtFile.write("nc_weight:{}\n".format(nc_weight))
    TxtFile.write("d_weight:{}\n".format(d_weight))
    TxtFile.write("g_weight:{}\n".format(g_weight))
    TxtFile.write("discriminator_continuous:{}\n".format(discriminator_continuous))
    TxtFile.write("prob_thresh:{}\n".format(prob_thresh))
    TxtFile.write("Segmentation, Overall Accuracy: {:.4f}, Kappa: {:.4f}, Precision Rate: {:.4f}, Recall Rate: {:.4f}, F1:{:.4f}, mIOU:{:.4f}, cIOU:{:.4f}\n".format(
            acc.Pixel_Accuracy(), acc.Pixel_Kappa(), acc.Pixel_Precision_Rate(),
            acc.Pixel_Recall_Rate(), acc.Pixel_F1_score(), acc.Mean_Intersection_over_Union()[0], acc.Mean_Intersection_over_Union()[1]))
    TxtFile.write("tips:{}\n".format(tips))

    TxtFile.close()

