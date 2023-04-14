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

# code for unsupervised change detection

if __name__ == '__main__':

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    init_num_epochs_G = 50  # initial training epochs for generator
    init_num_epochs_S = 50  # initial training epochs for segmentor
    num_epochs = 100  #  training epochs for iteration
    learning_rate = 0.0002
    batch_size = 10

    # parameter settings for model
    perception_weight = 0.4
    l1_weight = 0.65
    ssim_weight = 0
    perception_perBand = True
    perception_layer = 1

    # input path
    dir = r'/data'
    ImageXName = 'T1.tif'
    ImageYName = 'T2.tif'
    RefName = 'ref.tif'

    # output path
    outdir = dir
    ext = '_l1w065_pw04_github' # only used to label the output result with different file name
    CMapName = 'ChangeDensity{}'.format(ext)
    # a txt file to record the mean/std of the image
    statsName = 'stats'

    # for a large-scale image, this code will slice the image with 'patch_size', each patch will has a overlap padding
    # in the prediction, only the centering patch (220 - 2 * 10, 220 - 2 * 10) = (200, 200) is used to avoid the problem in edge
    patch_size = (220, 220)
    overlap_padding = (10, 10)
    # the label to indicate change/non-change in reference (ground truth) and prediction change map for convenience
    gt_map = [1, 2]
    pre_map = [0, 1]
    # the threshold to segment the prediction probability, mostly 0.5
    prob_thresh = 0.5

    # use different color to indicate tp / tn / fp / fn
    write_color = True
    discriminator_continuous = True
    # a tips to record experiment settings in a txt file
    tips = 'eval_patch'

    # tensorboard to record experiment
    writer = SummaryWriter(comment='USSS{}'.format(ext))

    ImgXPath = os.path.join(dir, ImageXName)
    ImgYPath = os.path.join(dir, ImageYName)
    FileName1, ext1 = os.path.splitext(ImageXName)
    FileName2, ext2 = os.path.splitext(ImageYName)
    outFileName = CMapName + ext1
    OutPath = os.path.join(outdir, outFileName)
    RefPath = os.path.join(dir, RefName)
    OutColorPath = os.path.join(outdir, "{}_acc_color{}".format(CMapName, ext1))

    # read the image to calculate the mean/std
    dataset = GDALDataset(ImgXPath, ImgYPath, outPath=OutPath, patch_size=patch_size,
                          overlap_padding=(0, 0))

    statsPath1 = os.path.join(dir, '{}_{}.txt'.format(FileName1, statsName))
    statsPath2 = os.path.join(dir, '{}_{}.txt'.format(FileName2, statsName))
    meanX, stdX, meanY, stdY = Dataset_meanstd(statsPath1, statsPath2, dataset)
    # normalize the input image
    scaler = NORMALIZE(meanX, stdX, meanY, stdY)

    # data loader
    dataset = GDALDataset(ImgXPath, ImgYPath, refPath=RefPath, outPath=OutPath, enhance=scaler, patch_size=patch_size, overlap_padding=overlap_padding)
    total_dataset_size = dataset.__len__()
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    xitem_count, yitem_count = dataset.patch_count()
    pad = dataset.overlap_padding
    xsize, ysize, nband = dataset.size()

    # accuracy evaluation
    acc = Evaluator(num_class=len(gt_map))

    # model
    netS = Segmentor(n_channels=nband, bilinear=True)
    netS.to(device)

    netG = Generator(n_channels=nband)
    netG.to(device)

    netS.train()
    netG.train()

    criterion = CNetLoss(channel=nband, perception_layer=perception_layer, perception_perBand=perception_perBand)
    criterion.to(device)
    optimizerS = torch.optim.Adam(netS.parameters(), lr=learning_rate, betas=(0.9, 0.99))
    optimizerG = torch.optim.Adam(netG.parameters(), lr=learning_rate, betas=(0.9, 0.99))

    print('Start Initial Generator Training')
    with torch.enable_grad():
        for i in range(init_num_epochs_G):
            NetLoss_aver = 0
            generator_loss_aver = 0
            l1_loss_aver = 0
            perception_loss_aver = 0
            ssim_loss_aver = 0
            # warm-up strategy
            adjust_learning_rate(optimizerG, i, lr_start=1e-5, lr_max=3e-4, lr_warm_up_epoch=10, lr_sustain_epochs=10)

            process_num = 0

            for data_array in train_dataloader:
                time_start = time.time()

                # Update G network:

                optimizerG.zero_grad()

                x = data_array[0]
                y = data_array[1]

                x = x.to(device)
                y = y.to(device)

                y_fake = netG(x)
                cmap = torch.zeros((x.size()[0], 1, x.size()[2], x.size()[3]))
                cmap = cmap.to(device)

                generator_loss, l1_loss, perception_loss, ssim_loss = criterion(y, y_fake, cmap)

                Loss = generator_loss + perception_weight * perception_loss + ssim_weight * ssim_loss

                Loss.backward()
                optimizerG.step()

                NetLoss_aver += Loss * x.size(0) / total_dataset_size
                generator_loss_aver += generator_loss * x.size(0) / total_dataset_size
                l1_loss_aver += l1_loss * x.size(0) / total_dataset_size
                perception_loss_aver += perception_loss * x.size(0) / total_dataset_size
                ssim_loss_aver += ssim_loss * x.size(0) / total_dataset_size

                process_num += x.size()[0]
                time_end = time.time()
                time_per_iter = (time_end - time_start) / x.size()[0] * total_dataset_size
                time_remaining = time_per_iter * (
                        (init_num_epochs_G - 1 - i) + (1 - process_num / total_dataset_size))
                time_desc_per = time_show(time_per_iter)
                time_desc = time_show(time_remaining)

                print('\rProcessing batch: {}/{}; Processing speed per iter: {}; Processing time remaining: {}'.format(
                    process_num, total_dataset_size, time_desc_per, time_desc), end='', flush=True)

            print('\r', end='', flush=True)

            print(
                'Epochs: {}/{}, NetLoss Loss: {:.4f}, generator_loss Loss: {:.4f}, l1_loss Loss: {:.4f}, perception_loss:{:.4f}, ssim_loss:{:.4f}'.format(
                    i + 1, init_num_epochs_G, NetLoss_aver, generator_loss_aver, l1_loss_aver, perception_loss_aver,
                    ssim_loss_aver))

            writer.add_scalar('NetLoss', NetLoss_aver, i)
            writer.add_scalar('generator_loss', generator_loss_aver, i)
            writer.add_scalar('l1_loss', l1_loss_aver, i)
            writer.add_scalar('perception_loss', perception_loss_aver, i)
            writer.add_scalar('ssim_loss', ssim_loss_aver, i)


    print('Start Initial Segmentor Training')
    with torch.enable_grad():
        for i in range(init_num_epochs_S):
            NetLoss_aver = 0
            generator_loss_aver = 0
            l1_loss_aver = 0
            perception_loss_aver = 0
            ssim_loss_aver = 0

            adjust_learning_rate(optimizerS, i, lr_start=1e-5, lr_max=3e-4, lr_warm_up_epoch=10, lr_sustain_epochs=10)

            acc.reset()

            process_num = 0

            for data_array in train_dataloader:

                time_start = time.time()

                x = data_array[0]
                y = data_array[1]
                item = data_array[2]
                ref = data_array[3]

                x = x.to(device)
                y = y.to(device)

                y_fake = netG(x)
                cmap = netS(x, y)

                generator_loss, l1_loss, perception_loss, ssim_loss = criterion(y, y_fake, cmap)

                NetLoss = generator_loss + l1_weight * l1_loss + perception_weight * perception_loss + ssim_weight * ssim_loss
                optimizerS.zero_grad()
                NetLoss.backward()

                optimizerS.step()

                NetLoss_aver += NetLoss * x.size(0) / total_dataset_size
                generator_loss_aver += generator_loss * x.size(0) / total_dataset_size
                l1_loss_aver += l1_loss * x.size(0) / total_dataset_size
                perception_loss_aver += perception_loss * x.size(0) / total_dataset_size
                ssim_loss_aver += ssim_loss * x.size(0) / total_dataset_size

                cmask = torch.zeros_like(cmap)
                cmask[cmap > prob_thresh] = 1
                for ns in range(x.size(0)):
                    change_mask = cmask[ns][0]
                    change_mask = change_mask.cpu().numpy()
                    ref_mask = ref[ns][0].numpy()

                    item_x = math.floor(item[ns].numpy() / yitem_count)
                    item_y = item[ns].numpy() % yitem_count
                    slice, _, _ = dataset.slice_assign(item_x, item_y)

                    # accuracy evaluation only with the centering region of the patch
                    acc.add_batch_map(ref_mask[pad[1]:pad[1] + slice[3], pad[0]:pad[0] + slice[2]].astype(np.int16), change_mask[pad[1]:pad[1] + slice[3], pad[0]:pad[0] + slice[2]].astype(np.int16), gt_map, pre_map)

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
                'Epochs: {}/{}, NetLoss Loss: {:.4f}, generator_loss Loss: {:.4f}, l1_loss Loss: {:.4f}, perception_loss:{:.4f}, ssim_loss:{:.4f}'.format(
                    i + 1, init_num_epochs_S, NetLoss_aver, generator_loss_aver, l1_loss_aver, perception_loss_aver,
                    ssim_loss_aver))

            print(
                'Epochs: {}/{}, Overall Accuracy: {:.4f}, Kappa: {:.4f}, Precision Rate: {:.4f}, Recall Rate: {:.4f}, F1:{:.4f}, mIOU:{:.4f}, cIoU:{:.4f}'.format(
                    i + 1, init_num_epochs_S, acc.Pixel_Accuracy(), acc.Pixel_Kappa(), acc.Pixel_Precision_Rate(),
                    acc.Pixel_Recall_Rate(), acc.Pixel_F1_score(), acc.Mean_Intersection_over_Union()[0],
                    acc.Mean_Intersection_over_Union()[1]))

            writer.add_scalar('NetLoss', NetLoss_aver, i + init_num_epochs_G)
            writer.add_scalar('generator_loss', generator_loss_aver, i + init_num_epochs_G)
            writer.add_scalar('l1_loss', l1_loss_aver, i + init_num_epochs_G)
            writer.add_scalar('perception_loss', perception_loss_aver, i + init_num_epochs_G)
            writer.add_scalar('ssim_loss', ssim_loss_aver, i + init_num_epochs_G)

            writer.add_scalar('Overall Accuracy:', acc.Pixel_Accuracy(), i + init_num_epochs_G)
            writer.add_scalar('Precision Rate', acc.Pixel_Precision_Rate(), i + init_num_epochs_G)
            writer.add_scalar('Recall Rate', acc.Pixel_Recall_Rate(), i + init_num_epochs_G)
            writer.add_scalar('Kappa Coefficient:', acc.Pixel_Kappa(), i + init_num_epochs_G)
            writer.add_scalar('F1', acc.Pixel_F1_score(), i + init_num_epochs_G)
            writer.add_scalar('mIOU', acc.Mean_Intersection_over_Union()[0], i + init_num_epochs_G)
            writer.add_scalar('cIOU', acc.Mean_Intersection_over_Union()[1], i + init_num_epochs_G)


    print('Start Training')
    with torch.enable_grad():
        for i in range(num_epochs):
            NetLoss_aver = 0
            generator_loss_aver = 0
            l1_loss_aver = 0
            perception_loss_aver = 0
            ssim_loss_aver = 0

            adjust_learning_rate(optimizerS, i, lr_start=1e-5, lr_max=1e-4)
            adjust_learning_rate(optimizerG, i, lr_start=1e-5, lr_max=1e-4)

            acc.reset()

            process_num = 0

            for data_array in train_dataloader:
                time_start = time.time()

                # Update G network:

                optimizerG.zero_grad()

                x = data_array[0]
                y = data_array[1]
                item = data_array[2]
                ref = data_array[3]

                x = x.to(device)
                y = y.to(device)

                y_fake = netG(x)
                cmap = netS(x, y)

                generator_loss, l1_loss, perception_loss, ssim_loss = criterion(y, y_fake, cmap)

                Loss = generator_loss + perception_weight * perception_loss + ssim_weight * ssim_loss

                Loss.backward(retain_graph=True)
                # optimizerG.step()

                # Update S network:

                # y_fake = netG(x)
                # cmap = netS(x, y)
                # generator_loss, l1_loss, perception_loss, ssim_loss = criterion(y, y_fake, cmap)

                NetLoss = generator_loss + l1_weight * l1_loss + perception_weight * perception_loss + ssim_weight * ssim_loss
                optimizerS.zero_grad()
                NetLoss.backward()

                optimizerG.step()
                optimizerS.step()

                NetLoss_aver += NetLoss * x.size(0) / total_dataset_size
                generator_loss_aver += generator_loss * x.size(0) / total_dataset_size
                l1_loss_aver += l1_loss * x.size(0) / total_dataset_size
                perception_loss_aver += perception_loss * x.size(0) / total_dataset_size
                ssim_loss_aver += ssim_loss * x.size(0) / total_dataset_size

                cmask = torch.zeros_like(cmap)
                cmask[cmap > prob_thresh] = 1
                for ns in range(x.size(0)):
                    change_mask = cmask[ns][0]
                    change_mask = change_mask.cpu().numpy()
                    ref_mask = ref[ns][0].numpy()

                    item_x = math.floor(item[ns].numpy() / yitem_count)
                    item_y = item[ns].numpy() % yitem_count
                    slice, _, _ = dataset.slice_assign(item_x, item_y)

                    acc.add_batch_map(ref_mask[pad[1]:pad[1] + slice[3], pad[0]:pad[0] + slice[2]].astype(np.int16),
                                      change_mask[pad[1]:pad[1] + slice[3], pad[0]:pad[0] + slice[2]].astype(np.int16),
                                      gt_map, pre_map)

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
                'Epochs: {}/{}, NetLoss Loss: {:.4f}, generator_loss Loss: {:.4f}, l1_loss Loss: {:.4f}, perception_loss:{:.4f}, ssim_loss:{:.4f}'.format(
                    i + 1, num_epochs, NetLoss_aver, generator_loss_aver, l1_loss_aver, perception_loss_aver,
                    ssim_loss_aver))

            print(
                'Epochs: {}/{}, Overall Accuracy: {:.4f}, Kappa: {:.4f}, Precision Rate: {:.4f}, Recall Rate: {:.4f}, F1:{:.4f}, mIOU:{:.4f}, cIoU:{:.4f}'.format(
                    i + 1, num_epochs, acc.Pixel_Accuracy(), acc.Pixel_Kappa(), acc.Pixel_Precision_Rate(),
                    acc.Pixel_Recall_Rate(), acc.Pixel_F1_score(), acc.Mean_Intersection_over_Union()[0],
                    acc.Mean_Intersection_over_Union()[1]))

            writer.add_scalar('NetLoss', NetLoss_aver, i + init_num_epochs_G + init_num_epochs_S)
            writer.add_scalar('generator_loss', generator_loss_aver, i + init_num_epochs_G + init_num_epochs_S)
            writer.add_scalar('l1_loss', l1_loss_aver, i + init_num_epochs_G + init_num_epochs_S)
            writer.add_scalar('perception_loss', perception_loss_aver, i + init_num_epochs_G + init_num_epochs_S)
            writer.add_scalar('ssim_loss', ssim_loss_aver, i + init_num_epochs_G + init_num_epochs_S)

            writer.add_scalar('Overall Accuracy:', acc.Pixel_Accuracy(), i + init_num_epochs_G + init_num_epochs_S)
            writer.add_scalar('Precision Rate', acc.Pixel_Precision_Rate(), i + init_num_epochs_G + init_num_epochs_S)
            writer.add_scalar('Recall Rate', acc.Pixel_Recall_Rate(), i + init_num_epochs_G + init_num_epochs_S)
            writer.add_scalar('Kappa Coefficient:', acc.Pixel_Kappa(), i + init_num_epochs_G + init_num_epochs_S)
            writer.add_scalar('F1', acc.Pixel_F1_score(), i + init_num_epochs_G + init_num_epochs_S)
            writer.add_scalar('mIOU', acc.Mean_Intersection_over_Union()[0], i + init_num_epochs_G + init_num_epochs_S)
            writer.add_scalar('cIOU', acc.Mean_Intersection_over_Union()[1], i + init_num_epochs_G + init_num_epochs_S)

    # obtain the change map

    netS.eval()
    netG.eval()

    test_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    print("Saving Change Map and Model")

    outDS = None
    print("Segmentation of Change")
    with torch.no_grad():

        process_num = 0
        acc.reset()

        for data_array in test_dataloader:

            x = data_array[0]
            y = data_array[1]
            item = data_array[2]
            ref = data_array[3]

            x = x.to(device)
            y = y.to(device)

            cmap = netS(x, y)

            cmask = torch.zeros_like(cmap)
            cmask[cmap > prob_thresh] = 1

            for ns in range(x.size(0)):
                write_cmap = cmap[ns].cpu().numpy()
                dataset.GDALwriteDefault(write_cmap, item[ns].numpy())

                # generate a color map indicating FP / FN / TP / TN
                if write_color == True:
                    if outDS == None:
                        xsize, ysize, nband = dataset.size()
                        driver = dataset.imgDS_x.GetDriver()
                        outDS = driver.Create(OutColorPath, xsize, ysize, 1, gdal.GDT_Int32)
                        if outDS == None:
                            print("Cannot make a output raster")
                            sys.exit(0)

                        outDS.SetGeoTransform(dataset.imgDS_x.GetGeoTransform())
                        outDS.SetProjection(dataset.imgDS_x.GetProjection())

                    change_mask = cmask[ns]
                    change_mask = change_mask.cpu().numpy()
                    ref_mask = ref[ns].numpy()
                    write_cmask = write_changemap_gdal(change_mask, ref_mask, write_color=write_color, ref_map=gt_map, dt_map=pre_map)
                    dataset.GDALwrite(write_cmask.astype(np.int32), item[ns].numpy(), outDS)

                item_x = math.floor(item[ns].numpy() / yitem_count)
                item_y = item[ns].numpy() % yitem_count
                slice, _, _ = dataset.slice_assign(item_x, item_y)

                acc.add_batch_map(ref_mask[0, pad[1]:pad[1] + slice[3], pad[0]:pad[0] + slice[2]].astype(np.int16),
                                  change_mask[0, pad[1]:pad[1] + slice[3], pad[0]:pad[0] + slice[2]].astype(np.int16),
                                  gt_map, pre_map)

            process_num += x.size()[0]
            print('\rProcessing batch: {}/{}'.format(process_num, total_dataset_size), end='', flush=True)

        print('\r', end='', flush=True)

        print(
            'Overall Accuracy: {:.4f}, Kappa: {:.4f}, Precision Rate: {:.4f}, Recall Rate: {:.4f}, F1:{:.4f}, mIOU:{:.4f}, cIoU:{:.4f}'.format(
                acc.Pixel_Accuracy(), acc.Pixel_Kappa(), acc.Pixel_Precision_Rate(),
                acc.Pixel_Recall_Rate(), acc.Pixel_F1_score(), acc.Mean_Intersection_over_Union()[0],
                acc.Mean_Intersection_over_Union()[1]))

    print('\r' + 'End of Saving', flush=True)

    path = os.path.join(outdir, 'SModel{}.pkl'.format(ext))
    torch.save(netS.state_dict(), path)

    path = os.path.join(outdir, 'GModel{}.pkl'.format(ext))
    torch.save(netG.state_dict(), path)

    writer.close()

    ParaTxtPath = os.path.join(outdir,'Para_{}{}.txt'.format(time.strftime("%b%d%H%M", time.localtime()), ext))
    TxtFile = open(ParaTxtPath, 'w')
    TxtFile.write("perception_weight:{}\n".format(perception_weight))
    TxtFile.write("ssim_weight:{}\n".format(ssim_weight))
    TxtFile.write("perception_perBand:{}\n".format(perception_perBand))
    TxtFile.write("perception_layer:{}\n".format(perception_layer))
    TxtFile.write("l1_weight:{}\n".format(l1_weight))
    TxtFile.write("discriminator_continuous:{}\n".format(discriminator_continuous))
    TxtFile.write("prob_thresh:{}\n".format(prob_thresh))
    TxtFile.write(
        "Segmentation, Overall Accuracy: {:.4f}, Kappa: {:.4f}, Precision Rate: {:.4f}, Recall Rate: {:.4f}, F1:{:.4f}, mIOU:{:.4f}, cIOU:{:.4f}\n".format(
            acc.Pixel_Accuracy(), acc.Pixel_Kappa(), acc.Pixel_Precision_Rate(),
            acc.Pixel_Recall_Rate(), acc.Pixel_F1_score(), acc.Mean_Intersection_over_Union()[0],
            acc.Mean_Intersection_over_Union()[1]))
    TxtFile.write("tips:{}\n".format(tips))

    TxtFile.close()

