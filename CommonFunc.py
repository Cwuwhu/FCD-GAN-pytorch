import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as trans
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torch.nn.functional as F
from torchvision.models.vgg import vgg16

import numpy as np
import random
import math
import sys
import os
import gc

from osgeo import gdal
from osgeo import ogr
from osgeo import osr
from tqdm import tqdm


def adjust_learning_rate(optimizer, epoch, lr_start=1e-4, lr_max=1e-3, lr_min=1e-6, lr_warm_up_epoch=20,
                         lr_sustain_epochs=0, lr_exp_decay=0.8):
    # warm-up strategy
    if epoch < lr_warm_up_epoch:
        lr = (lr_max - lr_start) / lr_warm_up_epoch * epoch + lr_start
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    elif epoch < lr_warm_up_epoch + lr_sustain_epochs:
        lr = lr_max
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    else:
        lr = (lr_max - lr_min) * lr_exp_decay ** (epoch - lr_warm_up_epoch - lr_sustain_epochs) + lr_min
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

def write_changemap(change_mask, ref_mask, write_color=False):
    if write_color == True:
        change_write = np.zeros((3, change_mask.shape[0], change_mask.shape[1]))
        # miss detection
        idx = np.logical_and((change_mask == 0), (ref_mask == 1))
        change_write[2, idx] = 255
        # false detection
        idx = np.logical_and((change_mask == 1), (ref_mask == 0))
        change_write[0, idx] = 255
        # true detection
        idx = np.logical_and((change_mask == 1), (ref_mask == 1))
        change_write[0, idx] = 255
        change_write[1, idx] = 255
        change_write[2, idx] = 255
    else:
        change_write = np.zeros((change_mask.shape[0], change_mask.shape[1]))
        idx = change_mask == 1
        change_write[idx] = 255
    return change_write

def write_changemap_gdal(change_mask, ref_mask, write_color=False, ref_map=[0, 1], dt_map=[0, 1]):
    # output true/false positive/negative to be [0, 1, 2, 3]
    change_write = np.zeros((1, change_mask.shape[1], change_mask.shape[2]))
    if write_color == True:
        # miss detection
        idx = np.logical_and((change_mask[0] == dt_map[0]), (ref_mask[0] == ref_map[1]))
        change_write[0, idx] = 1
        # false detection
        idx = np.logical_and((change_mask[0] == dt_map[1]), (ref_mask[0] == ref_map[0]))
        change_write[0, idx] = 2
        # true detection
        idx = np.logical_and((change_mask[0] == dt_map[1]), (ref_mask[0] == ref_map[1]))
        change_write[0, idx] = 3
    else:
        idx = change_mask[0] == dt_map[1]
        change_write[0, idx] = 1
    return change_write

# random mask the input image for data augmentation
class RANDOM_ERASER(torch.nn.Module):
    def __init__(self, erase_thresh=0.3, origin_prob=0.5):

        super(RANDOM_ERASER, self).__init__()
        self.erase_thresh = erase_thresh
        self.origin_prob = origin_prob

    def forward(self, img, region=None):
        # region format: (x, y, w, h)
        # erase_thresh indicates the maximum proportion of the mask region
        # origin_prob indicates the probability to use the original image instead of random mask
        if region is not None:
            x, y, w, h = region
            img[:, y:y + h, x:x + w] = 0
        else:
            if random.random() > self.origin_prob:
                band, ysize, xsize = img.shape
                x = random.randint(0, xsize - 1)
                y = random.randint(0, ysize - 1)
                w = random.randint(1, xsize - x)
                h = random.randint(1, ysize - y)
                if (w * h) / (xsize * ysize) > self.erase_thresh:
                    h = math.floor(xsize * ysize * self.erase_thresh / w)
                region = (x, y, w, h)
                img[:, y:y + h, x:x + w] = 0
            else:
                region = (0, 0, 0, 0)
        return img, region

# multiple random mask the input image for data augmentation
class RANDOM_ERASER_MULTI_REGION(torch.nn.Module):
    def __init__(self, erase_thresh=0.3, origin_prob=0.2, multi_region=5):

        super(RANDOM_ERASER_MULTI_REGION, self).__init__()
        self.erase_thresh = erase_thresh
        self.origin_prob = origin_prob
        if multi_region < 1:
            multi_region = 1
        self.multi_region = multi_region

    def forward(self, img, region=None):
        # region format: (x, y, w, h)
        # erase_thresh indicates the maximum proportion of the mask region
        # origin_prob indicates the probability to use the original image instead of random mask
        # multi_region indicates the maximum number of mask regions
        if region is not None:
            for r in region:
                x, y, w, h = r
                img[:, y:y + h, x:x + w] = 0
        else:
            region = []
            band, ysize, xsize = img.shape
            if random.random() > self.origin_prob:
                region_num = random.randint(1, self.multi_region)
                for i in range(region_num):
                    x = random.randint(0, xsize - 1)
                    y = random.randint(0, ysize - 1)
                    w = random.randint(1, xsize - x)
                    h = random.randint(1, ysize - y)
                    if (w * h) / (xsize * ysize) > self.erase_thresh:
                        h = math.floor(xsize * ysize * self.erase_thresh / w)
                    img[:, y:y + h, x:x + w] = 0
                    region.append([x, y, w, h])
        return img, region

# scale two images from their value range to [0, 1]
class SCALE(torch.nn.Module):
    def __init__(self, scale_list1=[[0, 255], [0, 255], [0, 255]], scale_list2=[[0, 255], [0, 255], [0, 255]]):

        super(SCALE, self).__init__()
        self.scale_list1 = scale_list1
        self.scale_list2 = scale_list2

    def forward(self, X, switch=1):
        if switch == 1:
            nchannel = X.shape[0]
            if nchannel > len(self.scale_list1):
                print('The input channel doesn\'t match the range list')
                sys.exit(1)
            for b in range(nchannel):
                X[b] = (X[b] - self.scale_list1[b][0]) / (self.scale_list1[b][1] - self.scale_list1[b][0])
        else:
            nchannel = X.shape[0]
            if nchannel > len(self.scale_list2):
                print('The input channel doesn\'t match the scale list')
                sys.exit(2)
            for b in range(nchannel):
                X[b] = (X[b] - self.scale_list2[b][0]) / (self.scale_list2[b][1] - self.scale_list2[b][0])

        return X

# scale two images from their value range to the given range
class SCALE_NORM(torch.nn.Module):
    def __init__(self, scale_list1=[[0, 255], [0, 255], [0, 255]], scale_list2=[[0, 255], [0, 255], [0, 255]], scale=[-1, 1]):

        super(SCALE_NORM, self).__init__()
        self.scale_list1 = scale_list1
        self.scale_list2 = scale_list2
        self.scale = scale

    def forward(self, X, switch=1):
        if switch == 1:
            nchannel = X.shape[0]
            if nchannel > len(self.scale_list1):
                print('The input channel doesn\'t match the range list')
                sys.exit(1)
            for b in range(nchannel):
                X[b] = (self.scale[1] - self.scale[0]) * (X[b] - self.scale_list1[b][0]) / (
                            self.scale_list1[b][1] - self.scale_list1[b][0]) + self.scale[0]
        else:
            nchannel = X.shape[0]
            if nchannel > len(self.scale_list2):
                print('The input channel doesn\'t match the scale list')
                sys.exit(2)
            for b in range(nchannel):
                X[b] = (self.scale[1] - self.scale[0]) * (X[b] - self.scale_list2[b][0]) / (
                            self.scale_list2[b][1] - self.scale_list2[b][0]) + self.scale[0]

        return X

# function to normalize the image with zero mean and unit std
class NORMALIZE(torch.nn.Module):
    def __init__(self, meansX, stdX, meansY, stdY):

        super(NORMALIZE, self).__init__()
        self.meansX = meansX
        self.stdX = stdX
        self.meansY = meansY
        self.stdY = stdY

    def forward(self, X, switch=1):
        if switch == 1:
            nchannel = X.shape[0]
            if nchannel > len(self.meansX):
                print('The input channel doesn\'t match the stats list')
                sys.exit(1)
            for b in range(nchannel):
                X[b] = (X[b] - self.meansX[b]) / self.stdX[b]
        else:
            nchannel = X.shape[0]
            if nchannel > len(self.meansY):
                print('The input channel doesn\'t match the stats list')
                sys.exit(2)
            for b in range(nchannel):
                X[b] = (X[b] - self.meansY[b]) / self.stdY[b]

        return X

def time_show(time):
    time_d = ''
    time_h = ''
    time_m = ''
    time_s = ''
    time_s = '{:.1f}s'.format(time % 60)
    if int(time / 60) > 0:
        time = int(time / 60)
        time_m = '{}m '.format(time % 60)
        if int(time / 60) > 0:
            time = int(time / 60)
            time_h = '{}h '.format(time % 60)
            if int(time / 24) > 0:
                time = int(time / 24)
                time_d = '{}d '.format(time)

    time_desc = '{}{}{}{}'.format(time_d, time_h, time_m, time_s)
    return time_desc

# function to calculate and record max and min value from an image by gdal
def GDALmaxmin(TxtPath, ImgPath):

    if os.path.exists(TxtPath) == False:

        ImgDS = gdal.Open(ImgPath)

        if ImgDS == None:
            print('No such a Image file')
            sys.exit(0)

        xsize = ImgDS.RasterXSize
        ysize = ImgDS.RasterYSize
        nband = ImgDS.RasterCount

        maxmin = []
        print("Reading Data")
        for b in tqdm(range(nband)):
            msImage = ImgDS.GetRasterBand(b + 1).ReadAsArray(0, 0, xsize, ysize)
            idx = msImage != 0
            maxmin.append([np.min(msImage[idx]), np.max(msImage[idx])])
            del msImage
            gc.collect()

        TxtFile = open(TxtPath, 'w')
        print("\n Save Stats Txt")
        TxtFile.write("max:")
        for b in range(nband):
            TxtFile.write(" {}".format(maxmin[b][1]))
        TxtFile.write("\n")
        TxtFile.write("min:")
        for b in range(nband):
            TxtFile.write(" {}".format(maxmin[b][0]))
        TxtFile.write("\n")
        TxtFile.close()
    else:
        TxtFile = open(TxtPath, 'r')
        contents = TxtFile.readlines()
        maxTXT = contents[0]
        minTXT = contents[1]
        maxval = [float(x) for x in maxTXT.split()[1:]]
        minval = [float(x) for x in minTXT.split()[1:]]
        maxmin = []
        for i in range(len(maxval)):
            maxmin.append([minval[i], maxval[i]])

    return maxmin

# function to calculate and record max and min value from a gdal dataset with multi-temporal images
def Dataset_maxmin(TxtPath1, TxtPath2, dataset):

    if os.path.exists(TxtPath1) == False or os.path.exists(TxtPath2) == False:

        maxmin_array1 = []
        maxmin_array2 = []
        for i in tqdm(range(dataset.__len__()), desc='Calculating Max and Min'):
            valset = dataset.__getitem__(i)
            x = valset[0]
            y = valset[1]

            if maxmin_array1 == [] or maxmin_array2 == []:
                maxmin_array1 = [[0.0] * 2 for _ in range(x.size()[0])]
                maxmin_array2 = [[0.0] * 2 for _ in range(y.size()[0])]

            idx = torch.sum(x, dim=0) != 0

            max1 = torch.max(x[:,idx], dim=1)[0]
            max2 = torch.max(y[:,idx], dim=1)[0]
            min1 = torch.min(x[:,idx], dim=1)[0]
            min2 = torch.min(y[:,idx], dim=1)[0]

            for b in range(x.size()[0]):
                maxmin_array1[b][0] = min1[b] if maxmin_array1[b][0] == 0 else maxmin_array1[b][0]
                maxmin_array1[b][0] = min1[b] if min1[b] < maxmin_array1[b][0] else maxmin_array1[b][0]
                maxmin_array1[b][1] = max1[b] if max1[b] > maxmin_array1[b][1] else maxmin_array1[b][1]

            for b in range(y.size()[0]):
                maxmin_array2[b][0] = min2[b] if maxmin_array2[b][0] == 0 else maxmin_array2[b][0]
                maxmin_array2[b][0] = min2[b] if min2[b] < maxmin_array2[b][0] else maxmin_array2[b][0]
                maxmin_array2[b][1] = max2[b] if max2[b] > maxmin_array2[b][1] else maxmin_array2[b][1]

        TxtFile = open(TxtPath1, 'w')
        print("\n Save Stats Txt")
        TxtFile.write("max:")
        for b in range(x.size()[0]):
            TxtFile.write(" {}".format(maxmin_array1[b][1]))
        TxtFile.write("\n")
        TxtFile.write("min:")
        for b in range(x.size()[0]):
            TxtFile.write(" {}".format(maxmin_array1[b][0]))
        TxtFile.write("\n")
        TxtFile.close()

        TxtFile = open(TxtPath2, 'w')
        print("\n Save Stats Txt")
        TxtFile.write("max:")
        for b in range(y.size()[0]):
            TxtFile.write(" {}".format(maxmin_array2[b][1]))
        TxtFile.write("\n")
        TxtFile.write("min:")
        for b in range(y.size()[0]):
            TxtFile.write(" {}".format(maxmin_array2[b][0]))
        TxtFile.write("\n")
        TxtFile.close()
    else:
        TxtFile = open(TxtPath1, 'r')
        contents = TxtFile.readlines()
        maxTXT = contents[0]
        minTXT = contents[1]
        maxval = [float(x) for x in maxTXT.split()[1:]]
        minval = [float(x) for x in minTXT.split()[1:]]
        maxmin_array1 = []
        for i in range(len(maxval)):
            maxmin_array1.append([minval[i], maxval[i]])

        TxtFile = open(TxtPath2, 'r')
        contents = TxtFile.readlines()
        maxTXT = contents[0]
        minTXT = contents[1]
        maxval = [float(x) for x in maxTXT.split()[1:]]
        minval = [float(x) for x in minTXT.split()[1:]]
        maxmin_array2 = []
        for i in range(len(maxval)):
            maxmin_array2.append([minval[i], maxval[i]])

    return maxmin_array1, maxmin_array2

# function to calculate and record mean and std value from a gdal dataset with multi-temporal images
def Dataset_meanstd(TxtPath1, TxtPath2, dataset):

    if os.path.exists(TxtPath1) == False or os.path.exists(TxtPath2) == False:

        meanX, meanY = Dataset_mean(dataset)
        stdX, stdY = Dataset_std(dataset, meanX, meanY)

        TxtFile = open(TxtPath1, 'w')
        print("\n Save Stats Txt")
        TxtFile.write("mean:")
        for b in range(meanX.size()[0]):
            TxtFile.write(" {}".format(meanX[b]))
        TxtFile.write("\n")
        TxtFile.write("std:")
        for b in range(stdX.size()[0]):
            TxtFile.write(" {}".format(stdX[b]))
        TxtFile.write("\n")
        TxtFile.close()

        TxtFile = open(TxtPath2, 'w')
        print("\n Save Stats Txt")
        TxtFile.write("mean:")
        for b in range(meanY.size()[0]):
            TxtFile.write(" {}".format(meanY[b]))
        TxtFile.write("\n")
        TxtFile.write("std:")
        for b in range(stdY.size()[0]):
            TxtFile.write(" {}".format(stdY[b]))
        TxtFile.write("\n")
        TxtFile.close()

        meanX = meanX.numpy().tolist()
        meanY = meanY.numpy().tolist()
        stdX = stdX.numpy().tolist()
        stdY = stdY.numpy().tolist()

    else:
        TxtFile = open(TxtPath1, 'r')
        contents = TxtFile.readlines()
        meanTXT = contents[0]
        stdTXT = contents[1]
        meanval = [float(x) for x in meanTXT.split()[1:]]
        stdval = [float(x) for x in stdTXT.split()[1:]]
        meanX = []
        stdX = []
        for i in range(len(meanval)):
            meanX.append(meanval[i])
            stdX.append(stdval[i])

        TxtFile = open(TxtPath2, 'r')
        contents = TxtFile.readlines()
        meanTXT = contents[0]
        stdTXT = contents[1]
        meanval = [float(x) for x in meanTXT.split()[1:]]
        stdval = [float(x) for x in stdTXT.split()[1:]]
        meanY = []
        stdY = []
        for i in range(len(meanval)):
            meanY.append(meanval[i])
            stdY.append(stdval[i])

    return meanX, stdX, meanY, stdY

def Dataset_mean(dataset):

    npixel_array = []
    meanX_array = []
    meanY_array = []
    for i in tqdm(range(dataset.__len__()), desc='Calculating Mean'):
        valset = dataset.__getitem__(i)
        x = valset[0]
        y = valset[1]

        idx = torch.sum(x, dim=0) != 0
        npixel = torch.sum(idx)

        npixel_array.append(npixel)

        meanX_array.append(torch.mean(x[:, idx], 1))
        meanY_array.append(torch.mean(y[:, idx], 1))

    meanX_array = torch.cat(meanX_array).reshape(len(meanX_array), -1)
    meanY_array = torch.cat(meanY_array).reshape(len(meanY_array), -1)
    npixel_array = torch.tensor(npixel_array).reshape(-1, 1)
    npixel = torch.sum(npixel_array)

    npixel_mat = npixel_array.repeat(1, meanX_array.size()[1]) / npixel
    meanX = torch.sum(meanX_array * npixel_mat, dim=0)

    npixel_mat = npixel_array.repeat(1, meanY_array.size()[1]) / npixel
    meanY = torch.sum(meanY_array * npixel_mat, dim=0)

    return meanX, meanY

def Dataset_std(dataset, meanX, meanY):

    npixel_array = []
    varX_array = []
    varY_array = []

    meanX = meanX.reshape(-1, 1)
    meanY = meanY.reshape(-1, 1)

    for i in tqdm(range(dataset.__len__()), desc='Calculating Std'):
        valset = dataset.__getitem__(i)
        x = valset[0]
        y = valset[1]

        idx = torch.sum(x, dim=0) != 0
        npixel = torch.sum(idx)

        npixel_array.append(npixel)

        varX_array.append(torch.mean(torch.square(x[:, idx] - meanX.repeat(1, npixel)), 1))
        varY_array.append(torch.mean(torch.square(y[:, idx] - meanY.repeat(1, npixel)), 1))

    varX_array = torch.cat(varX_array).reshape(len(varX_array), -1)
    varY_array = torch.cat(varY_array).reshape(len(varY_array), -1)
    npixel_array = torch.tensor(npixel_array).reshape(-1, 1)
    npixel = torch.sum(npixel_array)

    npixel_mat = npixel_array.repeat(1, varX_array.size()[1]) / (npixel - 1)
    stdX = torch.sqrt(torch.sum(varX_array * npixel_mat, dim=0))

    npixel_mat = npixel_array.repeat(1, varY_array.size()[1]) / (npixel - 1)
    stdY = torch.sqrt(torch.sum(varY_array * npixel_mat, dim=0))

    return stdX, stdY