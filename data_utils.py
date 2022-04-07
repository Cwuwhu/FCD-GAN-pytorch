import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as trans
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models

import os
import sys
from osgeo import gdal
from osgeo import ogr
from osgeo import osr
import numpy as np
import cv2
from tqdm import tqdm
import gc
import math
import random

from PIL import Image

# ERROR 1: PROJ
os.environ['PROJ_LIB'] = r'/anaconda3/share/proj/'

# dataset to read remote sensing images with gdal
# the read patch is obtained from the large-scale image with overlaps
# when writing the patches, only the centering region without overlap padding is written
class GDALDataset(Dataset):

    # 初始化
    def __init__(self, imgPathX, imgPathY, refPath=None, outPath=None, transforms=None, enhance=None, patch_size=(200, 200), overlap_padding=(10, 10)):
        super(GDALDataset, self).__init__()
        self.imgPathX = imgPathX
        self.imgDS_x = gdal.Open(imgPathX)
        if self.imgDS_x is None:
            print('No such a Image file:{}'.format(imgPathX))
            sys.exit(0)
        xsize = self.imgDS_x.RasterXSize
        ysize = self.imgDS_x.RasterYSize
        nband = self.imgDS_x.RasterCount

        self.imgPathY = imgPathY
        self.imgDS_y = gdal.Open(imgPathY)
        if self.imgDS_y is None:
            print('No such a Image file:{}'.format(imgPathY))
            sys.exit(0)
        xsize2 = self.imgDS_y.RasterXSize
        ysize2 = self.imgDS_y.RasterYSize
        nband2 = self.imgDS_y.RasterCount

        if xsize != xsize2 or ysize != ysize2 or nband != nband2:
            print('Image sizes don\'t match')
            sys.exit(0)

        self.transforms = transforms
        self.enhance = enhance

        xstart = list(range(0, xsize, patch_size[0] - 2 * overlap_padding[0]))
        xend = [(x + patch_size[0]  - 2 * overlap_padding[0]) for x in xstart if (x + patch_size[0] - 2 * overlap_padding[0] < xsize)]
        xend.append(xsize)

        ystart = list(range(0, ysize, patch_size[1] - 2 * overlap_padding[1]))
        yend = [(y + patch_size[1] - 2 * overlap_padding[1]) for y in ystart if (y + patch_size[1] - 2 * overlap_padding[1] < ysize)]
        yend.append(ysize)

        self.xstart = xstart
        self.xend = xend
        self.ystart = ystart
        self.yend = yend

        self.patch_size = patch_size
        self.overlap_padding = overlap_padding

        self.refPath = refPath
        if refPath is not None:
            self.imgDS_ref = gdal.Open(refPath)
            if self.imgDS_ref is None:
                print('No such a Image file:{}'.format(refPath))
                sys.exit(0)
            xsize3 = self.imgDS_ref.RasterXSize
            ysize3 = self.imgDS_ref.RasterYSize
            nband3 = self.imgDS_ref.RasterCount
            if xsize != xsize3 or ysize != ysize3 or nband3 != 1:
                print('Reference sizes don\'t match image')
                sys.exit(0)
        else:
            self.imgDS_ref = None

        self.outPath = outPath
        self.outDS = None

    def __getitem__(self, item):
        xitem_count, yitem_count = self.patch_count()

        item_x = math.floor(item / yitem_count)
        item_y = item % yitem_count

        slice, slice_read, slice_write = self.slice_assign(item_x, item_y)

        xsize, ysize, nband = self.size()

        tmp_x = []
        tmp_y = []
        for b in range(nband):
            tmp_x.append(self.imgDS_x.GetRasterBand(b + 1).ReadAsArray(slice_read[0], slice_read[1], slice_read[2], slice_read[3]))
            tmp_y.append(self.imgDS_y.GetRasterBand(b + 1).ReadAsArray(slice_read[0], slice_read[1], slice_read[2], slice_read[3]))

        tmp_x = np.array(tmp_x, dtype=float)
        tmp_y = np.array(tmp_y, dtype=float)

        if self.enhance is not None:
            tmp_x = self.enhance(tmp_x, switch=1)
            tmp_y = self.enhance(tmp_y, switch=2)

        msImage_x = np.zeros((nband, self.patch_size[1], self.patch_size[0]), dtype=float)
        msImage_y = np.zeros((nband, self.patch_size[1], self.patch_size[0]), dtype=float)

        msImage_x[:, slice_write[1]:slice_write[1] + slice_write[3],
        slice_write[0]:slice_write[0] + slice_write[2]] = tmp_x
        msImage_y[:, slice_write[1]:slice_write[1] + slice_write[3],
        slice_write[0]:slice_write[0] + slice_write[2]] = tmp_y

        msImage_x = torch.from_numpy(msImage_x).float()
        msImage_y = torch.from_numpy(msImage_y).float()
        item = torch.tensor(item)

        if self.transforms is not None:
            msImage_x, sync = self.transforms(msImage_x)
            msImage_y, sync = self.transforms(msImage_y, sync)

        refImage = np.zeros((1, self.patch_size[1], self.patch_size[0]), dtype=float)
        if self.imgDS_ref is not None:
            tmp_ref = []
            tmp_ref.append(self.imgDS_ref.GetRasterBand(1).ReadAsArray(slice_read[0], slice_read[1], slice_read[2],
                                                                       slice_read[3]))
            tmp_ref = np.array(tmp_ref, dtype=float)
            refImage[:, slice_write[1]:slice_write[1] + slice_write[3],
            slice_write[0]:slice_write[0] + slice_write[2]] = tmp_ref
        refImage = torch.from_numpy(refImage).float()

        return msImage_x, msImage_y, item, refImage

    def __len__(self):
        return len(self.xstart) * len(self.ystart)

    def patch_count(self):
        return len(self.xstart), len(self.ystart)

    def size(self):
        xsize = self.imgDS_x.RasterXSize
        ysize = self.imgDS_x.RasterYSize
        nband = self.imgDS_x.RasterCount
        return xsize, ysize, nband

    def slice_assign(self, item_x, item_y):

        pad = self.overlap_padding
        xsize, ysize, nband = self.size()

        xstart = self.xstart[item_x]
        xend = self.xend[item_x]
        ystart = self.ystart[item_y]
        yend = self.yend[item_y]
        slice = (xstart, ystart, xend - xstart, yend - ystart)

        x_ori = 0 if xstart - pad[0] > 0 else pad[0]
        y_ori = 0 if ystart - pad[1] > 0 else pad[1]

        xstart = xstart - pad[0] if xstart - pad[0] > 0 else 0
        ystart = ystart - pad[1] if ystart - pad[1] > 0 else 0
        xend = xend + pad[0] if xend + pad[0] < xsize else xsize
        yend = yend + pad[1] if yend + pad[1] < ysize else ysize
        slice_read = (xstart, ystart, xend - xstart, yend - ystart)

        slice_write = (x_ori, y_ori, xend - xstart, yend - ystart)

        return slice, slice_read, slice_write

    def GDALwriteDefault(self, outImage, item):
        # Only write one-band image

        if self.outPath == None:
            dir, fname = os.path.split(self.imgPathX)
            fname, ext = os.path.splitext(fname)
            fname = "{}_cmp{}".format(fname, ext)
            outPath = os.path.join(dir, fname)
            self.outPath = outPath

        xsize, ysize, nband = self.size()

        if self.outDS == None:
            driver = self.imgDS_x.GetDriver()
            self.outDS = driver.Create(self.outPath, xsize, ysize, 1, gdal.GDT_Float32)
            if self.outDS == None:
                print("Cannot make a output raster")
                sys.exit(0)

            self.outDS.SetGeoTransform(self.imgDS_x.GetGeoTransform())
            self.outDS.SetProjection(self.imgDS_x.GetProjection())

            outBand = self.outDS.GetRasterBand(1)
            # outBand.SetNoDataValue(0)
        else:
            outBand = self.outDS.GetRasterBand(1)

        xitem_count, yitem_count = self.patch_count()

        item_x = math.floor(item / yitem_count)
        item_y = item % yitem_count

        slice, slice_read, slice_write = self.slice_assign(item_x, item_y)

        pad = self.overlap_padding
        outBand.WriteArray(outImage[0, pad[1]:pad[1]+slice[3], pad[0]:pad[0]+slice[2]], slice[0], slice[1])

    def GDALwrite(self, outImage, item, outGDAL=None):

        if outGDAL == None:
            self.GDALwriteDefault(outImage.numpy(), item)
            return

        if outImage.shape[0] != outGDAL.RasterCount:
            print('The band of output image doesn\'t match the output GDAL dataset')
            sys.exit(0)

        xitem_count, yitem_count = self.patch_count()

        item_x = math.floor(item / yitem_count)
        item_y = item % yitem_count

        slice, slice_read, slice_write = self.slice_assign(item_x, item_y)

        pad = self.overlap_padding

        for b in range(outGDAL.RasterCount):
            outBand = outGDAL.GetRasterBand(b + 1)
            outBand.WriteArray(outImage[b, pad[1]:pad[1] + slice[3], pad[0]:pad[0] + slice[2]], slice[0], slice[1])

# read remote sensing images with gdal, and also the regional reference
class GDALDataset_RSS(Dataset):

    # 初始化
    def __init__(self, imgPathX, imgPathY, regionPath=None, refPath=None, outPath=None, transforms=None, enhance=None, patch_size=(200, 200), overlap_padding=(10, 10)):

        super(GDALDataset_RSS, self).__init__()
        self.DS = GDALDataset(imgPathX, imgPathY, refPath=refPath, outPath=outPath, transforms=transforms, enhance=enhance, patch_size=patch_size, overlap_padding=overlap_padding)
        self.ds_len = self.DS.__len__()
        self.regionPath = regionPath
        self.patch_size = patch_size

        if regionPath is not None:
            self.imgDS_region = gdal.Open(regionPath)
            if self.imgDS_region is None:
                print('No such a Image file:{}'.format(regionPath))
                sys.exit(0)
            xsize = self.imgDS_region.RasterXSize
            ysize = self.imgDS_region.RasterYSize
            nband = self.imgDS_region.RasterCount
            if xsize != self.DS.size()[0] or ysize != self.DS.size()[1] or nband != 1:
                print('Reference sizes don\'t match image')
                sys.exit(0)
        else:
            self.imgDS_region = None

    def __getitem__(self, item):
        msImage_x, msImage_y, item, refImage = self.DS.__getitem__(item)

        xitem_count, yitem_count = self.DS.patch_count()

        item_x = math.floor(item / yitem_count)
        item_y = item % yitem_count
        slice, slice_read, slice_write = self.DS.slice_assign(item_x, item_y)

        regionImage = np.zeros((1, self.patch_size[1], self.patch_size[0]), dtype=float)
        if self.imgDS_region is not None:
            tmp_ref = []
            tmp_ref.append(self.imgDS_region.GetRasterBand(1).ReadAsArray(slice_read[0], slice_read[1], slice_read[2],
                                                                       slice_read[3]))
            tmp_ref = np.array(tmp_ref, dtype=float)
            regionImage[:, slice_write[1]:slice_write[1] + slice_write[3],
            slice_write[0]:slice_write[0] + slice_write[2]] = tmp_ref
        regionImage[regionImage > 125] = 1
        regionImage = torch.from_numpy(regionImage).float()

        return msImage_x, msImage_y, item, refImage, regionImage

    def __len__(self):
        return self.ds_len

    def GDALwrite(self, outImage, item, outGDAL=None):
        self.DS.GDALwrite(outImage, item, outGDAL)


# read OSCD dataset with regional reference
class OSCD_Dataset_RSS(Dataset):

    def __init__(self, imgDir, txtName, scaler=None, transforms=None, patch_size=(200, 200), overlap_padding=(10, 10)):
        super(OSCD_Dataset_RSS, self).__init__()

        self.patch_size = patch_size
        self.overlap_padding = overlap_padding
        self.imgDir = imgDir
        self.txtName = txtName

        txtPath = os.path.join(imgDir, txtName)
        f = open(txtPath, 'r')
        if f is None:
            print('No txt file')
            sys.exit(0)

        line = f.readline()
        line = line.strip()
        filename = line.split(',')

        self.dslist = []
        self.numlist = []
        self.namelist = []
        self.pathlist = []

        # read the image pairs with the names in the txt file
        for name in filename:
            cur_path = os.path.join(imgDir, name, 'ImagePair')
            img_name = [x for x in os.listdir(cur_path) if (os.path.splitext(x)[-1] == '') & (x.find(name) != -1)]
            if len(img_name) != 2:
                print('Error in finding image file {}'.format(cur_path))
                sys.exit(0)
            ref_name = [x for x in os.listdir(cur_path) if x.split('-')[-1] == 'cm.tif']
            if len(ref_name) != 1:
                print('Error in finding reference file {}'.format(cur_path))
                sys.exit(0)
            region_name = [x for x in os.listdir(cur_path) if x.split('-')[-1] == 'region.tif']
            if len(region_name) != 1:
                print('Error in finding region file {}'.format(cur_path))
                sys.exit(0)
            ImgXPath = os.path.join(cur_path, img_name[0])
            ImgYPath = os.path.join(cur_path, img_name[1])
            RefPath = os.path.join(cur_path, ref_name[0])
            RegionPath = os.path.join(cur_path, region_name[0])

            self.pathlist.append([ImgXPath, ImgYPath, RefPath, RegionPath])

            if scaler is None:
                cur_scaler = None
            else:
                if len(scaler) != len(filename):
                    print('The list of scaler doesn\'t match the file list')
                    sys.exit(0)
                else:
                    idx = filename.index(name)
                    cur_scaler = scaler[idx]

            if transforms is None:
                cur_transforms = None
            else:
                if len(transforms) != len(filename):
                    print('The list of transforms doesn\'t match the file list')
                    sys.exit(0)
                else:
                    idx = filename.index(name)
                    cur_transforms = transforms[idx]

            dataset = GDALDataset_RSS(ImgXPath, ImgYPath, refPath=RefPath, regionPath=RegionPath, enhance=cur_scaler, transforms=cur_transforms, patch_size=patch_size, overlap_padding=overlap_padding)
            self.dslist.append(dataset)
            self.numlist.append(dataset.__len__())
            self.namelist.append(name)

        self.len = np.sum(np.array(self.numlist))
        self.cumlen = np.cumsum(np.array(self.numlist)).tolist()

        self.outGDALlist = []
        self.outFilterlist = []

    def __getitem__(self, item):
        if item > self.cumlen[-1]:
            print('item exceeds the len')
            sys.exit(0)

        item_ds = np.where(np.array(self.cumlen) > item)[0][0]
        cur_item = item - self.cumlen[item_ds - 1] if item_ds > 0 else item

        imgX, imgY, item, Ref, Region = self.dslist[item_ds].__getitem__(cur_item)

        item = item + self.cumlen[item_ds - 1] if item_ds > 0 else item

        return imgX, imgY, item, Ref, Region

    def __len__(self):
        return self.len

    # return the center range without overlap padding
    def EffRange(self, item):
        if item > self.cumlen[-1]:
            print('item exceeds the len')
            sys.exit(0)

        item_ds = np.where(np.array(self.cumlen) > item)[0][0]
        cur_item = item - self.cumlen[item_ds - 1] if item_ds > 0 else item

        xitem_count, yitem_count = self.dslist[item_ds].DS.patch_count()
        pad = self.dslist[item_ds].DS.overlap_padding

        item_x = math.floor(cur_item / yitem_count)
        item_y = cur_item % yitem_count
        slice, _, _ = self.dslist[item_ds].DS.slice_assign(item_x, item_y)

        return  pad[1], pad[1] + slice[3], pad[0], pad[0] + slice[2]

    # write the output image without overlap padding
    def GDALwrite(self, outImage, item, filterName):

        if filterName not in self.outFilterlist:
            self.outFilterlist.append(filterName)
            GDALarray = [None for _ in range(len(self.namelist))]
            self.outGDALlist.append(GDALarray)

        idx = self.outFilterlist.index(filterName)
        GDALarray = self.outGDALlist[idx]

        item_ds = np.where(np.array(self.cumlen) > item)[0][0]
        cur_item = item - self.cumlen[item_ds - 1] if item_ds > 0 else item

        outGDAL = GDALarray[item_ds]
        nband = outImage.shape[0]

        if outGDAL == None:
            driver = self.dslist[item_ds].DS.imgDS_x.GetDriver()
            xsize, ysize, _ = self.dslist[item_ds].DS.size()
            ds_name = self.namelist[item_ds]
            # outName = '{}{}'.format(ds_name, filterName)
            outName = '{}'.format(filterName)
            outImg_path = os.path.join(self.imgDir, ds_name, 'ImagePair', outName)
            outGDAL = driver.Create(outImg_path, xsize, ysize, nband, gdal.GDT_Float32)
            if outGDAL == None:
                print("Cannot make a output raster")
                sys.exit(0)

            outGDAL.SetGeoTransform(self.dslist[item_ds].DS.imgDS_x.GetGeoTransform())
            outGDAL.SetProjection(self.dslist[item_ds].DS.imgDS_x.GetProjection())

            # for b in range(nband):
            #     outBand = outGDAL.GetRasterBand(b + 1)
                # outBand.SetNoDataValue(0)

            GDALarray[item_ds] = outGDAL
            self.outGDALlist[idx] = GDALarray

        self.dslist[item_ds].GDALwrite(outImage, cur_item, outGDAL)