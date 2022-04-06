import os
import numpy as np
import cv2
from tqdm import tqdm
import gc
import math
import random
import sys

from osgeo import gdal
from osgeo import ogr
from osgeo import osr

from PIL import Image
from skimage import measure
from tqdm import tqdm

# This code is used to generate the dataset of weakly or regional supervised change detection with WHU Building Change Detection Dataset (https://study.rsgis.whu.edu.cn/pages/download/building_dataset.html)
# The large-scale images are sliced with a fixed size, the reference map is also sliced accordingly
# A set of regional supervised maps are generated with a given expanding distance
# A txt file, which indicate the image pairs with changes, is automatically generated

if __name__ == '__main__':

    InPath = r'\Building change detection dataset\1. The two-period image data\before\before.tif'
    RefPath = r'\Building change detection dataset\1. The two-period image data\after\after.tif'
    LabelPath = r'\Building change detection dataset\1. The two-period image data\change label\change_label.tif'
    OutPath = r'\Building CD Slice Dataset'

    # The folder name in OutPath
    RefExt = 'before'
    TarExt = 'after'

    # The ext name of the slice images
    ext = '.tif'

    # The slice shape
    slice_x = 200
    slice_y = 200

    # The expanding distance to draw regional supervised reference
    region_expand = 10

    OutPathRef = os.path.join(OutPath, RefExt)
    OutPathTar = os.path.join(OutPath, TarExt)
    OutPathLabel = os.path.join(OutPath, 'Label')
    OutPathRegion = os.path.join(OutPath, 'Region Label')

    if os.path.exists(OutPath) == False:
        os.mkdir(OutPath)
    if os.path.exists(OutPathRef) == False:
        os.mkdir(OutPathRef)
    if os.path.exists(OutPathTar) == False:
        os.mkdir(OutPathTar)
    if os.path.exists(OutPathLabel) == False:
        os.mkdir(OutPathLabel)
    if os.path.exists(OutPathRegion) == False:
        os.mkdir(OutPathRegion)

    imgDS_x = gdal.Open(InPath)
    if imgDS_x is None:
        print('No such a Image file:{}'.format(InPath))
        sys.exit(0)
    xsize = imgDS_x.RasterXSize
    ysize = imgDS_x.RasterYSize
    nband = imgDS_x.RasterCount

    imgDS_y = gdal.Open(RefPath)
    if imgDS_y is None:
        print('No such a Image file:{}'.format(RefPath))
        sys.exit(0)
    xsize2 = imgDS_y.RasterXSize
    ysize2 = imgDS_y.RasterYSize
    nband2 = imgDS_y.RasterCount

    imgDS_ref = gdal.Open(LabelPath)
    if imgDS_ref is None:
        print('No such a Image file:{}'.format(LabelPath))
        sys.exit(0)
    xsize3 = imgDS_ref.RasterXSize
    ysize3 = imgDS_ref.RasterYSize
    nband3 = imgDS_ref.RasterCount

    if xsize != xsize2 or ysize != ysize2 or nband != nband2 or xsize != xsize3 or ysize != ysize3 or nband3 != 1:
        print('Image sizes don\'t match')
        sys.exit(0)

    xstart = list(range(0, xsize, slice_x))
    xend = [(x + slice_x) for x in xstart if (x + slice_x < xsize)]
    xend.append(xsize)

    ystart = list(range(0, ysize, slice_y))
    yend = [(y + slice_y) for y in ystart if (y + slice_y < ysize)]
    yend.append(ysize)

    LabelTxtPath = os.path.join(OutPath, 'label.txt')
    TxtFile = open(LabelTxtPath, 'w')

    for i in tqdm(range(len(xstart)), desc='Processing axis-x'):
        for j in range(len(ystart)):
            slice_read = (xstart[i], ystart[j], xend[i]-xstart[i], yend[j]-ystart[j])

            tmp_x = []
            tmp_y = []
            for b in range(nband):
                tmp_x.append(imgDS_x.GetRasterBand(b + 1).ReadAsArray(slice_read[0], slice_read[1], slice_read[2],
                                                                      slice_read[3]))
                tmp_y.append(imgDS_y.GetRasterBand(b + 1).ReadAsArray(slice_read[0], slice_read[1], slice_read[2],
                                                                      slice_read[3]))

            tmp_ref = imgDS_ref.GetRasterBand(1).ReadAsArray(slice_read[0], slice_read[1], slice_read[2], slice_read[3])

            tmp_x = np.array(tmp_x, dtype=float)
            tmp_y = np.array(tmp_y, dtype=float)
            tmp_ref = np.array(tmp_ref, dtype=float)

            msImage_x = np.zeros((nband, slice_y, slice_x), dtype=float)
            msImage_y = np.zeros((nband, slice_y, slice_x), dtype=float)
            msImage_ref = np.zeros((slice_y, slice_x), dtype=float)
            msImage_region = np.zeros((slice_y, slice_x), dtype=float)

            slice_write = (0, 0, xend[i]-xstart[i], yend[j]-ystart[j])

            msImage_x[:, slice_write[1]:slice_write[3], slice_write[0]:slice_write[2]] = tmp_x
            msImage_y[:, slice_write[1]:slice_write[3], slice_write[0]:slice_write[2]] = tmp_y
            msImage_ref[slice_write[1]:slice_write[3], slice_write[0]:slice_write[2]] = tmp_ref
            msImage_ref[msImage_ref > 0] = 255

            change_label = 1 if np.sum(msImage_ref) > 0 else 0

            out_region_img, region_num = measure.label(msImage_ref, connectivity=2, background=0, return_num=True)
            region_props = measure.regionprops(out_region_img)
            out_region_img = np.zeros_like(out_region_img)
            for prop in region_props:
                min_y = prop.bbox[0]
                min_x = prop.bbox[1]
                max_y = prop.bbox[2]
                max_x = prop.bbox[3]

                min_y = min_y - region_expand if (min_y - region_expand) > 0 else 0
                min_x = min_x - region_expand if (min_x - region_expand) > 0 else 0
                max_y = max_y + region_expand if (max_y + region_expand) < slice_y else slice_y
                max_x = max_x + region_expand if (max_x + region_expand) < slice_x else slice_x

                msImage_region[min_y:max_y, min_x:max_x] = 255

            msImage_x = msImage_x.transpose([1, 2, 0])
            msImage_y = msImage_y.transpose([1, 2, 0])

            out_ref_file = os.path.join(OutPathRef, '{}_{}{}'.format(xstart[i], ystart[j], ext))
            out_tar_file = os.path.join(OutPathTar, '{}_{}{}'.format(xstart[i], ystart[j], ext))
            out_label_file = os.path.join(OutPathLabel, '{}_{}{}'.format(xstart[i], ystart[j], ext))
            out_region_file = os.path.join(OutPathRegion, '{}_{}{}'.format(xstart[i], ystart[j], ext))

            out_x_img = Image.fromarray(np.uint8(msImage_x))
            out_y_img = Image.fromarray(np.uint8(msImage_y))
            out_label_img = Image.fromarray(np.uint8(msImage_ref))
            out_region_img = Image.fromarray(np.uint8(msImage_region))

            out_x_img.save(out_ref_file)
            out_y_img.save(out_tar_file)
            out_label_img.save(out_label_file)
            out_region_img.save(out_region_file)

            # In the txt file "label.txt", the name of each image pair is listed with 0/1 to indicate unchanged/changed
            # The first two numbers are used to label the scene class of this image pair, which are useless in this experiment
            TxtFile.write('{}_{}{},0,0,{}\n'.format(xstart[i], ystart[j], ext, change_label))

    TxtFile.close()



