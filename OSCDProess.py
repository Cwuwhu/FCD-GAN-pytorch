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


# This code is used for generate the dataset of OSCD with the regional labelling reference
# In order to generate a standard regional supervised change detection dataset, OSCD dataset was further processed
# A manual co-registration is implemented, since I found that there are still some co-registration errors in OSCD
# After co-registration, the images including before, after change and cm are all sliced to the same shape
# I only select 4 bands with 10m resolution to generate a ENVI format file
# After the aforementioned process, we got a folder, named with the name of the image
# For example, in the folder of abudhabi\ImagePair\, we have the following data
#     abudhabi_20160120
#     abudhabi_20160120.hdr
#     abudhabi_20180328
#     abudhabi_20180328.hdr
#     abudhabi-cm.tif
# Finally, we can use the following code to generate a regional generated dataset of OSCD

if __name__ == '__main__':

    # Dir of the dataset OSCD
    InPath = r'/OSCD-10m-Dataset/'

    # The ext of the regional cm
    ext = '.tif'

    # The parameter to determine the expanding distance to draw a regional reference
    region_expand = 10

    dir_name = [x for x in os.listdir(InPath) if os.path.isdir(os.path.join(InPath, x))]

    for file_name in dir_name:
        cur_path = os.path.join(InPath, file_name, 'ImagePair')
        ref_name = [x for x in os.listdir(cur_path) if x.split('-')[-1] == 'cm.tif']
        if len(ref_name) != 1:
            print('Reference {} cannot be found'.format(cur_path))
            sys.exit(0)

        ref_name = ref_name[0]
        ref_path = os.path.join(cur_path, ref_name)
        msImage_ref = Image.open(ref_path)
        msImage_ref = np.array(msImage_ref)
        msImage_region = np.zeros_like(msImage_ref)
        msImage_region[msImage_ref > 1] = 255

        out_region_img, region_num = measure.label(msImage_region, connectivity=2, background=0, return_num=True)
        region_props = measure.regionprops(out_region_img)
        out_region_img = np.zeros_like(out_region_img)
        for prop in region_props:
            min_y = prop.bbox[0]
            min_x = prop.bbox[1]
            max_y = prop.bbox[2]
            max_x = prop.bbox[3]

            min_y = min_y - region_expand if (min_y - region_expand) > 0 else 0
            min_x = min_x - region_expand if (min_x - region_expand) > 0 else 0
            max_y = max_y + region_expand if (max_y + region_expand) < msImage_region.shape[0] else msImage_region.shape[0]
            max_x = max_x + region_expand if (max_x + region_expand) < msImage_region.shape[1] else msImage_region.shape[1]

            msImage_region[min_y:max_y, min_x:max_x] = 255

        out_region_file = os.path.join(cur_path, '{}-region{}'.format(file_name, ext))
        out_region_img = Image.fromarray(np.uint8(msImage_region))

        out_region_img.save(out_region_file)

        print('Saving region image of {}\n'.format(file_name))
