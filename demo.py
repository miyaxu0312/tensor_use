import numpy as np
import os
from glob import glob
import scipy.io as sio
from skimage.io import imread, imsave
from skimage.transform import rescale, resize
from time import time
import argparse
import ast
from api import PRN


def main(args):
    # ---- init PRN
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu # GPU number, -1 for CPU
    prn = PRN()

    # ------------- load data
    image_folder = args.inputDir
    save_folder = args.outputDir
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    types = ('*.jpg', '*.png')
    image_path_list= []
    for files in types:
        image_path_list.extend(glob(os.path.join(image_folder, files)))
    total_num = len(image_path_list)

    for i, image_path in enumerate(image_path_list):

        name = image_path.strip().split('/')[-1][:-4]
        # read image
        image = imread(image_path)
        [h, w, _] = image.shape
        pos = prn.net_forward(image) # input image has been cropped to 256x256
        imsave(os.path.join(save_folder, name + '.jpg'), image)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Joint 3D Face Reconstruction and Dense Alignment with Position Map Regression Network')

    parser.add_argument('-i', '--inputDir', default='TestImages/input', type=str,
                        help='path to the input directory, where input images are stored.')
    parser.add_argument('-o', '--outputDir', default='TestImages/output', type=str,
                        help='path to the output directory, where results(obj,txt files) will be stored.')
    parser.add_argument('--gpu', default='0', type=str,
                        help='set gpu id, -1 for CPU')
   
    main(parser.parse_args())
