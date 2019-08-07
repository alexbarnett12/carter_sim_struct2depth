# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

""" Offline data generation for the KITTI dataset."""

import os
from absl import app
import numpy as np
import cv2
import glob
import csv

# Segmentation mask generation
# from .gen_masks_kitti import MaskGenerator
# from .alignment import align

def crop(img, segimg, fx, fy, cx, cy):
    # Perform center cropping, preserving 50% vertically.
    middle_perc = 0.50
    left = 1 - middle_perc
    half = left / 2
    a = img[int(img.shape[0] * (half)):int(img.shape[0] * (1 - half)), :]
    aseg = segimg[int(segimg.shape[0] * (half)):int(segimg.shape[0] * (1 - half)), :]
    cy /= (1 / middle_perc)

    # Resize to match target height while preserving aspect ratio.
    wdt = int((128 * a.shape[1] / a.shape[0]))
    x_scaling = float(wdt) / a.shape[1]
    y_scaling = 128.0 / a.shape[0]
    b = cv2.resize(a, (wdt, 128))
    bseg = cv2.resize(aseg, (wdt, 128))

    # Adjust intrinsics.
    fx *= x_scaling
    fy *= y_scaling
    cx *= x_scaling
    cy *= y_scaling

    # Perform center cropping horizontally.
    remain = b.shape[1] - 416
    cx /= (b.shape[1] / 416)
    c = b[:, int(remain / 2):b.shape[1] - int(remain / 2)]
    cseg = bseg[:, int(remain / 2):b.shape[1] - int(remain / 2)]

    return c, cseg, fx, fy, cx, cy

def get_camera_intrinsics(file):
    camera_mat = np.zeros(shape=(3,3))
    with open(file) as file:
        csv_reader = csv.reader(file, delimiter=',')
        for row in csv_reader:
            camera_mat[0][0] = float(row[0])
            camera_mat[0][2]= float(row[2])
            camera_mat[1][1] = float(row[4])
            camera_mat[1][2] = float(row[5])
            camera_mat[2][2] = 1
    return camera_mat

def main():
    ct = 0
    SEQ_LENGTH = 3
    WIDTH = 416
    HEIGHT = 128
    STEPSIZE = 1
    INPUT_DIR = '/mnt/test_data/raw_images'
    OUTPUT_IMAGE_DIR = '/mnt/test_data/processed_images'
    OUTPUT_SEG_MASK_DIR = '/mnt/test_data/processed_seg_masks'
    OUTPUT_INTRINSICS_DIR = '/mnt/test_data/processed_intrinsics'
    CALIB_FILE = 'kinect_camera_intrinsics.csv'
    OPTIMIZE = True
    REPETITIONS = 5

    # Create a segmentation mask generator
    # mask_generator = MaskGenerator()
    if not OUTPUT_IMAGE_DIR.endswith('/'):
        OUTPUT_IMAGE_DIR = OUTPUT_IMAGE_DIR + '/'
    if not OUTPUT_SEG_MASK_DIR.endswith('/'):
        OUTPUT_SEG_MASK_DIR = OUTPUT_SEG_MASK_DIR + '/'
    if not OUTPUT_INTRINSICS_DIR.endswith('/'):
        OUTPUT_INTRINSICS_DIR = OUTPUT_INTRINSICS_DIR + '/'

    if not os.path.exists(OUTPUT_IMAGE_DIR):
        os.mkdir(OUTPUT_IMAGE_DIR)
    if not os.path.exists(OUTPUT_SEG_MASK_DIR):
        os.mkdir(OUTPUT_SEG_MASK_DIR)
    if not os.path.exists(OUTPUT_INTRINSICS_DIR):
        os.mkdir(OUTPUT_INTRINSICS_DIR)

    # Get camera intrinsics
    camera_mat = get_camera_intrinsics(CALIB_FILE)

    # Retrieve image file names and sort them
    files = sorted(glob.glob(INPUT_DIR + '/*.png'))
    for i in range(SEQ_LENGTH, len(files) + 1, STEPSIZE):
        imgnum = str(ct).zfill(10)

        big_img = np.zeros(shape=(HEIGHT, WIDTH * SEQ_LENGTH, 3))
        wct = 0

        # Define list of seg_mask images
        seg_list = []

        for j in range(i - SEQ_LENGTH, i):  # Collect frames for this sample.
            img = cv2.imread(files[j])
            ORIGINAL_HEIGHT, ORIGINAL_WIDTH, _ = img.shape

            zoom_x = WIDTH / ORIGINAL_WIDTH
            zoom_y = HEIGHT / ORIGINAL_HEIGHT

            # Adjust intrinsics.
            calib_current = camera_mat.copy()
            calib_current[0, 0] *= zoom_x
            calib_current[0, 2] *= zoom_x
            calib_current[1, 1] *= zoom_y
            calib_current[1, 2] *= zoom_y

            calib_representation = ','.join([str(c) for c in calib_current.flatten()])

            img = cv2.resize(img, (WIDTH, HEIGHT))
            big_img[:, wct * WIDTH:(wct + 1) * WIDTH] = img
            wct += 1

            # Generate seg_mask and add to list
            # seg_list.append(mask_generator.generate_seg_img(img))
            # mask_generator.visualize()

        # Align seg_masks
        # seg_list[0], seg_list[1], seg_list[2] = align(seg_list[0], seg_list[1], seg_list[2])
        big_seg_img = np.zeros(shape=(HEIGHT, WIDTH * SEQ_LENGTH, 3))

        # Create seg_mask triplet
        # for k in range(0, 3):
        #     big_seg_img[:, k * WIDTH:(k + 1) * WIDTH] = seg_list[k]

        # Write triplet, seg_mask triplet, and camera intrinsics to files
        # Write multiple times if planning on testing online refinement
        if OPTIMIZE:
            for k in range(REPETITIONS):
                cv2.imwrite(OUTPUT_IMAGE_DIR + '/' + str(ct) + '.png', big_img)
                cv2.imwrite(OUTPUT_SEG_MASK_DIR + '/' + str(ct) + '-fseg.png', big_seg_img)
                f = open(OUTPUT_INTRINSICS_DIR + '/' + str(ct) + '_cam.txt', 'w')
                f.write(calib_representation)
                f.close()
                ct += 1
        else:
            cv2.imwrite(OUTPUT_IMAGE_DIR + '/' + str(ct) + '.png', big_img)
            cv2.imwrite(OUTPUT_SEG_MASK_DIR + '/' + str(ct) + '-fseg.png', big_seg_img)
            f = open(OUTPUT_INTRINSICS_DIR + '/' + str(ct) + '_cam.txt', 'w')
            f.write(calib_representation)
            f.close()
            ct += 1

if __name__ == "__main__":
    main()