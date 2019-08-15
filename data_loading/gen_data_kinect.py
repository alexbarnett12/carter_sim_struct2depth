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
import csv
import numpy as np
import cv2
import glob

# Segmentation mask generation
# from data_loading.gen_masks_kitti import MaskGenerator
# from struct2depth.alignment import align
from data_loading.gen_train_txt_kinect import generate_train_txt_kinect

SEQ_LENGTH = 3
WIDTH = 416
HEIGHT = 128
STEPSIZE = 1
INPUT_DIR = '/mnt/test_images/warehouse_8_15/color'
OUTPUT_DIR = '/mnt/train_data/warehouse_8_15'
KINECT_CALIBRATION_FILE = 'kinect_camera_intrinsics.csv'


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


# Create a segmentation mask generator
# mask_generator = MaskGenerator()

if not OUTPUT_DIR.endswith('/'):
    OUTPUT_DIR = OUTPUT_DIR + '/'

if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

# Load Kinect camera intrinsics
calib_file = INPUT_DIR + '/' + KINECT_CALIBRATION_FILE
camera_mat = np.zeros(shape=(3, 3))
with open(calib_file) as file:
    csv_reader = csv.reader(file, delimiter=',')
    for row in csv_reader:
        camera_mat[0, 0] = float(row[0])
        camera_mat[0, 2] = float(row[2])
        camera_mat[1, 1] = float(row[4])
        camera_mat[1, 2] = float(row[5])
        camera_mat[2, 2] = 1

ct = 0
for d in glob.glob(INPUT_DIR + '/*/'):
    files = sorted(glob.glob(d + '/*.png'))
    print('Processing sequence: {}'.format(d))
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
        cv2.imwrite(OUTPUT_DIR + imgnum + '.png', big_img)
        cv2.imwrite(OUTPUT_DIR + imgnum + '-fseg.png', big_seg_img)
        f = open(OUTPUT_DIR + imgnum + '_cam.csv', 'w')
        f.write(calib_representation)
        f.close()
        ct += 1

# Generate train txt file
generate_train_txt_kinect()

