from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import tensorflow as tf
import numpy as np
import cv2

from isaac_app import create_isaac_app, start_isaac_app
from struct2depth.process_image import ImageProcessor
import time
import csv

# Root directory of the Isaac
ROOT_DIR = os.path.abspath("/mnt/isaac_2019_2")
sys.path.append(ROOT_DIR)
from engine.pyalice import *
import packages.ml
from differential_base_state import DifferentialBaseState

# Op names.
COLOR_IMAGE_NAME = 'rgb_image'
LABEL_IMAGE_NAME = 'sgementation_label'
INPUT_NAME = 'input'
OUTPUT_NAME = 'output'
SEQ_LENGTH = 3
STEPSIZE = 1
WIDTH = 416
HEIGHT = 128
TIME_DELAY = 0.4  # seconds

OUTPUT_DIR = 'synth_images'

# Number of samples Isaac Sim should accumulate
buffer_size = 1
sample_num = 1

flags = tf.app.flags
FLAGS = flags.FLAGS

# Create the application.
isaac_app = create_isaac_app()

# Startup the bridge to get data.
node = isaac_app.find_node_by_name("CarterTrainingSamples")
bridge = packages.ml.SampleAccumulator(node)

# Start the app
start_isaac_app(isaac_app)

img_processor = ImageProcessor()

count = 0
gct = 0
while True:
    # Retrieve rgb images from isaac sim
    while True:
        num = bridge.get_sample_number()
        if num >= buffer_size:
            break
        time.sleep(TIME_DELAY)
        print("waiting for samples: {}".format(num))

    # Retrieve differential base speed from file
    with open('/mnt/isaac_2019_2/apps/carter_sim_struct2depth/differential_base_speed/speed.csv') as speed_file:
        csv_reader = csv.reader(speed_file, delimiter=',')
        for row in csv_reader:
            speed = float(row[0])
            angular_speed = float(row[1])

    # Only save image if the robot is moving or rotating above a threshold speed
    # Images below these thresholds do not have a great enough disparity for the network to learn depth.
    if speed > 0.1 or angular_speed > 0.15:
        images = bridge.acquire_samples(sample_num)
        # print("{} Samples acquired".format(kSampleNumbers))
        while np.shape(images)[0] < SEQ_LENGTH:
            time.sleep(TIME_DELAY)
            images = np.concatenate((images, bridge.acquire_samples(sample_num)))

        # Create wide image and segmentation triplets
        image_seq = []
        seg_seq = []
        intrinsics = "208, 0, 208, 0, 113.778, 64, 0, 0, 1"
        for i in range(0, np.shape(images)[0] - 2):
            big_img, big_seg_img = img_processor.process_image(np.array([images[i][0],
                                                                         images[i + 1][0],
                                                                         images[i + 2][0]]))

            # Save to directory
            cv2.imwrite('/mnt/isaac/apps/carter_sim_struct2depth/sim_images/sim_images_40_delay/{}.png'.format(count),
                        np.uint8(big_img))
            cv2.imwrite('/mnt/isaac/apps/carter_sim_struct2depth/sim_seg_masks/{}-fseg.png'.format(count), big_seg_img)
            f = open('/mnt/isaac/apps/carter_sim_struct2depth/sim_intrinsics/{}.csv'.format(count), 'w')
            f.write(intrinsics)
            f.close()

            print('saved images: {}'.format(count))

            count += 1
