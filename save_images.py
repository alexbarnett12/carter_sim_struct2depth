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

# Root directory of the Isaac
ROOT_DIR = os.path.abspath("/mnt/isaac_2019_2")
sys.path.append(ROOT_DIR)
from engine.pyalice import *
import packages.ml

# Op names.
COLOR_IMAGE_NAME = 'rgb_image'
LABEL_IMAGE_NAME = 'sgementation_label'
INPUT_NAME = 'input'
OUTPUT_NAME = 'output'
STEPSIZE = 1
WIDTH = 416
HEIGHT = 128

OUTPUT_DIR = 'synth_images'

# Number of samples to acquire in batch
kSampleNumbers = 1

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
while True:
    while True:
        num = bridge.get_sample_number()
        if num >= kSampleNumbers:
            break
        time.sleep(1.0)
        print("waiting for samples: {}".format(num))

    images = bridge.acquire_samples(kSampleNumbers)
    print("{} Samples acquired".format(kSampleNumbers))

    while np.shape(images)[0] < 3:
        time.sleep(.25)
        images = np.concatenate((images, bridge.acquire_samples(kSampleNumbers)))

    for i in range(kSampleNumbers):
        # Create segmentation mask
        img = cv2.resize(images[i][0], (WIDTH, HEIGHT))
        # seg_img = img_processor.create_mask(img)
        seg_img = np.zeros(shape=(HEIGHT, WIDTH, 3))

        # Convert to uint8
        img = img.astype(np.uint8)

        # Save to directory
        cv2.imwrite('/mnt/isaac/apps/carter_sim_struct2depth/synth_images_single/{}.png'.format(count), img)
        # cv2.imwrite('/mnt/isaac/apps/carter_sim_struct2depth/synth_images/{}-fseg.png'.format(count), seg_img)
        print('saved images')

        count += 1



