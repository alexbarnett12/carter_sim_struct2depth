from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import tensorflow as tf
import numpy as np
import cv2

from struct2depth.process_image import ImageProcessor
import time
import csv

# Root directory of the Isaac
ROOT_DIR = os.path.abspath("/mnt/isaac")
sys.path.append(ROOT_DIR)

from engine.pyalice import *
import packages.ml
from differential_base_state import DifferentialBaseState

# Op names.
COLOR_IMAGE_NAME = 'rgb_image'
LABEL_IMAGE_NAME = 'sgementation_label'
INPUT_NAME = 'input'
OUTPUT_NAME = 'output'
STEPSIZE = 1
WIDTH = 416
HEIGHT = 128
TIME_DELAY = 0.25 # seconds

OUTPUT_DIR = 'synth_images'

# Number of samples to acquire at a time from Isaac sim
kSampleNumbers = 1

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('graph_filename', "apps/carter_sim_struct2depth/carter_save.graph.json",
                    'Where the isaac SDK app graph is stored')
flags.DEFINE_string('config_filename', "apps/carter_sim_struct2depth/carter_save.config.json",
                    'Where the isaac SDK app node configuration is stored')

# Create the application.
app = Application(name="carter_sim", modules=["map",
                                              "navigation",
                                              "perception",
                                              "planner",
                                              "viewers",
                                              "flatsim",
                                              "//packages/ml:ml"])

app.load_config(FLAGS.config_filename)
app.load_config("apps/carter_sim_struct2depth/navigation.config.json")
app.load_config("apps/assets/maps/carter_warehouse_p.config.json")
app.load_graph(FLAGS.graph_filename)
app.load_graph("apps/carter_sim_struct2depth/navigation.graph.json")
app.load_graph("apps/assets/maps/carter_warehouse_p.graph.json")
app.load_graph("apps/carter_sim_struct2depth/base_control.graph.json")

# Register custom PyCodelet
app.register({"differential_base_state": DifferentialBaseState})

# Startup the bridge to get data.
node = app.find_node_by_name("CarterTrainingSamples")
bridge = packages.ml.SampleAccumulator(node)
app.start_webserver()
app.start()

img_processor = ImageProcessor()

count = 0
while True:
    # Retrieve rgb images from isaac sim
    while True:
        num = bridge.get_sample_number()
        if num >= kSampleNumbers:
            break
        time.sleep(1.0)
        print("waiting for samples: {}".format(num))

    images = bridge.acquire_samples(kSampleNumbers)
    # print("{} Samples acquired".format(kSampleNumbers))

    # Retrieve differential base speed from file
    # speed = []
    # angular_speed = []
    with open('/mnt/isaac/apps/carter_sim_struct2depth/differential_base_speed/speed.csv') as speed_file:
        csv_reader = csv.reader(speed_file, delimiter=',')
        for row in csv_reader:
            speed = float(row[0])
            angular_speed = float(row[1])
    print("{}, {}".format(speed, angular_speed))

    # Only save image if the robot is moving or rotating above a threshold speed
    # Images below these thresholds do not have a great enough disparity for the network to learn depth.
    if speed > 0.1 or angular_speed > 0.15:
        while np.shape(images)[0] < 3:
            time.sleep(TIME_DELAY)
            images = np.concatenate((images, bridge.acquire_samples(kSampleNumbers)))

        # Create wide image and segmentation triplets
        image_seq = []
        seg_seq = []
        intrinsics = "208, 0, 208, 0, 113.778, 64, 0, 0, 1"
        for i in range(0, np.shape(images)[0] - 2):
            big_img, big_seg_img = img_processor.process_image(np.array([images[i][0],
                                                           images[i + 1][0],
                                                           images[i + 2][0]]))

            # Save to directory
            cv2.imwrite('/mnt/isaac/apps/carter_sim_struct2depth/synth_images_25_delay/{}.png'.format(count), np.uint8(big_img))
            # cv2.imwrite('/mnt/isaac/apps/carter_sim_struct2depth/synth_images_seg/{}-fseg.png'.format(count), big_seg_img)
            # f = open('/mnt/isaac/apps/carter_sim_struct2depth/synth_images_intrinsics/{}.csv'.format(count), 'w')
            # f.write(intrinsics)
            # f.close()

            print('saved images')

            count += 1
