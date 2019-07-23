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

# Root directory of the Isaac
ROOT_DIR = os.path.abspath("/mnt/isaac")
sys.path.append(ROOT_DIR)

from engine.pyalice import *
import packages.ml
from pinhole_to_tensor import PinholeToTensor
import capnp

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

# Startup the bridge to get data.
node = app.find_node_by_name("CarterTrainingSamples")
bridge = packages.ml.SampleAccumulator(node)
app.start_webserver()
app.start()

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



