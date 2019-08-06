from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import time
import sys
import csv

from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow as tf

from struct2depth import model
from struct2depth import nets
from struct2depth import reader
from struct2depth import util

# Isaac SDK imports
ROOT_DIR = os.path.abspath("/mnt/isaac_2019_2/")  # Root directory of the Isaac
sys.path.append(ROOT_DIR)
from engine.pyalice import *
import packages.ml
import apps.carter_sim_struct2depth
from differential_base_state import DifferentialBaseState

TIME_DELAY = 0.25
sample_numbers = 1


# Isaac Sim flags
flags.DEFINE_string('graph_filename', "apps/carter_sim_struct2depth/carter.graph.json",
                    'Where the isaac SDK app graph is stored')
flags.DEFINE_string('config_filename', "apps/carter_sim_struct2depth/carter.config.json",
                    'Where the isaac SDK app node configuration is stored')
FLAGS = flags.FLAGS


# Retrieve current robot linear and angular speed from Isaac Sim
def update_speed():
    with open('/mnt/isaac/apps/carter_sim_struct2depth/differential_base_speed/speed.csv') as speed_file:
        csv_reader = csv.reader(speed_file, delimiter=',')
        for row in csv_reader:
            if len(row) == 2:
                try:
                    float(row[0])
                    speed = float(row[0])
                except ValueError:
                    speed = 0
                try:
                    float(row[1])
                    angular_speed = float(row[1])
                except ValueError:
                    angular_speed = 0
        return speed, angular_speed

# Check if Isaac Sim bridge has samples
def has_samples(bridge):
    return bridge.get_sample_number() >= sample_numbers

# Create a training sample generator from Isaac Sim.
# bridge is the SampleAccumulator node that feeds image data
# Returns a generator function that yields batches of training examples.
def get_generator(self, bridge, img_processor):

    def _generator(self):
        # Infinitely generate training images
        while True:
            # Arrays to hold image batches
            image_batch = np.zeros((0, self.img_height, self.img_width * self.seq_length, 3))
            seg_mask_batch = np.zeros((0, self.img_height, self.img_width * self.seq_length, 3))

            # Wait until we get enough samples from Isaac
            while not has_samples(bridge):
                time.sleep(TIME_DELAY)
            # Retrieve current robot linear and angular speed
            speed, angular_speed = self.update_speed()

            # Only save image if the robot is moving or rotating above a threshold speed
            # Images below these thresholds do not have a great enough disparity for the network to learn depth.
            if speed > 0.1 or angular_speed > 0.15:
                images = []
                # Retrieve a total of (batch_size * seq_length) images
                for i in range(self.batch_size * self.seq_length):

                    # Wait for images to accumulate
                    while not self.has_samples(bridge):
                        time.sleep(TIME_DELAY)

                    # Acquire image
                    new_image = bridge.acquire_samples(self.sample_numbers)

                    # Add image to list
                    images.append(np.squeeze(new_image))

                    # Wait to increase disparity between images
                    time.sleep(TIME_DELAY)

                    # TODO: Turn seg mask generator into an Isaac node
                    # TODO: Fix MRCNN to work within another tf.Graph()
                    # Create wide image and segmentation triplets
                    if i != 0 and (i + 1) % self.seq_length == 0:
                        image_seq, seg_mask_seq = img_processor.process_image([images[i - 2],
                                                                               images[i - 1],
                                                                               images[i]])

                        if self.optimize:
                            repetitions = self.repetitions
                        else:
                            repetitions = 1
                        for j in range(repetitions):
                            # Add to total image lists; add repetitions if performing online refinement
                            image_batch = np.append(image_batch, np.expand_dims(image_seq, axis=0), axis=0)
                            seg_mask_batch = np.append(seg_mask_batch, np.expand_dims(seg_mask_seq, axis=0), axis=0)

                # TODO: Retrieve camera mat from Isaac instead of manually input
                intrinsics = np.array([[208., 0., 208.], [0., 113.778, 64.], [0., 0., 1.]])  # Scaled properly
                intrinsics_batch = np.repeat(intrinsics[None, :],
                                             repeats=self.batch_size if not self.optimize else self.repetitions,
                                             axis=0)  # Create batch

                if not self.optimize:
                    # Shuffle batch elements to reduce overfitting
                    np.random.seed(2)
                    np.random.shuffle(image_batch)
                    np.random.shuffle(seg_mask_batch)
                    np.random.shuffle(intrinsics_batch)

                # Yield batches
                yield {COLOR_IMAGE: np.array(image_batch),
                       SEG_MASK: np.array(seg_mask_batch),
                       INTRINSICS: intrinsics_batch}

    return lambda: _generator(self)

def get_dataset(self, bridge, img_processor):
    """Create a tf.data dataset which yields batches of samples for training.

  Args:
      bridge: the isaac sample accumulator node which we will acquire samples from

  Returns:
    A tf.data dataset which yields batches of training examples.
  """
    if self.optimize:
        dataset_size = self.repetitions
    else:
        dataset_size = self.batch_size

    dataset = tf.data.Dataset.from_generator(
        self.get_generator(bridge, img_processor), {
            COLOR_IMAGE: tf.float32,
            SEG_MASK: tf.uint8,
            INTRINSICS: tf.float32,
        }, {
            COLOR_IMAGE: (dataset_size, HEIGHT, TRIPLET_WIDTH, 3),
            SEG_MASK: (dataset_size, HEIGHT, TRIPLET_WIDTH, 3),
            INTRINSICS: (dataset_size, 3, 3),
        })

    return dataset

def main():
    # Create Isaac application.
    isaac_app = Application(app_filename="apps/carter_sim_struct2depth/carter_sim.app.json")
    # isaac_app = Application(name="carter_sim", modules=["map",
    #                                               "navigation",
    #                                               "perception",
    #                                               "planner",
    #                                               "viewers",
    #                                               "flatsim",
    #                                               "//packages/ml:ml"])

    # # Load config files
    # # isaac_app.load_config("apps/carter_sim_struct2depth/carter.config.json")
    # # isaac_app.load_config("apps/carter_sim_struct2depth/navigation.config.json")
    # isaac_app.load_config("apps/assets/maps/carter_warehouse_p.config.json")
    #
    # # Load graph files
    # # isaac_app.load_graph("apps/carter_sim_struct2depth/carter.graph.json")
    # # isaac_app.load_graph("apps/carter_sim_struct2depth/navigation.graph.json")
    # isaac_app.load_graph("apps/assets/maps/carter_warehouse_p.graph.json")
    # # isaac_app.load_graph("apps/carter_sim_struct2depth/base_control.graph.json")
    #
    # # Register custom Isaac codelets
    # isaac_app.register({"differential_base_state": DifferentialBaseState})
    #
    # node = isaac_app.find_node_by_name("CarterTrainingSamples")
    # bridge = packages.ml.SampleAccumulator(node)
    #
    # # Start the application and Sight server
    # # isaac_app.start_webserver()
    # isaac_app.start()
    # logging.info("Isaac application loaded")
    #
    # # Create image processor for generating triplets and seg masks
    # # img_processor = ImageProcessor()
    # logging.info("Image Processor created")
    #
    # # Create a Dataset and iterator from Isaac generator
    # # isaac_dataset = get_dataset(bridge, img_processor)
    # while True:
    #   # Wait until we get enough samples from Isaac
    #   while not bridge.get_sample_number() >= sample_numbers:
    #     print("Waiting for samples")
    #     time.sleep(TIME_DELAY)
    #
    #   print("Getting sample")
    #   image_and_labels = bridge.acquire_samples(sample_numbers)
    #   print(image_and_labels)



if __name__ == '__main__':
  main()