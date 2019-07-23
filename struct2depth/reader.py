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

"""Reads data that is produced by dataset/gen_data.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
from absl import logging
import tensorflow as tf
import numpy as np
import cv2
import csv

# from struct2depth import util
# from struct2depth.process_image import ImageProcessor
import util
from process_image import ImageProcessor
import glob

import time

gfile = tf.gfile
AUTOTUNE = tf.data.experimental.AUTOTUNE

QUEUE_SIZE = 2000
QUEUE_BUFFER = 3
# See nets.encoder_resnet as reference for below input-normalizing constants.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_SD = (0.229, 0.224, 0.225)
FLIP_RANDOM = 'random'  # Always perform random flipping.
FLIP_ALWAYS = 'always'  # Always flip image input, used for test augmentation.
FLIP_NONE = 'none'  # Always disables flipping.

'''Isaac SDK code'''
# Root directory of the Isaac
ROOT_DIR = os.path.abspath("/mnt/isaac/")
sys.path.append(ROOT_DIR)

# from engine.pyalice import *
import packages.ml
# from pinhole_to_tensor import PinholeToTensor

# Op names.
COLOR_IMAGE = 'image'
SEG_MASK = 'segmentation_mask'
CAMERA_MAT = 'camera_matrix'
INTRINSICS = 'camera_matrix'
INTRINSICS_INV = 'camera_matrix_inverse'
IMAGE_NORM = 'imagenet_norm'
SEQ_LENGTH = 3
HEIGHT = 128
WIDTH = 416
TRIPLET_WIDTH = WIDTH * SEQ_LENGTH
INPUT_NAME = 'input'
OUTPUT_NAME = 'output'
STEPSIZE = 1
REPEAT = 10000000

# Number of samples to acquire in batch
kSampleNumbers = 1


# Helper function for splitting datasets into sub-datasets.
def split_datasets(dataset):
    subsets = {}
    names = list(dataset.output_shapes.keys())
    for name in names:
        subsets[name] = dataset.map(lambda x: x[name])

    return subsets

class DataReader(object):
    """Reads stored sequences which are produced by dataset/gen_data.py."""

    def __init__(self, data_dir, batch_size, img_height, img_width, seq_length,
                 num_scales, file_extension, random_scale_crop, flipping_mode,
                 random_color, imagenet_norm, shuffle, input_file='train', isaac_app=None):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width
        self.seq_length = seq_length
        self.num_scales = num_scales
        self.file_extension = file_extension
        self.random_scale_crop = random_scale_crop
        self.flipping_mode = flipping_mode
        self.random_color = random_color
        self.imagenet_norm = imagenet_norm
        self.shuffle = shuffle
        self.input_file = input_file
        self.isaac_app = isaac_app
        self.steps_per_epoch = 1000
        self.speed = 1000
        self.angular_speed = 1000

    def update_speed(self):
        with open('/mnt/isaac/apps/carter_sim_struct2depth/differential_base_speed/speed.csv') as speed_file:
                csv_reader = csv.reader(speed_file, delimiter=',')
                for row in csv_reader:
                    self.speed = float(row[0])
                    self.angular_speed = float(row[1])
                    # print("{}, {}".format(self.speed, self.angular_speed))

    def has_samples(self, bridge):
        return bridge.get_sample_number() >= kSampleNumbers

    def get_generator(self, bridge, img_processor):
        """Create a training sample generator.

        Args:
          bridge: the isaac sample accumulator node which we will acquire samples from

        Returns:
          A generator function which yields a single training example.
      """

        def _generator(self):
            # Indefinitely yield samples.
            while True:

                # Wait until we get enough samples from Isaac
                while not self.has_samples(bridge):
                    time.sleep(1.0)
                # while True:
                #     num = bridge.get_sample_number()
                #     if num >= kSampleNumbers:
                #         break
                #     time.sleep(1.0)
                #     # logging.info("waiting for enough samples: {}".format(num))
                images = bridge.acquire_samples(kSampleNumbers)

                # Retrieve current robot linear and angular speed
                self.update_speed()
                # print("{}, {}".format(self.speed, self.angular_speed))

                # Only save image if the robot is moving or rotating above a threshold speed
                # Images below these thresholds do not have a great enough disparity for the network to learn depth.
                if self.speed > 0.1 or self.angular_speed > 0.15:
                    while np.shape(images)[0] < 3:
                        time.sleep(.25)
                        while not self.has_samples(bridge):
                            time.sleep(.25)
                        # print(np.shape(images))
                        test = bridge.acquire_samples(kSampleNumbers)
                        # print(np.shape(test))
                        images = np.concatenate((images, test))

                    # # Try to acquire a sample.
                    # # If none are available, then wait for a bit so we do not spam the app.
                    # # Samples are received as (128 x 416 x 3) float32 images in the range of 0-255.
                    # # They will be pre-processed to be in the range 0-1 for training.
                    # sample = bridge.acquire_samples(kSampleNumbers)
                    # if not sample:
                    #     time.sleep(1)
                    #     continue

                    # print(np.shape(sample))

                    # Create wide image and segmentation triplets
                    # TODO: Turn seg mask generator into an Isaac node
                    # TODO: Fix MRCNN to work within another tf.Graph()
                    image_seq, seg_mask_seq = img_processor.process_image(np.array([images[0][0],
                                                                                    images[1][0],
                                                                                    images[2][0]]))

                    # TODO: Retrieve camera mat from Isaac
                    intrinsics = np.array([[208, 0, 208], [0, 113.778, 64], [0, 0, 1]], dtype=np.float32) # Scaled properly

                    yield {COLOR_IMAGE: image_seq,
                           SEG_MASK: seg_mask_seq,
                           INTRINSICS: intrinsics}

        return lambda: _generator(self)

    def get_dataset(self, bridge, img_processor):
        """Create a tf.data dataset which yields batches of samples for training.

      Args:
          bridge: the isaac sample accumulator node which we will acquire samples from

      Returns:
        A tf.data dataset which yields batches of training examples.
      """

        dataset = tf.data.Dataset.from_generator(
            self.get_generator(bridge, img_processor), {
                COLOR_IMAGE: tf.float32,
                SEG_MASK: tf.uint8,
                INTRINSICS: tf.float32,
            }, {
                COLOR_IMAGE: (HEIGHT, TRIPLET_WIDTH, 3),
                SEG_MASK: (HEIGHT, TRIPLET_WIDTH, 3),
                INTRINSICS: (3, 3),
            })

        return dataset

    def read_data(self):
        """Provides images and camera intrinsics."""
        with tf.name_scope('data_loading'):

            # Startup the sample accumulator bridge to get Isaac Sim data
            node = self.isaac_app.find_node_by_name("CarterTrainingSamples")
            bridge = packages.ml.SampleAccumulator(node)

            # Start the application and Sight server
            self.isaac_app.start_webserver()
            self.isaac_app.start()
            logging.info("Isaac application loaded")

            # Create image processor for generating triplets and seg masks
            img_processor = ImageProcessor()
            logging.info("Image Processor created")

            # Create a Dataset and iterator from Isaac generator
            isaac_dataset = self.get_dataset(bridge, img_processor)

            # Split dataset into image, seg mask, and intrinsics datasets
            isaac_subsets = split_datasets(isaac_dataset)
            image_ds = isaac_subsets[COLOR_IMAGE]
            seg_ds = isaac_subsets[SEG_MASK]
            intrinsics_ds = isaac_subsets[INTRINSICS]

            # Make image dataset from saved directory
            # AUTOTUNE = tf.data.experimental.AUTOTUNE
            # data_root = '/mnt/isaac/apps/carter_sim_struct2depth/synth_images_50_delay'
            # data_root_seg = '/mnt/isaac/apps/carter_sim_struct2depth/synth_images_seg'
            # data_root_intrinsics = '/mnt/isaac/apps/carter_sim_struct2depth/synth_images_intrinsics'
            # all_image_paths = list(glob.glob(data_root + '/*'))
            # all_image_paths_seg = list(glob.glob(data_root_seg + '/*'))
            # all_image_paths_intrinsics = list(glob.glob(data_root_intrinsics + '/*'))
            #
            # # Raw image triplets
            # path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
            # image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
            #
            # # Seg masks
            # path_ds_seg = tf.data.Dataset.from_tensor_slices(all_image_paths_seg)
            # seg_ds = path_ds_seg.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
            #
            # # Camera intrinsics
            # record_defaults = [tf.float32] * 9
            # intrinsics_ds = tf.data.experimental.CsvDataset(all_image_paths_intrinsics, record_defaults)  # Dataset of .csv lines
            # intrinsics_ds = intrinsics_ds.map(lambda *x: tf.convert_to_tensor(x))  # Convert to tensors
            # intrinsics_ds = intrinsics_ds.map(lambda x: tf.reshape(x, [3, 3]))
            # logging.info("Datasets loaded")

            # data_dict = dataset.make_one_shot_iterator().get_next()
            # logging.info("Dataset from generator created")
            # print("Dataset: {}".format(dataset))
            #
            # # Extract image, seg mask, and camera matrix
            # image_seq = data_dict[COLOR_IMAGE]
            # # image_stack_norm = data_dict[IMAGE_NORM]
            # seg_mask_seq = data_dict[SEG_MASK]
            # intrinsics = data_dict[INTRINSICS]
            # # intrinsics_inv = data_dict[INTRINSICS_INV]
            # logging.info("Dictionary elements extracted")

            # print("Dataset: {}".format(isaac_dataset))
            print("Image dataset: {}".format(image_ds))
            print("Seg mask dataset: {}".format(seg_ds))
            print("Intrinsics dataset: {}".format(intrinsics_ds))

        # TODO: Replace with tf.Dataset pre-processing pipeline
        with tf.name_scope('preprocessing'):

            # Scale image values from 0-255 to 0-1
            image_ds = image_ds.map(lambda x: x / 255.0, num_parallel_calls=AUTOTUNE)

            # Randomly augment colorspace
            if self.random_color:
                with tf.name_scope('image_augmentation'):
                    image_ds = image_ds.map(self.augment_image_colorspace, num_parallel_calls=AUTOTUNE)
                logging.info("Image successfully augmented")

            # Unpack triplets; each tensor is unpacked into a stack of three images
            image_stack_ds = image_ds.map(self.unpack_images, num_parallel_calls=AUTOTUNE)
            seg_stack_ds = seg_ds.map(self.unpack_images, num_parallel_calls=AUTOTUNE)
            logging.info("Images unpacked")

            # Randomly flip images
            if self.flipping_mode != FLIP_NONE:
                random_flipping = (self.flipping_mode == FLIP_RANDOM)
                with tf.name_scope('image_augmentation_flip'):
                    # Create image flipper
                    flipper = Flipper(image_width=self.img_width, randomized=random_flipping)

                    # Flip images, seg masks, and intrinsics randomly or completely, depending on random_flipping
                    image_stack_ds = image_stack_ds.map(flipper.flip_images, num_parallel_calls=AUTOTUNE)
                    seg_stack_ds = seg_stack_ds.map(flipper.flip_images, num_parallel_calls=AUTOTUNE)
                    intrinsics_ds = intrinsics_ds.map(flipper.flip_intrinsics, num_parallel_calls=AUTOTUNE)

                    logging.info("Images flipped and intrinsics adjusted")

            # Randomly scale and crop images
            if self.random_scale_crop:
                with tf.name_scope('image_augmentation_scale_crop'):
                    # Create image cropper
                    cropper = Cropper(image_width=self.img_width, image_height=self.img_height)

                    # Crop images, seg masks, and intrinsics
                    image_stack_ds = image_stack_ds.map(cropper.scale_and_crop_image, num_parallel_calls=AUTOTUNE)
                    seg_stack_ds = seg_stack_ds.map(cropper.scale_and_crop_image, num_parallel_calls=AUTOTUNE)
                    intrinsics_ds = intrinsics_ds.map(cropper.scale_and_crop_intrinsics, num_parallel_calls=AUTOTUNE)

                    logging.info("Images scaled and cropped")

            # Adjust camera intrinsics to the correct scale and compute the inverse
            with tf.name_scope('multi_scale_intrinsics'):
                intrinsics_ds = intrinsics_ds.map(self.get_multi_scale_intrinsics, num_parallel_calls=AUTOTUNE)
                intrinsics_inv = intrinsics_ds.map(lambda x: tf.matrix_inverse(x), num_parallel_calls=AUTOTUNE)

                logging.info("Multi scale intrinsics received")

            # Normalize images by the Imagenet standard
            if self.imagenet_norm:
                image_stack_norm = image_stack_ds.map(self.normalize_by_imagenet, num_parallel_calls=AUTOTUNE)
                logging.info("Imagenet norm used")
            else:
                image_stack_norm = image_stack_ds
                logging.info("Imagenet norm not used")

        # Wait until we get enough samples from Isaac
        while True:
            num = bridge.get_sample_number()
            if num >= kSampleNumbers:
                break
            time.sleep(1.0)
            logging.info("waiting for enough samples: {}".format(num))

                # Shuffle and batch datasets
        with tf.name_scope('batching'):
            if self.shuffle:
                image_stack_ds = image_stack_ds.shuffle(buffer_size=self.batch_size, seed=2).batch(self.batch_size, drop_remainder=True)
                image_stack_norm = image_stack_norm.shuffle(buffer_size=self.batch_size, seed=2).batch(self.batch_size, drop_remainder=True)
                seg_stack_ds = seg_stack_ds.shuffle(buffer_size=self.batch_size, seed=2).batch(self.batch_size, drop_remainder=True)
                intrinsics_ds = intrinsics_ds.shuffle(buffer_size=self.batch_size, seed=2).batch(self.batch_size, drop_remainder=True)
                intrinsics_inv = intrinsics_inv.shuffle(buffer_size=self.batch_size, seed=2).batch(self.batch_size, drop_remainder=True)

                # (image_stack_ds, image_stack_norm, seg_stack, intrinsics,
                #  intrinsics_inv) = tf.train.shuffle_batch(
                #     [image_stack_ds, image_stack_norm, seg_stack_ds, intrinsics_ds,
                #      intrinsics_inv],
                #     batch_size=self.batch_size,
                #     capacity=QUEUE_SIZE + QUEUE_BUFFER * self.batch_size,
                #     min_after_dequeue=QUEUE_SIZE)
            else:
                image_stack_ds = image_stack_ds.batch(self.batch_size)
                image_stack_norm = image_stack_norm.batch(self.batch_size)
                seg_stack_ds = seg_stack_ds.batch(self.batch_size)
                intrinsics_ds = intrinsics_ds.batch(self.batch_size)
                intrinsics_inv = intrinsics_inv.batch(self.batch_size)
                # (image_stack_ds, image_stack_norm, seg_stack, intrinsics,
                #  intrinsics_inv) = tf.train.batch(
                #     [image_stack_ds, image_stack_norm, seg_stack_ds, intrinsics_ds,
                #      intrinsics_inv],
                #     batch_size=self.batch_size,
                #     capacity=QUEUE_SIZE + QUEUE_BUFFER * self.batch_size,
                #     min_after_dequeue=QUEUE_SIZE)

            # # Extract image, seg mask, and camera matrix
            # image_seq = data_dict[COLOR_IMAGE]
            # # image_stack_norm = data_dict[IMAGE_NORM]
            # seg_mask_seq = data_dict[SEG_MASK]
            # intrinsics = data_dict[INTRINSICS]
            # # intrinsics_inv = data_dict[INTRINSICS_INV]
            # logging.info("Dictionary elements extracted")


        # Create iterators over datasets
        image_it = image_stack_ds.make_one_shot_iterator().get_next()
        image_norm_it = image_stack_norm.make_one_shot_iterator().get_next()
        seg_it = seg_stack_ds.make_one_shot_iterator().get_next()
        intrinsics_it = intrinsics_ds.make_one_shot_iterator().get_next()
        intrinsics_inv_it = intrinsics_inv.make_one_shot_iterator().get_next()
        print(image_it)
        print(image_norm_it)
        print(seg_it)
        print(intrinsics_it)
        print(intrinsics_inv_it)

        # # Shuffle and batch datasets
        # with tf.name_scope('batching'):
        #     if self.shuffle:
        #         (image_stack_ds, image_stack_norm, seg_stack, intrinsics,
        #          intrinsics_inv) = tf.train.shuffle_batch(
        #             [image_stack_ds, image_stack_norm, seg_stack_ds, intrinsics_ds,
        #              intrinsics_inv],
        #             batch_size=self.batch_size,
        #             capacity=QUEUE_SIZE + QUEUE_BUFFER * self.batch_size,
        #             min_after_dequeue=QUEUE_SIZE)
        #     else:
        #         (image_stack_ds, image_stack_norm, seg_stack, intrinsics,
        #          intrinsics_inv) = tf.train.batch(
        #             [image_stack_ds, image_stack_norm, seg_stack_ds, intrinsics_ds,
        #              intrinsics_inv],
        #             batch_size=self.batch_size,
        #             capacity=QUEUE_SIZE + QUEUE_BUFFER * self.batch_size,
        #             min_after_dequeue=QUEUE_SIZE)


        #     # Scale image values from 0-255 to 0-1
        #     image_seq = image_seq / 255.0
        #
        #     # Randomly augment colorspace
        #     if self.random_color:
        #         with tf.name_scope('image_augmentation'):
        #             image_seq = self.augment_image_colorspace(image_seq)
        #         logging.info("Image successfully augmented")
        #     print("Image dataset: {}".format(image_seq))
        #     print("Seg mask dataset: {}".format(seg_mask_seq))
        #     print("Intrinsics dataset: {}".format(intrinsics))
        #
        #     # Unpack wide images into three stacked images
        #     image_stack = self.unpack_images(image_seq)
        #     seg_stack = self.unpack_images(seg_mask_seq)
        #     logging.info("Images unpacked")
        #     print("Image dataset: {}".format(image_stack))
        #     print("Seg mask dataset: {}".format(seg_stack))
        #     print("Intrinsics dataset: {}".format(intrinsics))
        #
        #     # Randomly flip images
        #     if self.flipping_mode != FLIP_NONE:
        #         random_flipping = (self.flipping_mode == FLIP_RANDOM)
        #         with tf.name_scope('image_augmentation_flip'):
        #             image_stack, seg_stack, intrinsics = self.augment_images_flip(
        #                 image_stack, seg_stack, intrinsics,
        #                 randomized=random_flipping)
        #             logging.info("Images flipped")
        #
        #     # Randomly scale and crop images
        #     if self.random_scale_crop:
        #         with tf.name_scope('image_augmentation_scale_crop'):
        #             image_stack, seg_stack, intrinsics = self.augment_images_scale_crop(
        #                 image_stack, seg_stack, intrinsics, self.img_height,
        #                 self.img_width)
        #         logging.info("Images scaled and cropped")
        #
        #     # Adjust camera intrinsics to the correct scale and compute the inverse
        #     with tf.name_scope('multi_scale_intrinsics'):
        #         intrinsics = self.get_multi_scale_intrinsics(intrinsics,
        #                                                      self.num_scales)
        #         intrinsics.set_shape([self.num_scales, 3, 3])
        #         intrinsics_inv = tf.matrix_inverse(intrinsics)
        #         intrinsics_inv.set_shape([self.num_scales, 3, 3])
        #         logging.info("Multi scale intrinsics received")
        #
        #     # Subtract Imagenet norm
        #     if self.imagenet_norm:
        #         im_mean = tf.tile(
        #             tf.constant(IMAGENET_MEAN), multiples=[self.seq_length])
        #         im_sd = tf.tile(
        #             tf.constant(IMAGENET_SD), multiples=[self.seq_length])
        #         image_stack_norm = (image_stack - im_mean) / im_sd
        #         logging.info("Imagenet norm used")
        #     else:
        #         image_stack_norm = image_stack
        #         logging.info("Imagenet norm not used")
        #
        #     # Wait until we get enough samples from Isaac
        #     while True:
        #         num = bridge.get_sample_number()
        #         if num >= kSampleNumbers:
        #             break
        #         time.sleep(1.0)
        #         logging.info("waiting for enough samples: {}".format(num))
        #
        # with tf.name_scope('batching'):
        #     if self.shuffle:
        #         (image_stack, image_stack_norm, seg_stack, intrinsics,
        #          intrinsics_inv) = tf.train.shuffle_batch(
        #             [image_stack, image_stack_norm, seg_stack, intrinsics,
        #              intrinsics_inv],
        #             batch_size=self.batch_size,
        #             capacity=QUEUE_SIZE + QUEUE_BUFFER * self.batch_size,
        #             min_after_dequeue=QUEUE_SIZE)
        #     else:
        #         (image_stack, image_stack_norm, seg_stack, intrinsics,
        #          intrinsics_inv) = tf.train.batch(
        #             [image_stack, image_stack_norm, seg_stack, intrinsics,
        #              intrinsics_inv],
        #             batch_size=self.batch_size,
        #             capacity=QUEUE_SIZE + QUEUE_BUFFER * self.batch_size,
        #             min_after_dequeue=QUEUE_SIZE)
        logging.info("Dataset successfuly processed")
        logging.info("Final dimensional check")
        # print(dataset)
        # print("Image dataset: {}".format(image_stack))
        # print("Image norm: {}".format(image_stack_norm))
        # print("Seg mask dataset: {}".format(seg_stack))
        # print("Intrinsics dataset: {}".format(intrinsics))
        # print("Intrinsics inverse dataset: {}".format(intrinsics_inv))
        return (image_it,
                image_norm_it,
                seg_it,
                intrinsics_it,
                intrinsics_inv_it)
        # return (image_stack, image_stack_norm, seg_stack, intrinsics,
        #         intrinsics_inv)


    # Unpack image triplet from [h, w * seq_length, 3] -> [h, w, 3 * seq_length] image stack.
    def unpack_images(self, image_seq):
        with tf.name_scope('unpack_images'):
            image_list = [
                image_seq[:, i * self.img_width:(i + 1) * self.img_width, :]
                for i in range(self.seq_length)
            ]
            image_stack = tf.concat(image_list, axis=2)
            image_stack.set_shape([self.img_height, self.img_width, self.seq_length * 3])
        return image_stack

    # Randomly augment the brightness contrast, saturation, and hue of the image.
    # Provides more variety in training set to avoid overfitting.
    def augment_image_colorspace(self, image_stack):
        image_stack_aug = image_stack
        # Randomly shift brightness.
        apply_brightness = tf.less(tf.random_uniform(
            shape=[], minval=0.0, maxval=1.0, dtype=tf.float32), 0.5)
        image_stack_aug = tf.cond(
            apply_brightness,
            lambda: tf.image.random_brightness(image_stack_aug, max_delta=0.1),
            lambda: image_stack_aug)

        # Randomly shift contrast.
        apply_contrast = tf.less(tf.random_uniform(
            shape=[], minval=0.0, maxval=1.0, dtype=tf.float32), 0.5)
        image_stack_aug = tf.cond(
            apply_contrast,
            lambda: tf.image.random_contrast(image_stack_aug, 0.85, 1.15),
            lambda: image_stack_aug)

        # Randomly change saturation.
        apply_saturation = tf.less(tf.random_uniform(
            shape=[], minval=0.0, maxval=1.0, dtype=tf.float32), 0.5)
        image_stack_aug = tf.cond(
            apply_saturation,
            lambda: tf.image.random_saturation(image_stack_aug, 0.85, 1.15),
            lambda: image_stack_aug)

        # Randomly change hue.
        apply_hue = tf.less(tf.random_uniform(
            shape=[], minval=0.0, maxval=1.0, dtype=tf.float32), 0.5)
        image_stack_aug = tf.cond(
            apply_hue,
            lambda: tf.image.random_hue(image_stack_aug, max_delta=0.1),
            lambda: image_stack_aug)

        # Clip image to be between 0 and 1
        image_stack_aug = tf.clip_by_value(image_stack_aug, 0, 1)
        return image_stack_aug


    # Creates multi scale intrinsics based off number of scales provided.
    def get_multi_scale_intrinsics(self, intrinsics):
        intrinsics_multi_scale = []
        # Scale the intrinsics accordingly for each scale
        for s in range(self.num_scales):
            fx = intrinsics[0, 0] / (2 ** s)
            fy = intrinsics[1, 1] / (2 ** s)
            cx = intrinsics[0, 2] / (2 ** s)
            cy = intrinsics[1, 2] / (2 ** s)
            intrinsics_multi_scale.append(make_intrinsics_matrix(fx, fy, cx, cy))
        intrinsics_multi_scale = tf.stack(intrinsics_multi_scale)
        return intrinsics_multi_scale

    # Normalize the image by the Imagenet mean and standard deviation.
    # This aligns the training dataset with the pre-trained Imagenet model, and allows
    # for standardized evaluation of test set.
    def normalize_by_imagenet(self, image_stack):
        # Copy constant values multiple times to fill up a tensor of length SEQ_LENGTH * len(IMAGENET_MEAN)
        im_mean = tf.tile(
            tf.constant(IMAGENET_MEAN), multiples=[SEQ_LENGTH])
        im_sd = tf.tile(
            tf.constant(IMAGENET_SD), multiples=[SEQ_LENGTH])
        return (image_stack - im_mean) / im_sd

# Class for flipping images, seg masks, and intrinsics
# Provides greater variety in training dataset to avoid overfitting
class Flipper:
    def __init__(self,
                 image_width=416,
                 randomized=True):
        self.randomized = randomized
        self.img_width = image_width  # Assumes all input images are the same size

    # Randomly flips the image horizontally.
    def flip_images(self, image_stack):

        if self.randomized:
            # Generate random number. Seed provided to ensure that image,
            # seg mask, and intrinsics are paired
            prob = tf.random_uniform(shape=[], minval=0.0, maxval=1.0,
                                     dtype=tf.float32, seed=2)
            predicate = tf.less(prob, 0.5)
        else:
            predicate = tf.less(0.0, 0.5)

        return tf.cond(predicate,
                       lambda: tf.image.flip_left_right(image_stack),
                       lambda: image_stack)

    # Randomly flips intrinsics.
    def flip_intrinsics(self, intrinsics):

        def flip(intrinsics, in_w):
            fx = intrinsics[0, 0]
            fy = intrinsics[1, 1]
            cx = in_w - intrinsics[0, 2]
            cy = intrinsics[1, 2]
            return make_intrinsics_matrix(fx, fy, cx, cy)

        if self.randomized:
            # Generate random probability. Seed provided to ensure that image,
            # seg mask, and intrinsics are paired
            prob = tf.random_uniform(shape=[], minval=0.0, maxval=1.0,
                                     dtype=tf.float32, seed=2)
            predicate = tf.less(prob, 0.5)
        else:
            predicate = tf.less(0.0, 0.5)

        return tf.cond(predicate,
                       lambda: flip(intrinsics, self.img_width),
                       lambda: intrinsics)


# Class for cropping images. First scales them to provide a greater area
# to crop from.
class Cropper:
    def __init__(self,
                 image_width=416,
                 image_height=128):
        self.orig_img_width = image_width
        self.orig_img_height = image_height
        self.scaled_img_width = image_width # New image width before cropping to original dimensions
        self.scaled_img_height = image_height # New image height before cropping to original dimensions

    # Scale and crop image.
    def scale_and_crop_image(self, im):
        im = self.scale_images_randomly(im)
        im = self.crop_images_randomly(im)
        return im


    # Scale image randomly. Seed used to match corresponding image, seg mask, and intrinsics.
    # Scales to a random number with greater dimensions than the original.
    def scale_images_randomly(self, im):
        scaling = tf.random_uniform([2], 1, 1.15, seed=2)
        x_scaling = scaling[0]
        y_scaling = scaling[1]
        self.scaled_img_height = tf.cast(self.orig_img_height * y_scaling, dtype=tf.int32)
        self.scaled_img_width = tf.cast(self.orig_img_width * x_scaling, dtype=tf.int32)

        # Add batch to resize the image area, then revert back
        im = tf.expand_dims(im, 0)
        im = tf.image.resize_area(im, [self.scaled_img_height, self.scaled_img_width])
        return im[0]


    # Crop image randomly. Outputs image with its original height and width.
    def crop_images_randomly(self, im):
        offset_y = tf.random_uniform([1], 0, self.scaled_img_height - self.orig_img_height + 1, dtype=tf.int32, seed=2)[
            0]
        offset_x = tf.random_uniform([1], 0, self.scaled_img_width - self.orig_img_width + 1, dtype=tf.int32, seed=2)[0]
        im = tf.image.crop_to_bounding_box(im, offset_y, offset_x, self.orig_img_height, self.orig_img_width)
        return im


    # Scales and crops intrinsics, keeping them matched with their corresponding image stacks.
    # Make sure to scale and crop images first, as the randomly scaled image widths and heights are needed to
    # scale and crop intrinsics.
    def scale_and_crop_intrinsics(self, intrinsics):
        intrinsics = self.scale_intrinsics_randomly(intrinsics)
        intrinsics = self.crop_intrinsics_randomly(intrinsics)
        return intrinsics


    # Scale intrinsics randomly. Seed used to match corresponding image, seg mask, and intrinsics.
    def scale_intrinsics_randomly(self, intrinsics):
        scaling = tf.random_uniform([2], 1, 1.15, seed=2)
        x_scaling = scaling[0]
        y_scaling = scaling[1]

        fx = intrinsics[0, 0] * x_scaling
        fy = intrinsics[1, 1] * y_scaling
        cx = intrinsics[0, 2] * x_scaling
        cy = intrinsics[1, 2] * y_scaling
        return make_intrinsics_matrix(fx, fy, cx, cy)


    # Crop intrinsics randomly. It is assumed that the images has already been cropped, so that
    # the scaled image height and width are already known and saved in class state.
    def crop_intrinsics_randomly(self, intrinsics):
        """Crops image and adjust intrinsics accordingly."""
        offset_y = tf.random_uniform([1], 0, self.scaled_img_height - self.orig_img_height + 1, dtype=tf.int32, seed=2)[
            0]
        offset_x = tf.random_uniform([1], 0, self.scaled_img_width - self.orig_img_width + 1, dtype=tf.int32, seed=2)[0]
        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        cx = intrinsics[0, 2] - tf.cast(offset_x, dtype=tf.float32)
        cy = intrinsics[1, 2] - tf.cast(offset_y, dtype=tf.float32)
        return make_intrinsics_matrix(fx, fy, cx, cy)


# Creates a 3x3 intrinsics matrix from camera essentials.
# Returned as a tf.float32 tensor
def make_intrinsics_matrix(fx, fy, cx, cy):
    r1 = tf.stack([fx, 0, cx])
    r2 = tf.stack([0, fy, cy])
    r3 = tf.constant([0., 0., 1.])
    intrinsics = tf.stack([r1, r2, r3])
    return intrinsics


    # # Make image dataset from saved directory
    # data_root = '/mnt/isaac/apps/carter_sim_struct2depth/synth_images'
    # data_root_seg = '/mnt/isaac/apps/carter_sim_struct2depth/synth_images'
    # data_root_intrinsics = '/mnt/isaac/apps/carter_sim_struct2depth/synth_images_intrinsics'
    # all_image_paths = list(glob.glob(data_root + '/*'))
    # all_image_paths_seg = list(glob.glob(data_root_seg + '/*'))
    # all_image_paths_intrinsics = list(glob.glob(data_root_intrinsics + '/*'))
    #
    # # Raw image triplets
    # path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
    # image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
    #
    # # Seg masks
    # path_ds_seg = tf.data.Dataset.from_tensor_slices(all_image_paths_seg)
    # seg_ds = path_ds_seg.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
    #
    # # Camera intrinsics
    # record_defaults = [tf.float32] * 9
    # intrinsics_ds = tf.data.experimental.CsvDataset(all_image_paths_intrinsics, record_defaults)  # Dataset of .csv lines
    # intrinsics_ds = intrinsics_ds.map(lambda *x: tf.convert_to_tensor(x))  # Convert to tensors
    # logging.info("Datasets loaded")

    # with tf.name_scope('preprocessing'):





    #
    # def unpack_images(self, image_seq):
    #     """[h, w * seq_length, 3] -> [h, w, 3 * seq_length]."""
    #     with tf.name_scope('unpack_images'):
    #         image_list = [
    #             image_seq[:, i * self.img_width:(i + 1) * self.img_width, :]
    #             for i in range(self.seq_length)
    #         ]
    #         image_stack = tf.concat(image_list, axis=2)
    #         image_stack.set_shape(
    #             [self.img_height, self.img_width, self.seq_length * 3])
    #     return image_stack
    #
    # @classmethod
    # def preprocess_image(cls, image):
    #     # Convert from uint8 to float.
    #     return tf.image.convert_image_dtype(image, dtype=tf.float32)
    #
    # @classmethod
    # def augment_image_colorspace(cls, image_stack):
    #     """Apply data augmentation to inputs."""
    #     image_stack_aug = image_stack
    #     # Randomly shift brightness.
    #     apply_brightness = tf.less(tf.random_uniform(
    #         shape=[], minval=0.0, maxval=1.0, dtype=tf.float32), 0.5)
    #     image_stack_aug = tf.cond(
    #         apply_brightness,
    #         lambda: tf.image.random_brightness(image_stack_aug, max_delta=0.1),
    #         lambda: image_stack_aug)
    #
    #     # Randomly shift contrast.
    #     apply_contrast = tf.less(tf.random_uniform(
    #         shape=[], minval=0.0, maxval=1.0, dtype=tf.float32), 0.5)
    #     image_stack_aug = tf.cond(
    #         apply_contrast,
    #         lambda: tf.image.random_contrast(image_stack_aug, 0.85, 1.15),
    #         lambda: image_stack_aug)
    #
    #     # Randomly change saturation.
    #     apply_saturation = tf.less(tf.random_uniform(
    #         shape=[], minval=0.0, maxval=1.0, dtype=tf.float32), 0.5)
    #     image_stack_aug = tf.cond(
    #         apply_saturation,
    #         lambda: tf.image.random_saturation(image_stack_aug, 0.85, 1.15),
    #         lambda: image_stack_aug)
    #
    #     # Randomly change hue.
    #     apply_hue = tf.less(tf.random_uniform(
    #         shape=[], minval=0.0, maxval=1.0, dtype=tf.float32), 0.5)
    #     image_stack_aug = tf.cond(
    #         apply_hue,
    #         lambda: tf.image.random_hue(image_stack_aug, max_delta=0.1),
    #         lambda: image_stack_aug)
    #
    #     image_stack_aug = tf.clip_by_value(image_stack_aug, 0, 1)
    #     return image_stack_aug
    #
    # @classmethod
    # def augment_images_flip(cls, image_stack, seg_stack, intrinsics,
    #                         randomized=True):
    #     """Randomly flips the image horizontally."""
    #
    #     def flip(cls, image_stack, seg_stack, intrinsics):
    #         _, in_w, _ = image_stack.get_shape().as_list()
    #         fx = intrinsics[0, 0]
    #         fy = intrinsics[1, 1]
    #         cx = in_w - intrinsics[0, 2]
    #         cy = intrinsics[1, 2]
    #         intrinsics = cls.make_intrinsics_matrix(fx, fy, cx, cy)
    #         return (tf.image.flip_left_right(image_stack),
    #                 tf.image.flip_left_right(seg_stack), intrinsics)
    #
    #     if randomized:
    #         prob = tf.random_uniform(shape=[], minval=0.0, maxval=1.0,
    #                                  dtype=tf.float32)
    #         predicate = tf.less(prob, 0.5)
    #         return tf.cond(predicate,
    #                        lambda: flip(cls, image_stack, seg_stack, intrinsics),
    #                        lambda: (image_stack, seg_stack, intrinsics))
    #     else:
    #         return flip(cls, image_stack, seg_stack, intrinsics)
    #
    # @classmethod
    # def augment_images_scale_crop(cls, im, seg, intrinsics, out_h, out_w):
    #     """Randomly scales and crops image."""
    #
    #     def scale_randomly(im, seg, intrinsics):
    #         """Scales image and adjust intrinsics accordingly."""
    #         in_h, in_w, _ = im.get_shape().as_list()
    #         scaling = tf.random_uniform([2], 1, 1.15)
    #         x_scaling = scaling[0]
    #         y_scaling = scaling[1]
    #         out_h = tf.cast(in_h * y_scaling, dtype=tf.int32)
    #         out_w = tf.cast(in_w * x_scaling, dtype=tf.int32)
    #         # Add batch.
    #         im = tf.expand_dims(im, 0)
    #         im = tf.image.resize_area(im, [out_h, out_w])
    #         im = im[0]
    #
    #     with gfile.Open(os.path.join(data_dir, '%s.txt' % split), 'r') as f:
    #         frames = f.readlines()
    #         frames = [k.rstrip() for k in frames]
    #         seg = tf.expand_dims(seg, 0)
    #         seg = tf.image.resize_area(seg, [out_h, out_w])
    #         seg = seg[0]
    #         fx = intrinsics[0, 0] * x_scaling
    #         fy = intrinsics[1, 1] * y_scaling
    #         cx = intrinsics[0, 2] * x_scaling
    #         cy = intrinsics[1, 2] * y_scaling
    #         intrinsics = cls.make_intrinsics_matrix(fx, fy, cx, cy)
    #         return im, seg, intrinsics
    #
    #     # Random cropping
    #     def crop_randomly(im, seg, intrinsics, out_h, out_w):
    #         """Crops image and adjust intrinsics accordingly."""
    #         # batch_size, in_h, in_w, _ = im.get_shape().as_list()
    #         in_h, in_w, _ = tf.unstack(tf.shape(im))
    #         offset_y = tf.random_uniform([1], 0, in_h - out_h + 1, dtype=tf.int32)[0]
    #         offset_x = tf.random_uniform([1], 0, in_w - out_w + 1, dtype=tf.int32)[0]
    #         im = tf.image.crop_to_bounding_box(im, offset_y, offset_x, out_h, out_w)
    #         seg = tf.image.crop_to_bounding_box(seg, offset_y, offset_x, out_h, out_w)
    #         fx = intrinsics[0, 0]
    #         fy = intrinsics[1, 1]
    #         cx = intrinsics[0, 2] - tf.cast(offset_x, dtype=tf.float32)
    #         cy = intrinsics[1, 2] - tf.cast(offset_y, dtype=tf.float32)
    #         intrinsics = cls.make_intrinsics_matrix(fx, fy, cx, cy)
    #         return im, seg, intrinsics
    #
    #     im, seg, intrinsics = scale_randomly(im, seg, intrinsics)
    #     im, seg, intrinsics = crop_randomly(im, seg, intrinsics, out_h, out_w)
    #     return im, seg, intrinsics
    #
    #
    # @classmethod
    # def make_intrinsics_matrix(cls, fx, fy, cx, cy):
    #     r1 = tf.stack([fx, 0, cx])
    #     r2 = tf.stack([0, fy, cy])
    #     r3 = tf.constant([0., 0., 1.])
    #     intrinsics = tf.stack([r1, r2, r3])
    #     return intrinsics
    #
    # @classmethod
    # def get_multi_scale_intrinsics(cls, intrinsics, num_scales):
    #     """Returns multiple intrinsic matrices for different scales."""
    #     intrinsics_multi_scale = []
    #     # Scale the intrinsics accordingly for each scale
    #     for s in range(num_scales):
    #         fx = intrinsics[0, 0] / (2 ** s)
    #         fy = intrinsics[1, 1] / (2 ** s)
    #         cx = intrinsics[0, 2] / (2 ** s)
    #         cy = intrinsics[1, 2] / (2 ** s)
    #         intrinsics_multi_scale.append(cls.make_intrinsics_matrix(fx, fy, cx, cy))
    #     intrinsics_multi_scale = tf.stack(intrinsics_multi_scale)
    #     return intrinsics_multi_scale

def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, [128, 1248])
    return image
