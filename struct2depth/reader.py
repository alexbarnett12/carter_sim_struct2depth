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

# System imports
import os
import sys
from absl import logging
import tensorflow as tf
import numpy as np
import csv
import cv2
import time

# Struct2depth imports
import util
from isaac_app import create_sample_bridge
from process_image import ImageProcessor

# Automatically parallelize tf.mapping function to maximize efficiency
AUTOTUNE = tf.data.experimental.AUTOTUNE

# Imagenet mean and standard deviation
# See nets.encoder_resnet as reference for below input-normalizing constants.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_SD = (0.229, 0.224, 0.225)

# Op names.
COLOR_IMAGE = 'image'
SEG_MASK = 'segmentation_mask'
CAMERA_MAT = 'camera_matrix'
INTRINSICS = 'camera_matrix'
INTRINSICS_INV = 'camera_matrix_inverse'
IMAGE_NORM = 'imagenet_norm'

# Variables to set
FLIP_RANDOM = 'random'  # Always perform random flipping of input images.
FLIP_ALWAYS = 'always'  # Always flip image input, used for test augmentation.
FLIP_NONE = 'none'  # Always disables flipping.


class DataReader(object):
    """Reads stored sequences which are produced by dataset/gen_data.py."""

    def __init__(self,
                 data_dir,
                 batch_size,
                 img_height,
                 img_width,
                 seq_length,
                 num_scales,
                 file_extension,
                 random_scale_crop,
                 flipping_mode,
                 random_color,
                 imagenet_norm,
                 shuffle,
                 input_file='train',
                 isaac_app=None,
                 time_delay=0.4,
                 num_isaac_samples=1,
                 speed_threshold=0.25,
                 angular_speed_threshold=0.25,
                 optimize=False,
                 repetitions=0):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width
        self.seq_length = seq_length
        self.seq_width = self.img_width * self.seq_length
        self.num_scales = num_scales
        self.file_extension = file_extension
        self.random_scale_crop = random_scale_crop
        self.flipping_mode = flipping_mode
        self.random_color = random_color
        self.imagenet_norm = imagenet_norm
        self.shuffle = shuffle
        self.input_file = input_file
        self.isaac_app = isaac_app
        self.time_delay = time_delay
        self.sample_numbers = num_isaac_samples
        self.speed = 0
        self.speed_threshold = speed_threshold
        self.angular_speed = 0
        self.angular_speed_threshold = angular_speed_threshold
        self.steps_per_epoch = 1000
        self.optimize = optimize
        self.repetitions = repetitions

    # Retrieve current robot linear and angular speed from Isaac Sim
    def update_speed(self):
        with open('/mnt/isaac_2019_2/apps/carter_sim_struct2depth/differential_base_speed/speed.csv') as speed_file:
            csv_reader = csv.reader(speed_file, delimiter=',')
            for row in csv_reader:
                if len(row) == 2:
                    self.speed = float(row[0])
                    self.angular_speed = float(row[1])

    # Check if Isaac Sim bridge has samples
    def has_samples(self, bridge):
        return bridge.get_sample_count() >= self.sample_numbers

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

                images = []

                self.update_speed()

                # Only collect samples if robot is moving faster than a certain threshold
                if self.speed > self.speed_threshold or self.angular_speed > self.angular_speed_threshold:

                    # Retrieve a total of (batch_size * seq_length) images
                    for i in range(self.batch_size * self.seq_length):

                        # Wait until we get enough samples from Isaac
                        while not self.has_samples(bridge):
                            time.sleep(self.time_delay)

                        # Acquire image
                        new_image = bridge.acquire_samples(self.sample_numbers)

                        # Add image to list
                        images.append(np.squeeze(new_image))

                        # Wait to increase disparity between images
                        time.sleep(self.time_delay)

                        # TODO: Turn seg mask generator into an Isaac node
                        # TODO: Fix MRCNN to work within another tf.Graph()
                        # Create wide image and segmentation triplets
                        if (i + 1) % self.seq_length == 0:
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
                COLOR_IMAGE: (dataset_size, self.img_height, self.seq_width, 3),
                SEG_MASK: (dataset_size, self.img_height, self.seq_width, 3),
                INTRINSICS: (dataset_size, 3, 3),
            })

        return dataset

    def read_data(self):
        """Provides images and camera intrinsics."""
        with tf.name_scope('data_loading'):

            # Startup the sample accumulator bridge to get Isaac Sim data
            bridge = create_sample_bridge(self.isaac_app)

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

            logging.info("Image dataset dimensions: {}".format(image_ds))
            logging.info("Seg mask dataset dimensions: {}".format(seg_ds))
            logging.info("Intrinsics dataset dimensions: {}".format(intrinsics_ds))

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
                    flipper = Flipper(image_width=self.img_width, batch_size=self.batch_size,
                                      randomized=random_flipping)

                    # Flip images, seg masks, and intrinsics randomly or completely, depending on random_flipping
                    image_stack_ds = image_stack_ds.map(flipper.flip_images, num_parallel_calls=AUTOTUNE)
                    seg_stack_ds = seg_stack_ds.map(flipper.flip_images, num_parallel_calls=AUTOTUNE)
                    intrinsics_ds = intrinsics_ds.map(flipper.flip_intrinsics, num_parallel_calls=AUTOTUNE)

                    logging.info("Images flipped and intrinsics adjusted")

            # TODO: Make this functional
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

        # Create iterators over datasets
        image_it = image_stack_ds.make_one_shot_iterator().get_next()
        image_norm_it = image_stack_norm.make_one_shot_iterator().get_next()
        seg_it = seg_stack_ds.make_one_shot_iterator().get_next()
        intrinsics_it = intrinsics_ds.make_one_shot_iterator().get_next()
        intrinsics_inv_it = intrinsics_inv.make_one_shot_iterator().get_next()

        logging.info("Dataset successfuly processed")
        logging.info("Final image dimensions: {}".format(image_it))
        logging.info("Final image norm dimensions: {}".format(image_norm_it))
        logging.info("Final seg mask dimensions: {}".format(seg_it))
        logging.info("Final intrinsics dimensions: {}".format(intrinsics_it))
        logging.info("Final intrinsics inverse dimensions: {}".format(intrinsics_inv_it))

        return (image_it,
                image_norm_it,
                seg_it,
                intrinsics_it,
                intrinsics_inv_it)

    # Unpack image triplet from [h, w * seq_length, 3] -> [h, w, 3 * seq_length] image stack.
    def unpack_images(self, image_seq):
        with tf.name_scope('unpack_images'):
            image_list = [
                image_seq[:, :, i * self.img_width:(i + 1) * self.img_width, :]
                for i in range(self.seq_length)
            ]
            image_stack = tf.concat(image_list, axis=3)
            image_stack.set_shape([self.batch_size if not self.optimize else self.repetitions,
                                   self.img_height, self.img_width, self.seq_length * 3])
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
        intrinsics_ds_multi_scale = []
        intrinsics_multi_scale = []
        # Scale the intrinsics accordingly for each scale
        for x in range(self.batch_size if not self.optimize else self.repetitions):
            intrinsics_multi_scale = []
            for s in range(self.num_scales):
                fx = intrinsics[x, 0, 0] / (2 ** s)
                fy = intrinsics[x, 1, 1] / (2 ** s)
                cx = intrinsics[x, 0, 2] / (2 ** s)
                cy = intrinsics[x, 1, 2] / (2 ** s)
                intrinsics_multi_scale.append(make_intrinsics_matrix(fx, fy, cx, cy))
            intrinsics_ds_multi_scale.append(intrinsics_multi_scale)
        intrinsics_ds_multi_scale = tf.stack(intrinsics_ds_multi_scale)
        return intrinsics_ds_multi_scale

    # Normalize the image by the Imagenet mean and standard deviation.
    # This aligns the training dataset with the pre-trained Imagenet model, and allows
    # for standardized evaluation of test set.
    def normalize_by_imagenet(self, image_stack):
        # Copy constant values multiple times to fill up a tensor of length SEQ_LENGTH * len(IMAGENET_MEAN)
        im_mean = tf.tile(
            tf.constant(IMAGENET_MEAN), multiples=[self.seq_length])
        im_sd = tf.tile(
            tf.constant(IMAGENET_SD), multiples=[self.seq_length])
        return (image_stack - im_mean) / im_sd


# Class for flipping images, seg masks, and intrinsics
# Provides greater variety in training dataset to avoid overfitting
class Flipper:
    def __init__(self,
                 image_width=416,
                 batch_size=1,
                 randomized=True):
        self.randomized = randomized
        self.batch_size = batch_size
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

        if self.randomized:
            # Generate random probability. Seed provided to ensure that image,
            # seg mask, and intrinsics are paired
            prob = tf.random_uniform(shape=[], minval=0.0, maxval=1.0,
                                     dtype=tf.float32, seed=2)
            predicate = tf.less(prob, 0.5)
        else:
            predicate = tf.less(0.0, 0.5)

        intrinsics_flipped = tf.map_fn(lambda x: tf.cond(predicate,
                                                         lambda: make_intrinsics_matrix(x[0, 0], x[1, 1],
                                                                                        self.img_width - x[0, 2],
                                                                                        x[1, 2]),
                                                         lambda: x), intrinsics)

        return intrinsics_flipped


# Class for cropping images. First scales them to provide a greater area
# to crop from.
class Cropper:
    def __init__(self,
                 image_width=416,
                 image_height=128):
        self.orig_img_width = image_width
        self.orig_img_height = image_height
        self.scaled_img_width = image_width  # New image width before cropping to original dimensions
        self.scaled_img_height = image_height  # New image height before cropping to original dimensions

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


# Helper function for splitting tf datasets into sub-datasets.
def split_datasets(dataset):
    subsets = {}
    names = list(dataset.output_shapes.keys())
    for name in names:
        subsets[name] = dataset.map(lambda x: x[name])

    return subsets
