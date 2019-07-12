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

from struct2depth import util
from struct2depth.process_image import ImageProcessor
import time

gfile = tf.gfile

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
ROOT_DIR = os.path.abspath("/mnt/isaac")
sys.path.append(ROOT_DIR)

from engine.pyalice import *
import packages.ml
from pinhole_to_tensor import PinholeToTensor

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

# Number of samples to acquire in batch
kSampleNumbers = 1

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('graph_filename', "apps/carter_sim_struct2depth/carter.graph.json",
                    'Where the isaac SDK app graph is stored')
flags.DEFINE_string('config_filename', "apps/carter_sim_struct2depth/carter.config.json",
                    'Where the isaac SDK app node configuration is stored')


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
        self.steps_per_epoch = 500


    def get_generator(self, bridge, img_processor):
        """Create a training sample generator.

        Args:
          bridge: the isaac sample accumulator node which we will acquire samples from

        Returns:
          A generator function which yields a single training example.
      """

        def _generator():
            # Indefinitely yield samples.
            while True:

                # Wait until we get enough samples from Isaac
                while True:
                    num = bridge.get_sample_number()
                    if num >= kSampleNumbers:
                        break
                    time.sleep(1.0)
                    # logging.info("waiting for enough samples: {}".format(num))

                # Try to acquire a sample.
                # If none are available, then wait for a bit so we do not spam the app.
                sample = bridge.acquire_samples(kSampleNumbers)
                print("SAMPLES ACQUIRED")
                if not sample:
                    time.sleep(1)
                    continue
                print(sample)
                # print(np.shape(sample))

                # Create wide image and segmentation triplets
                # TODO: Turn seg mask generator into an Isaac node
                # TODO: Fix MRCNN to work within another tf.Graph()
                image_seq, seg_mask_seq = img_processor.process_image(np.array([sample[0][0],
                                                                              sample[1][0],
                                                                              sample[2][0]]))

                # TODO: Retrieve camera mat from Isaac
                intrinsics = np.array([[480, 0, 270], [0, 480, 480], [0, 0, 1]], dtype=np.float32)

                # print(np.shape(image_seq))
                # print(np.shape(seg_mask_seq))
                # print(np.shape(intrinsics))

                # with tf.name_scope('preprocessing'):
                #
                #     # Convert from uint8 to float32 (if necessary)
                #     # image_seq = self.preprocess_image(image_seq)
                #     # seg_mask_seq = self.preprocess_image(seg_mask_seq)
                #     # intrinsics = tf.dtypes.cast(intrinsics, dtype=tf.float32)
                #     # print("Converted to float32")
                #
                #     # Randomly augment colorspace
                #     if self.random_color:
                #         with tf.name_scope('image_augmentation'):
                #             image = self.augment_image_colorspace(image_seq)
                #         logging.info("Image successfully augmented")
                #
                #     print(tf.shape(image_seq))
                #     print(tf.shape(seg_mask_seq))
                #     print(tf.shape(intrinsics))
                #
                #     # Unpack wide images into three stacked images
                #     image_stack = self.unpack_images(image_seq)
                #     seg_stack = self.unpack_images(seg_mask_seq)
                #     logging.info("Images unpacked")
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

                # Return nparray of image
                # yield {COLOR_IMAGE: image_stack,
                #        IMAGE_NORM: image_stack_norm,
                #        SEG_MASK: seg_stack,
                #        INTRINSICS: intrinsics,
                #        INTRINSICS_INV: intrinsics_inv}
                yield {COLOR_IMAGE: image_seq,
                       SEG_MASK: seg_mask_seq,
                       INTRINSICS: intrinsics}

        return _generator

    def get_dataset(self, bridge, batch_size, img_processor):
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
                COLOR_IMAGE: (None, None, 3),
                SEG_MASK: (None, None, 3),
                INTRINSICS: (3, 3),
            })

        return dataset

    def read_data(self):
        """Provides images and camera intrinsics."""
        with tf.name_scope('data_loading'):

            # Startup the sample accumulator bridge to get data
            sample_accumulator = self.isaac_app.find_node_by_name("CarterTrainingSamples")
            pinhole_to_tensor = self.isaac_app.find_node_by_name("pinhole_to_tensor")
            bridge = packages.ml.SampleAccumulator(sample_accumulator)
            pinhole_to_tensor = PinholeToTensor(pinhole_to_tensor, 0, 0)

            # Start the application and Sight server
            self.isaac_app.start_webserver()
            self.isaac_app.start()
            logging.info("Isaac application loaded")

            # Create image processor for generating triplets and seg masks
            img_processor = ImageProcessor()
            logging.info("Image Processor created")

            # Create a Dataset and iterator from Isaac generator
            dataset = self.get_dataset(bridge, self.batch_size, img_processor)
            data_dict = dataset.make_one_shot_iterator().get_next()
            logging.info("Dataset from generator created")
            print("Dataset: {}".format(dataset))

            # Extract image, seg mask, and camera matrix
            image_seq = data_dict[COLOR_IMAGE]
            # image_stack_norm = data_dict[IMAGE_NORM]
            seg_mask_seq = data_dict[SEG_MASK]
            intrinsics = data_dict[INTRINSICS]
            # intrinsics_inv = data_dict[INTRINSICS_INV]
            logging.info("Dictionary elements extracted")

            print("Dataset: {}".format(dataset))
            print("Image dataset: {}".format(image_seq))
            print("Seg mask dataset: {}".format(seg_mask_seq))
            print("Intrinsics dataset: {}".format(intrinsics))

            # Resize images to network input size
            image_seq = tf.image.resize_images(data_dict[COLOR_IMAGE], (HEIGHT, TRIPLET_WIDTH))
            seg_mask_seq = tf.dtypes.cast(tf.image.resize_images(data_dict[SEG_MASK], (HEIGHT, TRIPLET_WIDTH)), np.uint8)
            print("Resized image dataset: {}".format(image_seq))
            print("Resized seg mask dataset: {}".format(seg_mask_seq))
            print("Dataset after resize: {}".format(dataset))

        with tf.name_scope('preprocessing'):

            # Randomly augment colorspace
            if self.random_color:
                with tf.name_scope('image_augmentation'):
                    image = self.augment_image_colorspace(image_seq)
                logging.info("Image successfully augmented")
            print("Image dataset: {}".format(image_seq))
            print("Seg mask dataset: {}".format(seg_mask_seq))
            print("Intrinsics dataset: {}".format(intrinsics))

            # Unpack wide images into three stacked images
            image_stack = self.unpack_images(image_seq)
            seg_stack = self.unpack_images(seg_mask_seq)
            logging.info("Images unpacked")
            print("Image dataset: {}".format(image_stack))
            print("Seg mask dataset: {}".format(seg_stack))
            print("Intrinsics dataset: {}".format(intrinsics))

            # Randomly flip images
            if self.flipping_mode != FLIP_NONE:
                random_flipping = (self.flipping_mode == FLIP_RANDOM)
                with tf.name_scope('image_augmentation_flip'):
                    image_stack, seg_stack, intrinsics = self.augment_images_flip(
                        image_stack, seg_stack, intrinsics,
                        randomized=random_flipping)
                    logging.info("Images flipped")

            # Randomly scale and crop images
            if self.random_scale_crop:
                with tf.name_scope('image_augmentation_scale_crop'):
                    image_stack, seg_stack, intrinsics = self.augment_images_scale_crop(
                        image_stack, seg_stack, intrinsics, self.img_height,
                        self.img_width)
                logging.info("Images scaled and cropped")

            # Adjust camera intrinsics to the correct scale and compute the inverse
            with tf.name_scope('multi_scale_intrinsics'):
                intrinsics = self.get_multi_scale_intrinsics(intrinsics,
                                                             self.num_scales)
                intrinsics.set_shape([self.num_scales, 3, 3])
                intrinsics_inv = tf.matrix_inverse(intrinsics)
                intrinsics_inv.set_shape([self.num_scales, 3, 3])
                logging.info("Multi scale intrinsics received")

            # Subtract Imagenet norm
            if self.imagenet_norm:
                im_mean = tf.tile(
                    tf.constant(IMAGENET_MEAN), multiples=[self.seq_length])
                im_sd = tf.tile(
                    tf.constant(IMAGENET_SD), multiples=[self.seq_length])
                image_stack_norm = (image_stack - im_mean) / im_sd
                logging.info("Imagenet norm used")
            else:
                image_stack_norm = image_stack
                logging.info("Imagenet norm not used")

            # Wait until we get enough samples from Isaac
            while True:
                num = bridge.get_sample_number()
                if num >= kSampleNumbers:
                    break
                time.sleep(1.0)
                logging.info("waiting for enough samples: {}".format(num))

        with tf.name_scope('batching'):
            if self.shuffle:
                (image_stack, image_stack_norm, seg_stack, intrinsics,
                 intrinsics_inv) = tf.train.shuffle_batch(
                    [image_stack, image_stack_norm, seg_stack, intrinsics,
                     intrinsics_inv],
                    batch_size=self.batch_size,
                    capacity=QUEUE_SIZE + QUEUE_BUFFER * self.batch_size,
                    min_after_dequeue=QUEUE_SIZE)
            else:
                (image_stack, image_stack_norm, seg_stack, intrinsics,
                 intrinsics_inv) = tf.train.batch(
                    [image_stack, image_stack_norm, seg_stack, intrinsics,
                     intrinsics_inv],
                    batch_size=self.batch_size,
                    capacity=QUEUE_SIZE + QUEUE_BUFFER * self.batch_size,
                    min_after_dequeue=QUEUE_SIZE)

            logging.info("Dataset successfuly processed")
        logging.info("Final dimensional check")
        print(dataset)
        print("Image dataset: {}".format(image_stack))
        print("Image norm: {}".format(image_stack_norm))
        print("Seg mask dataset: {}".format(seg_stack))
        print("Intrinsics dataset: {}".format(intrinsics))
        print("Intrinsics inverse dataset: {}".format(intrinsics_inv))
        return (image_stack, image_stack_norm, seg_stack, intrinsics,
                intrinsics_inv)


    def unpack_images(self, image_seq):
        """[h, w * seq_length, 3] -> [h, w, 3 * seq_length]."""
        with tf.name_scope('unpack_images'):
            image_list = [
                image_seq[:, i * self.img_width:(i + 1) * self.img_width, :]
                for i in range(self.seq_length)
            ]
            image_stack = tf.concat(image_list, axis=2)
            image_stack.set_shape(
                [self.img_height, self.img_width, self.seq_length * 3])
        return image_stack

    @classmethod
    def preprocess_image(cls, image):
        # Convert from uint8 to float.
        return tf.image.convert_image_dtype(image, dtype=tf.float32)

    @classmethod
    def augment_image_colorspace(cls, image_stack):
        """Apply data augmentation to inputs."""
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

        image_stack_aug = tf.clip_by_value(image_stack_aug, 0, 1)
        return image_stack_aug

    @classmethod
    def augment_images_flip(cls, image_stack, seg_stack, intrinsics,
                            randomized=True):
        """Randomly flips the image horizontally."""

        def flip(cls, image_stack, seg_stack, intrinsics):
            _, in_w, _ = image_stack.get_shape().as_list()
            fx = intrinsics[0, 0]
            fy = intrinsics[1, 1]
            cx = in_w - intrinsics[0, 2]
            cy = intrinsics[1, 2]
            intrinsics = cls.make_intrinsics_matrix(fx, fy, cx, cy)
            return (tf.image.flip_left_right(image_stack),
                    tf.image.flip_left_right(seg_stack), intrinsics)

        if randomized:
            prob = tf.random_uniform(shape=[], minval=0.0, maxval=1.0,
                                     dtype=tf.float32)
            predicate = tf.less(prob, 0.5)
            return tf.cond(predicate,
                           lambda: flip(cls, image_stack, seg_stack, intrinsics),
                           lambda: (image_stack, seg_stack, intrinsics))
        else:
            return flip(cls, image_stack, seg_stack, intrinsics)

    @classmethod
    def augment_images_scale_crop(cls, im, seg, intrinsics, out_h, out_w):
        """Randomly scales and crops image."""

        def scale_randomly(im, seg, intrinsics):
            """Scales image and adjust intrinsics accordingly."""
            in_h, in_w, _ = im.get_shape().as_list()
            scaling = tf.random_uniform([2], 1, 1.15)
            x_scaling = scaling[0]
            y_scaling = scaling[1]
            out_h = tf.cast(in_h * y_scaling, dtype=tf.int32)
            out_w = tf.cast(in_w * x_scaling, dtype=tf.int32)
            # Add batch.
            im = tf.expand_dims(im, 0)
            im = tf.image.resize_area(im, [out_h, out_w])
            im = im[0]

        with gfile.Open(os.path.join(data_dir, '%s.txt' % split), 'r') as f:
            frames = f.readlines()
            frames = [k.rstrip() for k in frames]
            seg = tf.expand_dims(seg, 0)
            seg = tf.image.resize_area(seg, [out_h, out_w])
            seg = seg[0]
            fx = intrinsics[0, 0] * x_scaling
            fy = intrinsics[1, 1] * y_scaling
            cx = intrinsics[0, 2] * x_scaling
            cy = intrinsics[1, 2] * y_scaling
            intrinsics = cls.make_intrinsics_matrix(fx, fy, cx, cy)
            return im, seg, intrinsics

        # Random cropping
        def crop_randomly(im, seg, intrinsics, out_h, out_w):
            """Crops image and adjust intrinsics accordingly."""
            # batch_size, in_h, in_w, _ = im.get_shape().as_list()
            in_h, in_w, _ = tf.unstack(tf.shape(im))
            offset_y = tf.random_uniform([1], 0, in_h - out_h + 1, dtype=tf.int32)[0]
            offset_x = tf.random_uniform([1], 0, in_w - out_w + 1, dtype=tf.int32)[0]
            im = tf.image.crop_to_bounding_box(im, offset_y, offset_x, out_h, out_w)
            seg = tf.image.crop_to_bounding_box(seg, offset_y, offset_x, out_h, out_w)
            fx = intrinsics[0, 0]
            fy = intrinsics[1, 1]
            cx = intrinsics[0, 2] - tf.cast(offset_x, dtype=tf.float32)
            cy = intrinsics[1, 2] - tf.cast(offset_y, dtype=tf.float32)
            intrinsics = cls.make_intrinsics_matrix(fx, fy, cx, cy)
            return im, seg, intrinsics

        im, seg, intrinsics = scale_randomly(im, seg, intrinsics)
        im, seg, intrinsics = crop_randomly(im, seg, intrinsics, out_h, out_w)
        return im, seg, intrinsics

    def compile_file_list(self, data_dir, split, load_pose=False):
        """Creates a list of input files."""
        logging.info('data_dir: %s', data_dir)
        with gfile.Open(os.path.join(data_dir, '%s.txt' % split), 'r') as f:
            frames = f.readlines()
            frames = [k.rstrip() for k in frames]
        subfolders = [x.split(' ')[0] for x in frames]
        frame_ids = [x.split(' ')[1] for x in frames]
        image_file_list = [
            os.path.join(data_dir, subfolders[i], frame_ids[i] + '.' +
                         self.file_extension)
            for i in range(len(frames))
        ]
        segment_file_list = [
            os.path.join(data_dir, subfolders[i], frame_ids[i] + '-fseg.' +
                         self.file_extension)
            for i in range(len(frames))
        ]
        cam_file_list = [
            os.path.join(data_dir, subfolders[i], frame_ids[i] + '_cam.txt')
            for i in range(len(frames))
        ]
        file_lists = {}
        file_lists['image_file_list'] = image_file_list
        file_lists['segment_file_list'] = segment_file_list
        file_lists['cam_file_list'] = cam_file_list
        if load_pose:
            pose_file_list = [
                os.path.join(data_dir, subfolders[i], frame_ids[i] + '_pose.txt')
                for i in range(len(frames))
            ]
            file_lists['pose_file_list'] = pose_file_list
        self.steps_per_epoch = len(image_file_list) // self.batch_size
        return file_lists

    @classmethod
    def make_intrinsics_matrix(cls, fx, fy, cx, cy):
        r1 = tf.stack([fx, 0, cx])
        r2 = tf.stack([0, fy, cy])
        r3 = tf.constant([0., 0., 1.])
        intrinsics = tf.stack([r1, r2, r3])
        return intrinsics

    @classmethod
    def get_multi_scale_intrinsics(cls, intrinsics, num_scales):
        """Returns multiple intrinsic matrices for different scales."""
        intrinsics_multi_scale = []
        # Scale the intrinsics accordingly for each scale
        for s in range(num_scales):
            fx = intrinsics[0, 0] / (2 ** s)
            fy = intrinsics[1, 1] / (2 ** s)
            cx = intrinsics[0, 2] / (2 ** s)
            cy = intrinsics[1, 2] / (2 ** s)
            intrinsics_multi_scale.append(cls.make_intrinsics_matrix(fx, fy, cx, cy))
        intrinsics_multi_scale = tf.stack(intrinsics_multi_scale)
        return intrinsics_multi_scale
