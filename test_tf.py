# import tensorflow as tf
# tf.enable_eager_execution()
# print("Eager execution: {}".format(tf.executing_eagerly()))
# import matplotlib.pyplot as plt
# import glob
# import random
#
# def preprocess_image(image):
#     image = tf.image.decode_image(image, channels=3)
#     # image = tf.math.divide(image, 255.0)
#     image = tf.image.resize(image, [192, 192])
#     image /= 255.0  # normalize to [0,1] range
#
#
#     return image
#
# def load_and_preprocess_image(path):
#     image = tf.read_file(path)
#     return preprocess_image(image)
#
#
# AUTOTUNE = tf.data.experimental.AUTOTUNE
# data_root = '/mnt/isaac/apps/carter_sim_struct2depth/synth_images'
# all_image_paths = list(glob.glob(data_root + '/*'))
# all_image_paths = [str(path) for path in all_image_paths]
# random.shuffle(all_image_paths)
# print(all_image_paths)
#
# img_path = all_image_paths[0]
# img_raw = tf.read_file(img_path)
# img_tensor = tf.image.decode_png(img_raw)
# print(img_tensor.shape)
# print(img_tensor.dtype)
#
# path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
# image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
#
# plt.figure(figsize=(8,8))
# for n,image in enumerate(image_ds.take(4)):
#   plt.subplot(2,2,n+1)
#   plt.imshow(image)
#   plt.grid(False)
#   plt.xticks([])
#   plt.yticks([])
#   plt.show()
#
# features = image_ds.make_one_shot_iterator().get_next()
# print(image_ds)

from __future__ import absolute_import, division, print_function, unicode_literals
import glob
import os
import sys
from absl import logging
import matplotlib.pyplot as plt
import tensorflow as tf
# tf.enable_eager_execution()
import numpy as np

logging._warn_preinit_stderr = 0
gfile = tf.gfile

# tf.enable_eager_execution()
print(tf.__version__)

# Op names.
COLOR_IMAGE = 'image'
SEG_MASK = 'segmentation_mask'
CAMERA_MAT = 'camera_matrix'
INTRINSICS = 'camera_matrix'
INTRINSICS_INV = 'camera_matrix_inverse'
IMAGE_NORM = 'imagenet_norm'

# Dimensions
SEQ_LENGTH = 3
HEIGHT = 128
WIDTH = 416
TRIPLET_WIDTH = WIDTH * SEQ_LENGTH
BATCH_SIZE = 20
NUM_SCALES = 4
BUFFER_SIZE = 50
REPEAT = 200

# Imagenet normalization
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_SD = (0.229, 0.224, 0.225)

# Data augmentation options
RANDOM_COLOR = True
FLIP = True
FLIP_RANDOM = True
RANDOM_SCALE_CROP = False
IMAGENET_NORM = True
SHUFFLE = True


def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, [128, 1248])
    return image


def display_images(image_ds):
    plt.figure(figsize=(8, 8))
    for n, image in enumerate(image_ds.take(4)):
        plt.subplot(2, 2, n + 1)
        plt.imshow(image[:, :, 3])
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        plt.show()

# Unpack image triplet from [h, w * seq_length, 3] -> [h, w, 3 * seq_length] image stack.
def unpack_images(image_seq):
    with tf.name_scope('unpack_images'):
        image_list = [
            image_seq[:, i * WIDTH:(i + 1) * WIDTH, :]
            for i in range(SEQ_LENGTH)
        ]
        image_stack = tf.concat(image_list, axis=2)
        image_stack.set_shape([HEIGHT, WIDTH, SEQ_LENGTH * 3])
    return image_stack

# Randomly augment the brightness contrast, saturation, and hue of the image.
# Provides more variety in training set to avoid overfitting.
def augment_image_colorspace(image_stack):
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


# Scale intrinsics randomly. Uses seed to be paired with images
# def scale_intrinsics_randomly(intrinsics):
#     """Scales image and adjust intrinsics accordingly."""
#     scaling = tf.random_uniform([2], 1, 1.15, seed=2)
#     x_scaling = scaling[0]
#     y_scaling = scaling[1]
#
#     fx = intrinsics[0, 0] * x_scaling
#     fy = intrinsics[1, 1] * y_scaling
#     cx = intrinsics[0, 2] * x_scaling
#     cy = intrinsics[1, 2] * y_scaling
#     return make_intrinsics_matrix(fx, fy, cx, cy), x_scaling, y_scaling
#
# # Crop intrinsics randomly. It is assumed that the images has already been cropped, so that
# # the scaled image height and width are already known.
# def crop_intrinsics_randomly(intrinsics, x_scaling, y_scaling):
#     """Crops image and adjust intrinsics accordingly."""
#     orig_img_height = 128
#     orig_img_width = 416
#     scaled_img_height =tf.cast(orig_img_height * y_scaling, dtype=tf.int32)
#     scaled_img_width = tf.cast(orig_img_width * x_scaling, dtype=tf.int32)
#     offset_y = tf.random_uniform([1], 0, scaled_img_height - orig_img_height + 1, dtype=tf.int32)[0]
#     offset_x = tf.random_uniform([1], 0, scaled_img_width - orig_img_width + 1, dtype=tf.int32)[0]
#     fx = intrinsics[0, 0]
#     fy = intrinsics[1, 1]
#     cx = intrinsics[0, 2] - tf.cast(offset_x, dtype=tf.float32)
#     cy = intrinsics[1, 2] - tf.cast(offset_y, dtype=tf.float32)
#     return make_intrinsics_matrix(fx, fy, cx, cy)

# Scales and crops intrinsics, keeping them matched with their corresponding image stacks.
# Make sure to scale and crop images first, as the randomly scaled image widths and heights are needed to
# scale and crop intrinsics.
# def scale_and_crop_intrinsics(intrinsics):
#     intrinsics, x_scaling, y_scaling = scale_intrinsics_randomly(intrinsics)
#     intrinsics = crop_intrinsics_randomly(intrinsics, x_scaling, y_scaling)
#     return intrinsics

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

# Creates multi scale intrinsics based off number of scales provided.
def get_multi_scale_intrinsics(intrinsics):
    intrinsics_multi_scale = []
    # Scale the intrinsics accordingly for each scale
    for s in range(NUM_SCALES):
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
def normalize_by_imagenet(image_stack):
    # Copy constant values multiple times to fill up a tensor of length SEQ_LENGTH * len(IMAGENET_MEAN)
    im_mean = tf.tile(
        tf.constant(IMAGENET_MEAN), multiples=[SEQ_LENGTH])
    im_sd = tf.tile(
        tf.constant(IMAGENET_SD), multiples=[SEQ_LENGTH])
    return (image_stack - im_mean) / im_sd


# Make image dataset from saved directory
AUTOTUNE = tf.data.experimental.AUTOTUNE
data_root = '/mnt/isaac/apps/carter_sim_struct2depth/synth_images'
data_root_seg = '/mnt/isaac/apps/carter_sim_struct2depth/synth_images_seg'
data_root_intrinsics = '/mnt/isaac/apps/carter_sim_struct2depth/synth_images_intrinsics'
all_image_paths = list(glob.glob(data_root + '/*'))
all_image_paths_seg = list(glob.glob(data_root_seg + '/*'))
all_image_paths_intrinsics = list(glob.glob(data_root_intrinsics + '/*'))

# Raw image triplets
path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)

# Seg masks
path_ds_seg = tf.data.Dataset.from_tensor_slices(all_image_paths_seg)
seg_ds = path_ds_seg.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)

# Camera intrinsics
record_defaults = [tf.float32] * 9
intrinsics_ds = tf.data.experimental.CsvDataset(all_image_paths_intrinsics, record_defaults)  # Dataset of .csv lines
intrinsics_ds = intrinsics_ds.map(lambda *x: tf.convert_to_tensor(x))  # Convert to tensors
logging.info("Datasets loaded")

# DEBUGGING
# print("Image input shape: {}".format(tf.data.get_output_shapes(image_ds)))
# print("Seg mask input shape: {}".format(tf.data.get_output_shapes(seg_ds)))
# print("Intrinsics input shape: {}".format(tf.data.get_output_shapes(intrinsics_ds)))

with tf.name_scope('preprocessing'):
    # Scale image values from 0-255 to 0-1
    image_ds = image_ds.map(lambda x: x / 255.0, num_parallel_calls=AUTOTUNE)
    seg_ds = seg_ds.map(lambda x: x / 255.0, num_parallel_calls=AUTOTUNE)  # NOTE: Only needed for testing

    # Convert cam intrinsics to 3x3 matrix
    # NOTE: only needed for testing
    intrinsics_ds = intrinsics_ds.map(lambda x: tf.reshape(x, [3, 3]))
    # print("Reshaped intrinsics shape: {}".format(tf.data.get_output_shapes(intrinsics_ds)))

    # Randomly augment colorspace
    if RANDOM_COLOR:
        with tf.name_scope('image_augmentation'):
            image_ds = image_ds.map(augment_image_colorspace, num_parallel_calls=AUTOTUNE)
        logging.info("Image successfully augmented")

    # Unpack triplets; each tensor is unpacked into a stack of three images
    image_stack_ds = image_ds.map(unpack_images, num_parallel_calls=AUTOTUNE)
    seg_stack_ds = seg_ds.map(unpack_images, num_parallel_calls=AUTOTUNE)
    logging.info("Images unpacked")

    # Randomly flip images
    if FLIP:
        random_flipping = (FLIP_RANDOM == True)
        with tf.name_scope('image_augmentation_flip'):
            # Create image flipper
            flipper = Flipper(image_width=WIDTH, randomized=random_flipping)

            # Flip images, seg masks, and intrinsics randomly or completely, depending on random_flipping
            image_stack_ds = image_stack_ds.map(flipper.flip_images, num_parallel_calls=AUTOTUNE)
            seg_stack_ds = seg_stack_ds.map(flipper.flip_images, num_parallel_calls=AUTOTUNE)
            intrinsics_ds = intrinsics_ds.map(flipper.flip_intrinsics, num_parallel_calls=AUTOTUNE)

            logging.info("Images flipped and intrinsics adjusted")

    # Randomly scale and crop images
    if RANDOM_SCALE_CROP:
        with tf.name_scope('image_augmentation_scale_crop'):
            # Create image cropper
            cropper = Cropper(image_width=WIDTH, image_height=HEIGHT)

            # Crop images, seg masks, and intrinsics
            image_stack_ds = image_stack_ds.map(cropper.scale_and_crop_image, num_parallel_calls=AUTOTUNE)
            seg_stack_ds = seg_stack_ds.map(cropper.scale_and_crop_image, num_parallel_calls=AUTOTUNE)
            intrinsics_ds = intrinsics_ds.map(cropper.scale_and_crop_intrinsics, num_parallel_calls=AUTOTUNE)

            logging.info("Images scaled and cropped")

    # Adjust camera intrinsics to the correct scale and compute the inverse
    with tf.name_scope('multi_scale_intrinsics'):
        intrinsics_ds = intrinsics_ds.map(get_multi_scale_intrinsics, num_parallel_calls=AUTOTUNE)
        intrinsics_inv = intrinsics_ds.map(lambda x: tf.matrix_inverse(x), num_parallel_calls=AUTOTUNE)

        logging.info("Multi scale intrinsics received")

    # Normalize images by the Imagenet standard
    if IMAGENET_NORM:
        image_stack_norm = image_stack_ds.map(normalize_by_imagenet, num_parallel_calls=AUTOTUNE)
        logging.info("Imagenet norm used")
    else:
        image_stack_norm = image_stack_ds
        logging.info("Imagenet norm not used")

with tf.name_scope('batching'):

        if SHUFFLE:
            image_stack_ds = image_stack_ds.shuffle(buffer_size=BUFFER_SIZE, seed=2).batch(BATCH_SIZE).repeat(REPEAT)
            image_stack_norm = image_stack_norm.shuffle(buffer_size=BUFFER_SIZE, seed=2).batch(BATCH_SIZE).repeat(REPEAT)
            seg_stack_ds = seg_stack_ds.shuffle(buffer_size=BUFFER_SIZE, seed=2).batch(BATCH_SIZE).repeat(REPEAT)
            intrinsics_ds = intrinsics_ds.shuffle(buffer_size=BUFFER_SIZE, seed=2).batch(BATCH_SIZE).repeat(REPEAT)
            intrinsics_inv = intrinsics_inv.shuffle(buffer_size=BUFFER_SIZE, seed=2).batch(BATCH_SIZE).repeat(REPEAT)

        else:
            image_stack_ds = image_stack_ds.batch(BATCH_SIZE).repeat(REPEAT)
            image_stack_norm = image_stack_norm.batch(BATCH_SIZE).repeat(REPEAT)
            seg_stack_ds = seg_stack_ds.batch(BATCH_SIZE).repeat(REPEAT)
            intrinsics_ds = intrinsics_ds.batch(BATCH_SIZE).repeat(REPEAT)
            intrinsics_inv = intrinsics_inv.batch(BATCH_SIZE).repeat(REPEAT)

        # Create iterators over datasets
        image_it = image_stack_ds.make_one_shot_iterator().get_next()
        image_norm_it = image_stack_norm.make_one_shot_iterator().get_next()
        seg_it = seg_stack_ds.make_one_shot_iterator().get_next()
        intrinsics_it = intrinsics_ds.make_one_shot_iterator().get_next()
        intrinsics_inv_it = intrinsics_inv.make_one_shot_iterator().get_next()

        # Create image summary
        tf.summary.image('Image 1', image_it[:,:,:,0:3])
        tf.summary.image('Image 2', image_it[0:4,:,:,3:6])
        tf.summary.image('Image 3', image_it[0:4,:,:,6:9])
        tf.summary.image('Seg Image 1', seg_it[0:4,:,:,0:3])
        tf.summary.image('Seg Image 2', seg_it[0:4,:,:,3:6])
        tf.summary.image('Seg Image 3', seg_it[0:4,:,:,6:9])
        tf.summary.tensor_summary('Camera intrinsics', intrinsics_it)
        tf.summary.tensor_summary('Camera intrinsics inverse', intrinsics_inv_it)


        # print(image_it)
        # print(image_norm_it)
        # print(seg_it)
        # print(intrinsics_it)
        # print(intrinsics_inv_it)
        # print(dataset)
        print("Image dataset: {}".format(image_stack_ds))
        print("Image norm: {}".format(image_stack_norm))
        print("Seg mask dataset: {}".format(seg_stack_ds))
        print("Intrinsics dataset: {}".format(intrinsics_ds))
        print("Intrinsics inverse dataset: {}".format(intrinsics_inv))

        with tf.Session() as sess:
            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            merged_summary = tf.summary.merge_all()
            writer = tf.summary.FileWriter("/mnt/isaac/apps/carter_sim_struct2depth/logs/5")
            writer.add_graph(sess.graph)

            for i in range(5):
                summary = sess.run(merged_summary)
                writer.add_summary(summary, i)
