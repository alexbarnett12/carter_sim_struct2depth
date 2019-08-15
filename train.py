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

"""Trains a monocular depth and egomotion network. Can optionally train to predict object motion.
   Please refer to README for example usage."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import time
import json

from absl import app
from absl import logging

import numpy as np
import tensorflow as tf

from isaac_app import create_isaac_app, start_isaac_app
from struct2depth import model
from struct2depth import nets
from struct2depth import util

gfile = tf.gfile

# Paths to config files with training parameters. Most of the editable parameters should be modified through these files.
TRAINING_CONFIG_PATH = "/mnt/isaac_2019_2/apps/carter_sim_struct2depth/configs/train_parameters.json"
ISAAC_CONFIG_PATH = "/mnt/isaac_2019_2/apps/carter_sim_struct2depth/configs/isaac_parameters.json"

def load_training_parameters():
    with open(TRAINING_CONFIG_PATH) as f:
        config = json.load(f)

    return config["data_dir"], \
           config["using_saved_images"], \
           config["pretrained_ckpt"], \
           config["imagenet_ckpt"], \
           config["checkpoint_dir"], \
           config["cuda_device"], \
           config["file_extension"], \
           config["batch_size"], \
           config["save_ckpt_every"], \
           config["learning_rate"], \
           config["beta1"], \
           config["reconstr_weight"], \
           config["ssim_weight"], \
           config["smooth_weight"], \
           config["icp_weight"], \
           config["size_constraint_weight"], \
           config["img_height"], \
           config["img_width"], \
           config["seq_length"], \
           config["architecture"], \
           config["imagenet_norm"], \
           config["weight_reg"], \
           config["exhaustive_mode"], \
           config["random_scale_crop"], \
           config["flipping_mode"], \
           config["train_steps"], \
           config["summary_freq"], \
           config["depth_upsampling"], \
           config["depth_normalization"], \
           config["compute_minimum_loss"], \
           config["use_skip"], \
           config["equal_weighting"], \
           config["joint_encoder"], \
           config["handle_motion"], \
           config["master"], \
           config["shuffle"], \
           config["max_ckpts_to_keep"]

def load_isaac_parameters():
    with open(ISAAC_CONFIG_PATH) as f:
        config = json.load(f)

    return config["isaac_app_filename"], \
           config["time_delay"], \
           config["num_isaac_samples"], \
           config["speed_threshold"], \
           config["angular_speed_threshold"]

# Checks that chosen parameters do not conflict with each other and do not extend beyond the current scope of the project.
def verify_parameters(data_dir, handle_motion, joint_encoder, seq_length, compute_minimum_loss, img_height, img_width,
                      imagenet_ckpt, imagenet_norm, architecture, exhaustive_mode, icp_weight, checkpoint_dir):
    if handle_motion and joint_encoder:
        raise ValueError('Using a joint encoder is currently not supported when '
                         'modeling object motion.')
    if handle_motion and seq_length != 3:
        raise ValueError('The current motion model implementation only supports '
                         'using a sequence length of three.')
    if handle_motion and not compute_minimum_loss:
        raise ValueError('Computing the minimum photometric loss is required when '
                         'enabling object motion handling.')
        # if FLAGS.size_constraint_weight > 0 and not FLAGS.handle_motion:
        #   raise ValueError('To enforce object size constraints, enable motion '
        #                    'handling.')
    if imagenet_ckpt and not imagenet_norm:
        logging.warn('When initializing with an ImageNet-pretrained model, it is '
                     'recommended to normalize the image inputs accordingly using '
                     'imagenet_norm.')
    if compute_minimum_loss and seq_length % 2 != 1:
        raise ValueError('Compute minimum loss requires using an odd number of '
                         'images in a sequence.')
    if architecture != nets.RESNET and imagenet_ckpt:
        raise ValueError('Can only load weights from pre-trained ImageNet model '
                         'when using ResNet-architecture.')
    if compute_minimum_loss and exhaustive_mode:
        raise ValueError('Exhaustive mode has no effect when compute_minimum_loss '
                         'is enabled.')
    if img_width % (2 ** 5) != 0 or img_height % (2 ** 5) != 0:
        logging.warn('Image size is not divisible by 2^5. For the architecture '
                     'employed, this could cause artefacts caused by resizing in '
                     'lower dimensions.')
    if icp_weight > 0.0:
        raise ValueError('ICP is currently not supported.')

    if checkpoint_dir is None:
        raise ValueError('Must specify a checkpoint directory')

        # Check if data paths exist
    if not gfile.Exists(data_dir):
        raise ValueError("Not a valid data directory")

    # Create a checkpoint directory if it does not already exist.
    if not gfile.Exists(checkpoint_dir):
        gfile.MakeDirs(checkpoint_dir)


def main(_):
    # Fixed seed for repeatability
    seed = 8964
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Load training parameters
    data_dir, \
    using_saved_images, \
    pretrained_ckpt, \
    imagenet_ckpt, \
    checkpoint_dir, \
    cuda_device, \
    file_extension, \
    batch_size, \
    save_ckpt_every, \
    learning_rate, \
    beta1, \
    reconstr_weight, \
    ssim_weight, \
    smooth_weight, \
    icp_weight, \
    size_constraint_weight, \
    img_height, \
    img_width, \
    seq_length, \
    architecture, \
    imagenet_norm, \
    weight_reg, \
    exhaustive_mode, \
    random_scale_crop, \
    flipping_mode, \
    train_steps, \
    summary_freq, \
    depth_upsampling, \
    depth_normalization, \
    compute_minimum_loss, \
    use_skip, \
    equal_weighting, \
    joint_encoder, \
    handle_motion, \
    master, \
    shuffle, \
    max_ckpts_to_keep = load_training_parameters()

    # Load isaac sim parameters
    isaac_app_filename, \
    time_delay, \
    num_isaac_samples, \
    speed_threshold, \
    angular_speed_threshold = load_isaac_parameters()

    # Ensure that parameters aren't breaking current model functionality
    verify_parameters(data_dir, handle_motion, joint_encoder, seq_length, compute_minimum_loss, img_height, img_width,
                      imagenet_ckpt, imagenet_norm, architecture, exhaustive_mode, icp_weight, checkpoint_dir)

    # Create Isaac application.
    isaac_app = None
    if not using_saved_images:
        isaac_app = create_isaac_app(isaac_app_filename)

    # Set which GPU to run
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device

    # Create the training model.
    train_model = model.Model(data_dir=data_dir,
                              using_saved_images=using_saved_images,
                              file_extension=file_extension,
                              is_training=True,
                              learning_rate=learning_rate,
                              beta1=beta1,
                              reconstr_weight=reconstr_weight,
                              smooth_weight=smooth_weight,
                              ssim_weight=ssim_weight,
                              icp_weight=icp_weight,
                              batch_size=batch_size,
                              img_height=img_height,
                              img_width=img_width,
                              seq_length=seq_length,
                              architecture=architecture,
                              imagenet_norm=imagenet_norm,
                              weight_reg=weight_reg,
                              exhaustive_mode=exhaustive_mode,
                              random_scale_crop=random_scale_crop,
                              flipping_mode=flipping_mode,
                              depth_upsampling=depth_upsampling,
                              depth_normalization=depth_normalization,
                              compute_minimum_loss=compute_minimum_loss,
                              use_skip=use_skip,
                              joint_encoder=joint_encoder,
                              shuffle=shuffle,
                              handle_motion=handle_motion,
                              equal_weighting=equal_weighting,
                              size_constraint_weight=size_constraint_weight,
                              isaac_app=isaac_app,
                              time_delay=time_delay,
                              num_isaac_samples=num_isaac_samples,
                              speed_threshold=speed_threshold,
                              angular_speed_threshold=angular_speed_threshold)

    # Perform training
    train(train_model, pretrained_ckpt, imagenet_ckpt, checkpoint_dir, train_steps,
          summary_freq, isaac_app, max_ckpts_to_keep, save_ckpt_every, using_saved_images)

# Train the model. First attempts to restore either a pretrained Imagenet checkpoint or the most recent checkpoint
# in the checkpoint directory. Then loops until max training steps specified has passed. Periodically saves checkpoints
# tf summaries.
def train(train_model, pretrained_ckpt, imagenet_ckpt, checkpoint_dir, train_steps,
          summary_freq, isaac_app, max_ckpts_to_keep, save_ckpt_every, using_saved_images):

    # Restore variables from checkpoint
    vars_to_restore = None
    if pretrained_ckpt is not None:
        vars_to_restore = util.get_vars_to_save_and_restore(pretrained_ckpt)
        ckpt_path = pretrained_ckpt
    elif imagenet_ckpt:
        vars_to_restore = util.get_imagenet_vars_to_restore(imagenet_ckpt)
        ckpt_path = imagenet_ckpt

    pretrain_restorer = tf.train.Saver(vars_to_restore)
    vars_to_save = util.get_vars_to_save_and_restore()
    vars_to_save[train_model.global_step.op.name] = train_model.global_step
    saver = tf.train.Saver(vars_to_save, max_to_keep=max_ckpts_to_keep)

    # Create a supervisor.
    sv = tf.train.Supervisor(logdir=checkpoint_dir, save_summaries_secs=0,
                             saver=None)

    # Set configs.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True # Doesn't limit GPU usage.
    with sv.managed_session(config=config) as sess:

        # Load ckpt variables.
        if pretrained_ckpt is not None or imagenet_ckpt:
            logging.info('Restoring pretrained weights from %s', ckpt_path)
            pretrain_restorer.restore(sess, ckpt_path)

        logging.info('Attempting to resume training from %s...', checkpoint_dir)
        checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        logging.info('Last checkpoint found: %s', checkpoint)
        if checkpoint:
            saver.restore(sess, checkpoint)

        logging.info('Training...')

        # Start the Isaac application and Sight server.
        if not using_saved_images:
            start_isaac_app(isaac_app)
            logging.info("Isaac application loaded")

        start_time = time.time()
        last_summary_time = time.time()
        steps_per_epoch = train_model.reader.steps_per_epoch
        step = 1
        while step < train_steps:

            # Define training on one batch.
            fetches = {
                'train': train_model.train_op,
                'global_step': train_model.global_step,
                'incr_global_step': train_model.incr_global_step
            }

            # Retrieve loss and summaries.
            if step % summary_freq == 0:
                fetches['loss'] = train_model.total_loss
                fetches['summary'] = sv.summary_op

            # Execute training
            results = sess.run(fetches)
            global_step = results['global_step']

            # Save summaries.
            if step % summary_freq == 0:
                sv.summary_writer.add_summary(results['summary'], global_step)

                # Calculate current epoch, training step, and cycle.
                train_epoch = math.ceil(global_step / steps_per_epoch)
                train_step = global_step - (train_epoch - 1) * steps_per_epoch
                this_cycle = time.time() - last_summary_time
                last_summary_time += this_cycle

                logging.info(
                    'Epoch: [%2d] [%5d/%5d] time: %4.2fs (%ds total) loss: %.3f',
                    train_epoch, train_step, steps_per_epoch, this_cycle,
                    time.time() - start_time, results['loss'])

            # Save ckpts.
            if step % save_ckpt_every == 0:
                logging.info('[*] Saving checkpoint to %s...', checkpoint_dir)
                saver.save(sess, os.path.join(checkpoint_dir, 'model'),
                           global_step=global_step)

            # Setting step to global_step allows for training for a total of
            # train_steps even if the program is restarted during training.
            step = global_step + 1


if __name__ == '__main__':
    app.run(main)
