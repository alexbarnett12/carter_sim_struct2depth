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

"""Applies online refinement while running inference.

Instructions: Run static inference first before calling this script. Make sure
to point output_dir to the same folder where static inference results were
saved previously.

For example use, please refer to README.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import random
import time
from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow as tf
import json

from isaac_app import create_isaac_app, start_isaac_app
from struct2depth import model
from struct2depth import nets
from struct2depth import reader
from struct2depth import util

gfile = tf.gfile
FIXED_SEED = 8964  # Fixed seed for repeatability.

TRAINING_CONFIG_PATH = "/mnt/isaac_2019_2/apps/carter_sim_struct2depth/configs/optimize_parameters.json"
ISAAC_CONFIG_PATH = "/mnt/isaac_2019_2/apps/carter_sim_struct2depth/configs/isaac_parameters.json"

def load_training_parameters():
    with open(TRAINING_CONFIG_PATH) as f:
        config = json.load(f)

    return config["output_dir"], \
           config["data_dir"], \
           config["using_saved_images"], \
           config["model_ckpt"], \
           config["checkpoint_dir"], \
           config["cuda_device"], \
           config["save_every"], \
           config["save_previews"], \
           config["file_extension"], \
           config["batch_size"], \
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
           config["depth_upsampling"], \
           config["depth_normalization"], \
           config["compute_minimum_loss"], \
           config["use_skip"], \
           config["joint_encoder"], \
           config["egomotion_threshold"], \
           config["num_steps"], \
           config["handle_motion"], \
           config["flip"]


def load_isaac_parameters():
    with open(ISAAC_CONFIG_PATH) as f:
        config = json.load(f)

    return config["isaac_app_filename"], \
           config["time_delay"], \
           config["num_isaac_samples"], \
           config["speed_threshold"], \
           config["angular_speed_threshold"]


def verify_parameters(output_dir, image_dir, seg_mask_dir, intrinsics_dir, handle_motion, joint_encoder, seq_length,
                      compute_minimum_loss, img_height, img_width,
                      exhaustive_mode, icp_weight, checkpoint_dir,
                      size_constraint_weight):
    if handle_motion and joint_encoder:
        raise ValueError('Using a joint encoder is currently not supported when '
                         'modeling object motion.')
    if handle_motion and seq_length != 3:
        raise ValueError('The current motion model implementation only supports '
                         'using a sequence length of three.')
    if handle_motion and not compute_minimum_loss:
        raise ValueError('Computing the minimum photometric loss is required when '
                         'enabling object motion handling.')
    if size_constraint_weight > 0 and not handle_motion:
        raise ValueError('To enforce object size constraints, enable motion '
                         'handling.')
    if icp_weight > 0.0:
        raise ValueError('ICP is currently not supported.')
    if compute_minimum_loss and seq_length % 2 != 1:
        raise ValueError('Compute minimum loss requires using an odd number of '
                         'images in a sequence.')
    if compute_minimum_loss and exhaustive_mode:
        raise ValueError('Exhaustive mode has no effect when compute_minimum_loss '
                         'is enabled.')
    if img_width % (2 ** 5) != 0 or img_height % (2 ** 5) != 0:
        logging.warn('Image size is not divisible by 2^5. For the architecture '
                     'employed, this could cause artefacts caused by resizing in '
                     'lower dimensions.')

    if output_dir.endswith('/'):
        output_dir = output_dir[:-1]
    if image_dir.endswith('/'):
        image_dir = image_dir[:-1]
    if seg_mask_dir.endswith('/'):
        seg_mask_dir = seg_mask_dir[:-1]
    if intrinsics_dir.endswith('/'):
        intrinsics_dir = intrinsics_dir[:-1]
    if checkpoint_dir is None:
        raise ValueError('Must specify a checkpoint directory')

    if not gfile.Exists(checkpoint_dir):
        gfile.MakeDirs(checkpoint_dir)
    if not gfile.Exists(output_dir):
        gfile.MakeDirs(output_dir)

    return output_dir, image_dir, seg_mask_dir, intrinsics_dir


def main(_):
    """Runs fine-tuning and inference.

    There are three categories of images.
    1) Images where we have previous and next frame, and that are not filtered
       out by the heuristic. For them, we will use the fine-tuned predictions.
    2) Images where we have previous and next frame, but that were filtered out
       by our heuristic. For them, we will use the ordinary prediction instead.
    3) Images where we have at least one missing adjacent frame. For them, we will
       use the ordinary prediction as indicated by triplet_list_file_remains (if
       provided). They will also not be part of the generated inference list in
       the first place.

    Raises:
       ValueError: Invalid parameters have been passed.
    """

    # Load optimize parameters
    output_dir, \
    data_dir, \
    using_saved_images, \
    model_ckpt, \
    checkpoint_dir, \
    cuda_device, \
    save_every, \
    save_previews, \
    file_extension, \
    batch_size, \
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
    depth_upsampling, \
    depth_normalization, \
    compute_minimum_loss, \
    use_skip, \
    joint_encoder, \
    egomotion_threshold, \
    num_steps, \
    handle_motion, \
    flip = load_training_parameters()

    isaac_app_filename, \
    time_delay, \
    num_isaac_samples, \
    speed_threshold, \
    angular_speed_threshold = load_isaac_parameters()

    # Create Isaac application.
    isaac_app = None
    if not using_saved_images:
        isaac_app = create_isaac_app(filename=isaac_app_filename)
        logging.info("Isaac application loaded")

    # Run fine-tuning process and save predictions in id-folders.
    tf.set_random_seed(FIXED_SEED)
    np.random.seed(FIXED_SEED)
    random.seed(FIXED_SEED)
    flipping_mode = reader.FLIP_ALWAYS if flip else reader.FLIP_NONE
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
                              random_color=False,
                              depth_upsampling=depth_upsampling,
                              depth_normalization=depth_normalization,
                              compute_minimum_loss=compute_minimum_loss,
                              use_skip=use_skip,
                              joint_encoder=joint_encoder,
                              build_sum=False,
                              shuffle=False,
                              input_file=None,
                              handle_motion=handle_motion,
                              size_constraint_weight=size_constraint_weight,
                              train_global_scale_var=False,
                              isaac_app=isaac_app,
                              optimize=True,
                              num_steps=num_steps)

    finetune_inference(train_model, model_ckpt, output_dir, isaac_app, using_saved_images, num_steps, batch_size, flip,
                       save_every, save_previews, file_extension)


# Run inference while finetuning model
def finetune_inference(train_model, model_ckpt, output_dir, isaac_app, using_saved_images, num_steps, batch_size, flip,
                       save_every, save_previews, file_extension):

    # Restore checkpoint variables
    vars_to_restore = None
    if model_ckpt is not None:
        vars_to_restore = util.get_vars_to_save_and_restore(model_ckpt)
        ckpt_path = model_ckpt
    pretrain_restorer = tf.train.Saver(vars_to_restore)
    sv = tf.train.Supervisor(logdir=None, save_summaries_secs=0, saver=None,
                             summary_op=None)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with sv.managed_session(config=config) as sess:

        # Start the application and Sight server
        if not using_saved_images:
            start_isaac_app(isaac_app)
            logging.info("Isaac application loaded")

        # Restore ckpt weights
        if model_ckpt is not None:
            logging.info('Restored weights from %s', ckpt_path)
            pretrain_restorer.restore(sess, ckpt_path)

        # Create output directory
        if not gfile.Exists(output_dir):
            gfile.MakeDirs(output_dir)

        logging.info('Running fine-tuning:')
        img_nr = 0
        step = 1
        while True:

            # Run fine-tuning.
            logging.info('Running step %s of %s.', step, int(num_steps / batch_size))
            fetches = {
                'train': train_model.train_op,
                'global_step': train_model.global_step,
                'incr_global_step': train_model.incr_global_step
            }
            _ = sess.run(fetches)
            if step % save_every == 0:
                # Get latest prediction for middle frame, highest scale.
                pred = train_model.depth[1][0].eval(session=sess)
                if flip:
                    pred = np.flip(pred, axis=2)
                input_img = train_model.image_stack.eval(session=sess)
                input_img_prev = input_img[0, :, :, 0:3]
                input_img_center = input_img[0, :, :, 3:6]
                input_img_next = input_img[0, :, :, 6:]
                img_pred_dir = os.path.join(output_dir, str(img_nr))
                img_pred_file = os.path.join(img_pred_dir,
                                             str(step).zfill(10) + ('_flip' if flip else '') + '.npy')
                if not os.path.exists(img_pred_dir):
                    os.mkdir(img_pred_dir)
                motion = np.squeeze(train_model.egomotion.eval(session=sess))
                # motion of shape (seq_length - 1, 6).
                motion = np.mean(motion, axis=0)  # Average egomotion across frames.

                if save_previews or step == num_steps:
                    # Also save preview of depth map.
                    color_map = util.normalize_depth_for_display(
                        np.squeeze(pred[0, :, :]))
                    visualization = np.concatenate(
                        (input_img_prev, input_img_center, input_img_next, color_map))
                    motion_s = [str(m) for m in motion]
                    s_rep = ','.join(motion_s)
                    with gfile.Open(img_pred_file.replace('.npy', '.txt'), 'w') as f:
                        f.write(s_rep)
                    util.save_image(
                        img_pred_file.replace('.npy', '.%s' % file_extension),
                        visualization, file_extension)

                with gfile.Open(img_pred_file, 'wb') as f:
                    np.save(f, pred)

            # # Apply heuristic to not finetune if egomotion magnitude is too low.
            # ego_magnitude = np.linalg.norm(motion[:3], ord=2)
            # heuristic = ego_magnitude >= FLAGS.egomotion_threshold
            # if not heuristic and step == FLAGS.num_steps:
            #   failed_heuristic.append(img_nr)
            if step == num_steps:
                step = 0
                img_nr += 1

            step += 1


if __name__ == '__main__':
    app.run(main)
