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

import datetime
import os
import sys
import random
import time
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
ROOT_DIR = os.path.abspath("/mnt/isaac/")  # Root directory of the Isaac
sys.path.append(ROOT_DIR)
from engine.pyalice import *
import packages.ml

# Map strings
WAREHOUSE = 'carter_warehouse_p'
HOSPITAL = 'hospital'

# Isaac Sim flags
flags.DEFINE_string('graph_filename', "apps/carter_sim_struct2depth/carter.graph.json",
                    'Where the isaac SDK app graph is stored')
flags.DEFINE_string('config_filename', "apps/carter_sim_struct2depth/carter.config.json",
                    'Where the isaac SDK app node configuration is stored')
flags.DEFINE_string('map_config_filename', "apps/assets/maps/" + HOSPITAL + ".config.json",
                    "Config file for Isaac Sim map")
flags.DEFINE_string('map_graph_filename', "apps/assets/maps/" + HOSPITAL + ".graph.json",
                    "Graph file for Isaac Sim map")

gfile = tf.gfile
SAVE_EVERY = 1  # Defines the interval that predictions should be saved at.
SAVE_PREVIEWS = True  # If set, while save image previews of depth predictions.
FIXED_SEED = 8964  # Fixed seed for repeatability.

flags.DEFINE_string('output_dir', '/mnt/isaac/apps/carter_sim_struct2depth/results/results_sim_ft_20', 'Directory to store predictions. '
                                                         'Assumes that regular inference has been executed before '
                                                         'and results were stored in this folder.')
flags.DEFINE_string('data_dir', None, 'Folder pointing to preprocessed '
                                      'triplets to fine-tune on.')
flags.DEFINE_string('triplet_list_file', None, 'Text file containing paths to '
                                               'image files to process. Paths should be relative with '
                                               'respect to the list file location. Every line should be '
                                               'of the form [input_folder_name] [input_frame_num] '
                                               '[output_path], where [output_path] is optional to specify '
                                               'a different path to store the prediction.')
flags.DEFINE_string('triplet_list_file_remains', None, 'Optional text file '
                                                       'containing relative paths to image files which should not '
                                                       'be fine-tuned, e.g. because of missing adjacent frames. '
                                                       'For all files listed, the static prediction will be '
                                                       'copied instead. File can be empty. If not, every line '
                                                       'should be of the form [input_folder_name] '
                                                       '[input_frame_num] [output_path], where [output_path] is '
                                                       'optional to specify a different path to take and store '
                                                       'the unrefined prediction from/to.')
flags.DEFINE_string('model_ckpt',
                    '/mnt/isaac/apps/carter_sim_struct2depth/struct2depth/pretrained_ckpt/model-199160.ckpt',
                    'Model checkpoint to optimize.')
flags.DEFINE_string('ft_name', '', 'Optional prefix for temporary files.')
flags.DEFINE_string('file_extension', 'png', 'Image data file extension.')
flags.DEFINE_float('learning_rate', 0.0001, 'Adam learning rate.')
flags.DEFINE_float('beta1', 0.9, 'Adam momentum.')
flags.DEFINE_float('reconstr_weight', 0.85, 'Frame reconstruction loss weight.')
flags.DEFINE_float('ssim_weight', 0.15, 'SSIM loss weight.')
flags.DEFINE_float('smooth_weight', 0.01, 'Smoothness loss weight.')
flags.DEFINE_float('icp_weight', 0.0, 'ICP loss weight.')
flags.DEFINE_float('size_constraint_weight', 0.0005, 'Weight of the object '
                                                     'size constraint loss. Use only with motion handling.')
flags.DEFINE_integer('batch_size', 1, 'The size of a sample batch')
flags.DEFINE_integer('img_height', 128, 'Input frame height.')
flags.DEFINE_integer('img_width', 416, 'Input frame width.')
flags.DEFINE_integer('seq_length', 3, 'Number of frames in sequence.')
flags.DEFINE_enum('architecture', nets.RESNET, nets.ARCHITECTURES,
                  'Defines the architecture to use for the depth prediction '
                  'network. Defaults to ResNet-based encoder and accompanying '
                  'decoder.')
flags.DEFINE_boolean('imagenet_norm', True, 'Whether to normalize the input '
                                            'images channel-wise so that they match the distribution '
                                            'most ImageNet-models were trained on.')
flags.DEFINE_float('weight_reg', 0.05, 'The amount of weight regularization to '
                                       'apply. This has no effect on the ResNet-based encoder '
                                       'architecture.')
flags.DEFINE_boolean('exhaustive_mode', False, 'Whether to exhaustively warp '
                                               'from any frame to any other instead of just considering '
                                               'adjacent frames. Where necessary, multiple egomotion '
                                               'estimates will be applied. Does not have an effect if '
                                               'compute_minimum_loss is enabled.')
flags.DEFINE_boolean('random_scale_crop', False, 'Whether to apply random '
                                                 'image scaling and center cropping during training.')
flags.DEFINE_bool('depth_upsampling', True, 'Whether to apply depth '
                                            'upsampling of lower-scale representations before warping to '
                                            'compute reconstruction loss on full-resolution image.')
flags.DEFINE_bool('depth_normalization', True, 'Whether to apply depth '
                                               'normalization, that is, normalizing inverse depth '
                                               'prediction maps by their mean to avoid degeneration towards '
                                               'small values.')
flags.DEFINE_bool('compute_minimum_loss', True, 'Whether to take the '
                                                'element-wise minimum of the reconstruction/SSIM error in '
                                                'order to avoid overly penalizing dis-occlusion effects.')
flags.DEFINE_bool('use_skip', True, 'Whether to use skip connections in the '
                                    'encoder-decoder architecture.')
flags.DEFINE_bool('joint_encoder', False, 'Whether to share parameters '
                                          'between the depth and egomotion networks by using a joint '
                                          'encoder architecture. The egomotion network is then '
                                          'operating only on the hidden representation provided by the '
                                          'joint encoder.')
flags.DEFINE_float('egomotion_threshold', 0.01, 'Minimum egomotion magnitude '
                                                'to apply finetuning. If lower, just forwards the ordinary '
                                                'prediction.')
flags.DEFINE_integer('num_steps', 5, 'Number of optimization steps to run.')
flags.DEFINE_boolean('handle_motion', True, 'Whether the checkpoint was '
                                            'trained with motion handling.')
flags.DEFINE_bool('flip', False, 'Whether images should be flipped as well as '
                                 'resulting predictions (for test-time augmentation). This '
                                 'currently applies to the depth network only.')

FLAGS = flags.FLAGS


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

    if FLAGS.handle_motion and FLAGS.joint_encoder:
        raise ValueError('Using a joint encoder is currently not supported when '
                         'modeling object motion.')
    if FLAGS.handle_motion and FLAGS.seq_length != 3:
        raise ValueError('The current motion model implementation only supports '
                         'using a sequence length of three.')
    if FLAGS.handle_motion and not FLAGS.compute_minimum_loss:
        raise ValueError('Computing the minimum photometric loss is required when '
                         'enabling object motion handling.')
    if FLAGS.size_constraint_weight > 0 and not FLAGS.handle_motion:
        raise ValueError('To enforce object size constraints, enable motion '
                         'handling.')
    if FLAGS.icp_weight > 0.0:
        raise ValueError('ICP is currently not supported.')
    if FLAGS.compute_minimum_loss and FLAGS.seq_length % 2 != 1:
        raise ValueError('Compute minimum loss requires using an odd number of '
                         'images in a sequence.')
    if FLAGS.compute_minimum_loss and FLAGS.exhaustive_mode:
        raise ValueError('Exhaustive mode has no effect when compute_minimum_loss '
                         'is enabled.')
    if FLAGS.img_width % (2 ** 5) != 0 or FLAGS.img_height % (2 ** 5) != 0:
        logging.warn('Image size is not divisible by 2^5. For the architecture '
                     'employed, this could cause artefacts caused by resizing in '
                     'lower dimensions.')

    if FLAGS.output_dir.endswith('/'):
        FLAGS.output_dir = FLAGS.output_dir[:-1]

    # Create Isaac application.
    isaac_app = Application(name="carter_sim", modules=["map",
                                                        "navigation",
                                                        "perception",
                                                        "planner",
                                                        "viewers",
                                                        "flatsim",
                                                        "//packages/ml:ml"])

    # Load config files
    isaac_app.load_config(FLAGS.config_filename)
    isaac_app.load_config("apps/carter_sim_struct2depth/navigation.config.json")
    isaac_app.load_config(FLAGS.map_config_filename)

    # Load graph files
    isaac_app.load_graph(FLAGS.graph_filename)
    isaac_app.load_graph("apps/carter_sim_struct2depth/navigation.graph.json")
    isaac_app.load_graph(FLAGS.map_graph_filename)
    isaac_app.load_graph("apps/carter_sim_struct2depth/base_control.graph.json")

    # Start the application and Sight server
    isaac_app.start_webserver()
    isaac_app.start()
    logging.info("Isaac application loaded")

    # Run fine-tuning process and save predictions in id-folders.
    tf.set_random_seed(FIXED_SEED)
    np.random.seed(FIXED_SEED)
    random.seed(FIXED_SEED)
    flipping_mode = reader.FLIP_ALWAYS if FLAGS.flip else reader.FLIP_NONE
    train_model = model.Model(data_dir=FLAGS.data_dir,
                              file_extension=FLAGS.file_extension,
                              is_training=True,
                              learning_rate=FLAGS.learning_rate,
                              beta1=FLAGS.beta1,
                              reconstr_weight=FLAGS.reconstr_weight,
                              smooth_weight=FLAGS.smooth_weight,
                              ssim_weight=FLAGS.ssim_weight,
                              icp_weight=FLAGS.icp_weight,
                              batch_size=FLAGS.batch_size,
                              img_height=FLAGS.img_height,
                              img_width=FLAGS.img_width,
                              seq_length=FLAGS.seq_length,
                              architecture=FLAGS.architecture,
                              imagenet_norm=FLAGS.imagenet_norm,
                              weight_reg=FLAGS.weight_reg,
                              exhaustive_mode=FLAGS.exhaustive_mode,
                              random_scale_crop=FLAGS.random_scale_crop,
                              flipping_mode=flipping_mode,
                              random_color=False,
                              depth_upsampling=FLAGS.depth_upsampling,
                              depth_normalization=FLAGS.depth_normalization,
                              compute_minimum_loss=FLAGS.compute_minimum_loss,
                              use_skip=FLAGS.use_skip,
                              joint_encoder=FLAGS.joint_encoder,
                              build_sum=False,
                              shuffle=False,
                              input_file=None,
                              handle_motion=FLAGS.handle_motion,
                              size_constraint_weight=FLAGS.size_constraint_weight,
                              train_global_scale_var=False,
                              isaac_app=isaac_app,
                              optimize=True,
                              num_steps=FLAGS.num_steps)

    finetune_inference(train_model, FLAGS.model_ckpt,
                       FLAGS.output_dir, isaac_app)


def finetune_inference(train_model, model_ckpt, output_dir, isaac_app):
    """Train model."""
    vars_to_restore = None
    if model_ckpt is not None:
        vars_to_restore = util.get_vars_to_save_and_restore(model_ckpt)
        ckpt_path = model_ckpt
    pretrain_restorer = tf.train.Saver(vars_to_restore)
    sv = tf.train.Supervisor(logdir=None, save_summaries_secs=0, saver=None,
                             summary_op=None)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    img_nr = 0
    failed_heuristic = []
    with sv.managed_session(config=config) as sess:
        # TODO: Caching the weights would be better to avoid I/O bottleneck.

        node = isaac_app.find_node_by_name("CarterTrainingSamples")
        bridge = packages.ml.SampleAccumulator(node)

        # Wait until we get enough samples from Isaac
        while True:
            num = bridge.get_sample_number()
            if num >= 1:
                break
            time.sleep(1.0)
            logging.info("waiting for enough samples: {}".format(num))

        logging.info('Running fine-tuning:')
        step = 1
        while True:
            if model_ckpt is not None:
                logging.info('Restored weights from %s', ckpt_path)
                pretrain_restorer.restore(sess, ckpt_path)
            if not gfile.Exists(output_dir):
                gfile.MakeDirs(output_dir)

            # Run fine-tuning.
            logging.info('Training:')
            fetches = {
                'train': train_model.train_op,
                'global_step': train_model.global_step,
                'incr_global_step': train_model.incr_global_step
            }
            _ = sess.run(fetches)
            if step % SAVE_EVERY == 0:
                # Get latest prediction for middle frame, highest scale.
                pred = train_model.depth[1][0].eval(session=sess)
                if FLAGS.flip:
                    pred = np.flip(pred, axis=2)
                input_img = train_model.image_stack.eval(session=sess)
                input_img_prev = input_img[0, :, :, 0:3]
                input_img_center = input_img[0, :, :, 3:6]
                input_img_next = input_img[0, :, :, 6:]
                img_pred_file = os.path.join(
                    output_dir,
                    str(step).zfill(10) + ('_flip' if FLAGS.flip else '') + '.npy')
                motion = np.squeeze(train_model.egomotion.eval(session=sess))
                # motion of shape (seq_length - 1, 6).
                motion = np.mean(motion, axis=0)  # Average egomotion across frames.

                if SAVE_PREVIEWS or step == FLAGS.num_steps:
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
                        img_pred_file.replace('.npy', '.%s' % FLAGS.file_extension),
                        visualization, FLAGS.file_extension)

                with gfile.Open(img_pred_file, 'wb') as f:
                    np.save(f, pred)

            # # Apply heuristic to not finetune if egomotion magnitude is too low.
            # ego_magnitude = np.linalg.norm(motion[:3], ord=2)
            # heuristic = ego_magnitude >= FLAGS.egomotion_threshold
            # if not heuristic and step == FLAGS.num_steps:
            #   failed_heuristic.append(img_nr)

                step += 1
            img_nr += 1


if __name__ == '__main__':
    app.run(main)
