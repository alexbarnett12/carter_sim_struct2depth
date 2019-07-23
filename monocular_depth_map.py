from __future__ import absolute_import, division, print_function
import os
import sys
import numpy as np
from absl import logging
# import matplotlib.pyplot as plt
from struct2depth import model
import numpy as np
import fnmatch
import tensorflow as tf
from struct2depth import nets
from struct2depth import util
import cv2

gfile = tf.gfile

ROOT_DIR = os.path.abspath("/mnt/isaac")
sys.path.append(ROOT_DIR)

# Parameters
OUTPUT_DIR = "/mnt/isaac/apps/carter_sim_struct2depth/results_actual"
FILE_EXT = "png"
DEPTH = True
EGOMOTION = False
MODEL_CKPT = "/mnt/isaac/apps/carter_sim_struct2depth/struct2depth/ckpts_saved_images/model-2772"
BATCH_SIZE = 1
IMG_HEIGHT = 128
IMG_WIDTH = 416
SEQ_LENGTH = 3
ARCHITECTURE = nets.RESNET
IMAGENET_NORM = True
USE_SKIP = True
JOINT_ENCODER = False
SHUFFLE = False
FLIP = False
INFERENCE_MODE_SINGLE = 'single'
INFERENCE_MODE_TRIPLET = 'triplets'
INFERENCE_CROP_NONE = 'none'
INFERENCE_MODE = INFERENCE_MODE_SINGLE
INFERENCE_CROP = INFERENCE_CROP_NONE
USE_MASKS = False

# Root directory of the Isaac
ROOT_DIR = os.path.abspath("/mnt/isaac/apps/carter_sim_struct2depth")
sys.path.append(ROOT_DIR)
from engine.pyalice import *


class MonocularDepthMap(Codelet):
    def start(self):
        # This part will be run once in the beginning of the program
        logging.info("Running MonocularDepthMap initialization")

        # Input and output messages for the Codelet
        self.rx = self.isaac_proto_rx("ColorCameraProto", "rgb_image")
        self.tx = self.isaac_proto_tx("DepthCameraProto", "depth_listener")
        self.depth_image_proto = self.isaac_proto_tx("ImageProto", "rgb_image")
        self.depth_pinhole_proto = self.isaac_proto_tx("PinholeProto", "pinhole")
        logging.info("RX and TX protos have been created")

        self.count = 0  # Count how many images taken

        # Create inference model
        self.inference_model = model.Model(is_training=False,
                                           batch_size=BATCH_SIZE,
                                           img_height=IMG_HEIGHT,
                                           img_width=IMG_WIDTH,
                                           seq_length=SEQ_LENGTH,
                                           architecture=ARCHITECTURE,
                                           imagenet_norm=IMAGENET_NORM,
                                           use_skip=USE_SKIP,
                                           joint_encoder=JOINT_ENCODER)
        logging.info("Inference model created")

        # Restore model ckpt and configuration settings
        vars_to_restore = util.get_vars_to_save_and_restore(MODEL_CKPT)
        self.saver = tf.train.Saver(vars_to_restore)
        self.sv = tf.train.Supervisor(logdir='/tmp/', saver=None)
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True

        if not gfile.Exists(OUTPUT_DIR):
            gfile.MakeDirs(OUTPUT_DIR)
        logging.info('Predictions will be saved in %s.', OUTPUT_DIR)
        logging.info("Initialization successful")

        # Tick every time we receive an image
        self.tick_on_message(self.rx)

    def tick(self):
        logging.info("Ticking")

        # Extract image proto
        rgb_image_proto = self.rx.get_proto().image

        # Extract image attributes
        element_type = rgb_image_proto.elementType
        rows = rgb_image_proto.rows
        cols = rgb_image_proto.cols
        channels = rgb_image_proto.channels
        image_buffer_id = rgb_image_proto.dataBufferIndex

        print("Element type: {}".format(element_type))
        print("Rows: {}".format(rows))
        print("Cols: {}".format(cols))
        print("Channels: {}".format(channels))
        print("Image buffer id: {}\n".format(image_buffer_id))

        # Read image buffer
        image_buffer = self.rx.get_buffer_content(image_buffer_id)

        # Transform into image
        image = np.frombuffer(image_buffer, dtype=np.uint8)
        image = np.reshape(image, (rows, cols, channels)) / 255.
        # print(image)
        print("Image initial shape: {}".format(np.shape(image)))
        # cv2.imshow('image',image)
        # k = cv2.waitKey(0)
        # if k == 27:         # wait for ESC key to exit
        #     cv2.destroyAllWindows()

        # Reshape to inference network size and homogeneous coordinates
        image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
        print("Image resized shape: {}".format(np.shape(image)))

        # cv2.imshow('image',image)
        # k = cv2.waitKey(0)
        # if k == 27:         # wait for ESC key to exit
        #     cv2.destroyAllWindows()
        image_tensor = np.reshape(image, (1, IMG_HEIGHT, IMG_WIDTH, channels))
        print("Image reshaped shape: {}".format(np.shape(image)))

        # Extract camera attributes
        pinhole_proto = self.rx.get_proto().pinhole

        # Run inference
        est_depth, visualization = self.run_inference(image_tensor,
                                                      output_dir=OUTPUT_DIR,
                                                      file_extension=FILE_EXT,
                                                      depth=DEPTH,
                                                      egomotion=EGOMOTION,
                                                      flip_for_depth=FLIP,
                                                      inference_crop=INFERENCE_CROP,
                                                      use_masks=USE_MASKS
                                                      )

        # cv2.imshow('Depth Map', visualization)
        # k = cv2.waitKey(0)
        # if k == 27:
        #     cv2.destroyAllWindows()

        print("Depth map shape: {}".format(np.shape(est_depth)))
        print("Depth min: {}".format(np.min(est_depth)))
        print("Depth max: {}".format(np.max(est_depth)))

        # Initialize tx protos
        depth_camera_viewer = self.tx.init_proto()
        depth_image_proto = self.depth_image_proto.init_proto()
        depth_pinhole_proto = self.depth_pinhole_proto.init_proto()

        # Set image proto attributes
        depth_image_proto.elementType = 'float32'
        depth_image_proto.rows = IMG_HEIGHT
        depth_image_proto.cols = IMG_WIDTH
        depth_image_proto.channels = 1

        # Serialize image
        depth_buffer = est_depth.tobytes()
        depth_image_proto.dataBufferIndex = 0  # TODO: Figure out how to store image data as buffer

        # Set pinhole intrinsics
        depth_pinhole_proto.rows = IMG_HEIGHT
        depth_pinhole_proto.cols = IMG_WIDTH
        depth_pinhole_proto.focal = pinhole_proto.focal  # TODO: Check that these values are correct
        depth_pinhole_proto.center = pinhole_proto.center
        print("Pinhole focal: {}".format(depth_pinhole_proto.focal))
        print("Pinhole center: {}".format(depth_pinhole_proto.center))

        # Set depth camera proto
        depth_camera_viewer.depthImage = depth_image_proto
        depth_camera_viewer.minDepth = 0
        depth_camera_viewer.maxDepth = 1  # TODO: Find min and maxes
        depth_camera_viewer.pinhole = depth_pinhole_proto

        # Publish DepthCameraProto
        self.tx.publish()
        logging.info("Published!")

    # Run inference. Expects input image with values from 0-1.
    def run_inference(self,
                      image=None,
                      output_dir=None,
                      file_extension='png',
                      depth=True,
                      egomotion=False,
                      flip_for_depth=False,
                      inference_crop=INFERENCE_CROP_NONE,
                      use_masks=False
                      ):

        est_depth = None
        visualization = None

        logging.info('Running inference:')
        # Run depth prediction network.
        if depth:

            # Flip
            if flip_for_depth:
                image = np.flip(image, axis=1)

            with self.sv.managed_session(config=self.config) as sess:
                self.saver.restore(sess, MODEL_CKPT)
                logging.info("Model ckpt restored")

                est_depth = self.inference_model.inference_depth(image, sess)

            # Flip back
            if flip_for_depth:
                est_depth = np.flip(est_depth, axis=2)
                image = np.flip(image, axis=2)

            # Reshape image from homogeneous coordinates to standard
            image = np.reshape(image, (IMG_HEIGHT, IMG_WIDTH, 3))

            # cv2.imshow('image',image)
            # k = cv2.waitKey(0)
            # if k == 27:         # wait for ESC key to exit
            #     cv2.destroyAllWindows()

            # Create color map for visualization
            color_map = util.normalize_depth_for_display(np.squeeze(est_depth))
            visualization = np.concatenate((image, color_map), axis=0)

            # Save images
            # Save raw prediction and color visualization. Extract filename
            # without extension from full path: e.g. path/to/input_dir/folder1/
            # file1.png -> file1

            pref = '_flip' if flip_for_depth else ''
            output_raw = os.path.join(output_dir, 'image' + pref + '{}'.format(self.count) + '.npy')
            output_vis = os.path.join(output_dir, 'image' + pref + '{}'.format(self.count) + '.png')

            with gfile.Open(output_raw, 'wb') as f:
                np.save(f, est_depth)
            util.save_image(output_vis, visualization, file_extension)
            self.count += 1
            logging.info("Inference successful")
        return est_depth, visualization

    # Run egomotion network.
    # if egomotion:
    #     if inference_mode == INFERENCE_MODE_SINGLE:
    #         # Run regular egomotion inference loop.
    #         input_image_seq = []
    #         input_seg_seq = []
    #         current_sequence_dir = None
    #         current_output_handle = None
    #         for i in range(len(im_files)):
    #             sequence_dir = os.path.dirname(im_files[i])
    #             if sequence_dir != current_sequence_dir:
    #                 # Assume start of a new sequence, since this image lies in a
    #                 # different directory than the previous ones.
    #                 # Clear egomotion input buffer.
    #                 output_filepath = os.path.join(output_dirs[i], 'egomotion.txt')
    #                 if current_output_handle is not None:
    #                     current_output_handle.close()
    #                 current_sequence_dir = sequence_dir
    #                 logging.info('Writing egomotion sequence to %s.', output_filepath)
    #                 current_output_handle = gfile.Open(output_filepath, 'w')
    #                 input_image_seq = []
    #             im = util.load_image(im_files[i], resize=(img_width, img_height))
    #             input_image_seq.append(im)
    #             if use_masks:
    #                 im_seg_path = im_files[i].replace('.%s' % file_extension,
    #                                                   '-seg.%s' % file_extension)
    #                 if not gfile.Exists(im_seg_path):
    #                     raise ValueError('No segmentation mask %s has been found for '
    #                                      'image %s. If none are available, disable '
    #                                      'use_masks.' % (im_seg_path, im_files[i]))
    #                 input_seg_seq.append(util.load_image(im_seg_path,
    #                                                      resize=(img_width, img_height),
    #                                                      interpolation='nn'))
    #
    #             if len(input_image_seq) < seq_length:  # Buffer not filled yet.
    #                 continue
    #             if len(input_image_seq) > seq_length:  # Remove oldest entry.
    #                 del input_image_seq[0]
    #                 if use_masks:
    #                     del input_seg_seq[0]
    #
    #             input_image_stack = np.concatenate(input_image_seq, axis=2)
    #             input_image_stack = np.expand_dims(input_image_stack, axis=0)
    #             if use_masks:
    #                 input_image_stack = mask_image_stack(input_image_stack,
    #                                                      input_seg_seq)
    #             est_egomotion = np.squeeze(inference_model.inference_egomotion(
    #                 input_image_stack, sess))
    #             egomotion_str = []
    #             for j in range(seq_length - 1):
    #                 egomotion_str.append(','.join([str(d) for d in est_egomotion[j]]))
    #             current_output_handle.write(
    #                 str(i) + ' ' + ' '.join(egomotion_str) + '\n')
    #         if current_output_handle is not None:
    #             current_output_handle.close()
    #     elif inference_mode == INFERENCE_MODE_TRIPLETS:
    #         written_before = []
    #         for i in range(len(im_files)):
    #             im = util.load_image(im_files[i], resize=(img_width * 3, img_height))
    #             input_image_stack = np.concatenate(
    #                 [im[:, :img_width], im[:, img_width:img_width * 2],
    #                  im[:, img_width * 2:]], axis=2)
    #             input_image_stack = np.expand_dims(input_image_stack, axis=0)
    #             if use_masks:
    #                 im_seg_path = im_files[i].replace('.%s' % file_extension,
    #                                                   '-seg.%s' % file_extension)
    #                 if not gfile.Exists(im_seg_path):
    #                     raise ValueError('No segmentation mask %s has been found for '
    #                                      'image %s. If none are available, disable '
    #                                      'use_masks.' % (im_seg_path, im_files[i]))
    #                 seg = util.load_image(im_seg_path,
    #                                       resize=(img_width * 3, img_height),
    #                                       interpolation='nn')
    #                 input_seg_seq = [seg[:, :img_width], seg[:, img_width:img_width * 2],
    #                                  seg[:, img_width * 2:]]
    #                 input_image_stack = mask_image_stack(input_image_stack,
    #                                                      input_seg_seq)
    #             est_egomotion = inference_model.inference_egomotion(
    #                 input_image_stack, sess)
    #             est_egomotion = np.squeeze(est_egomotion)
    #             egomotion_1_2 = ','.join([str(d) for d in est_egomotion[0]])
    #             egomotion_2_3 = ','.join([str(d) for d in est_egomotion[1]])
    #
    #             output_filepath = os.path.join(output_dirs[i], 'egomotion.txt')
    #             file_mode = 'w' if output_filepath not in written_before else 'a'
    #             with gfile.Open(output_filepath, file_mode) as current_output_handle:
    #                 current_output_handle.write(str(i) + ' ' + egomotion_1_2 + ' ' +
    #                                             egomotion_2_3 + '\n')
    #             written_before.append(output_filepath)
    #     logging.info('Done.')
