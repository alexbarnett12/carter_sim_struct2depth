from __future__ import division
import sys
import cv2
import os
import numpy as np
import glob
import argparse
from depth_evaluation_utils import *

# Command line flags
parser = argparse.ArgumentParser()
# parser.add_argument("--test_dir", type=str, help='Path to the test image directory')
parser.add_argument("--pred_dir", type=str, default='/mnt/results_s2d/warehouse_8_9_pretrained_s2d', help="Path to prediction directory")
parser.add_argument("--gt_dir", type=str, default='/mnt/test_images/warehouse_8_9/depth',
                    help="path to the ground truth directory")
parser.add_argument('--min_depth', type=float, default=1e-3, help="Threshold for minimum depth")
parser.add_argument('--max_depth', type=float, default=51, help="Threshold for maximum depth")
args = parser.parse_args()

HEIGHT = 128
WIDTH = 416


def main():
    # Load files
    pred_depths_files = sorted(glob.glob(args.pred_dir + "/*.npy"))
    test_files = sorted(glob.glob(args.gt_dir + "/*"))

    # Load predicted depths
    pred_depths = []
    for i in range(len(pred_depths_files)):
        pred_depths.append(np.load(pred_depths_files[i]))

    # Load ground truth
    gt_depths = []
    for i in range(len(test_files)):
        gt_depths.append(cv2.imread(test_files[i]))

    # Define Comparison metrics
    rms = np.zeros(len(pred_depths), np.float32)
    log_rms = np.zeros(len(pred_depths), np.float32)
    abs_rel = np.zeros(len(pred_depths), np.float32)
    sq_rel = np.zeros(len(pred_depths), np.float32)
    d1_all = np.zeros(len(pred_depths), np.float32)
    a1 = np.zeros(len(pred_depths), np.float32)
    a2 = np.zeros(len(pred_depths), np.float32)
    a3 = np.zeros(len(pred_depths), np.float32)

    print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('abs_rel',
                                                                                  'sq_rel',
                                                                                  'rms',
                                                                                  'log_rms',
                                                                                  'd1_all',
                                                                                  'a1',
                                                                                  'a2',
                                                                                  'a3'))

    for i in range(len(pred_depths)):


        gt_depth = gt_depths[i]
        pred_depth = np.copy(pred_depths[i])

        gt_depth = cv2.resize(gt_depth, dsize=(WIDTH, HEIGHT))[:, :, 0]
        pred_depth = np.squeeze(pred_depth)

        mask = np.logical_and(gt_depth > args.min_depth,
                              gt_depth < args.max_depth)
        # crop used by Garg ECCV16 to reprocude Eigen NIPS14 results
        # if used on gt_size 370x1224 produces a crop of [-218, -3, 44, 1180]
        gt_height, gt_width = gt_depth.shape
        crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                         0.03594771 * gt_width, 0.96405229 * gt_width]).astype(np.int32)

        crop_mask = np.zeros(mask.shape)
        crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
        mask = np.logical_and(mask, crop_mask)

        # Scale matching
        scalor = np.median(gt_depth[mask]) / np.median(pred_depth[mask])
        pred_depth[mask] *= scalor

        # Thresholding
        pred_depth[pred_depth < args.min_depth] = args.min_depth
        pred_depth[pred_depth > args.max_depth] = args.max_depth

        # Compute errors
        abs_rel[i], sq_rel[i], rms[i], log_rms[i], a1[i], a2[i], a3[i] = \
            compute_errors(gt_depth[mask], pred_depth[mask])

    print("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(abs_rel[i].mean(),
                                                                                                  sq_rel[i].mean(),
                                                                                                  rms[i].mean(),
                                                                                                  log_rms[i].mean(),
                                                                                                  d1_all[i].mean(),
                                                                                                  a1[i].mean(),
                                                                                                  a2[i].mean(),
                                                                                                  a3[i].mean()))


#     for i in range(len(pred_depths)):
#
#         gt_depth = gt_depths[i]
#         pred_depth = pred_depths[i]
#
#         # Reshape GT
#         gt_depth = cv2.resize(gt_depth, dsize=(WIDTH, HEIGHT))[:, :, 0]
#         pred_depth = np.squeeze(pred_depth)
#
#         # Standardize to between 0 and 1
#         pred_depth = pred_depth / np.amax(pred_depth)
#         gt_depth = gt_depth / np.amax(gt_depth)
#
#         # Create ground truth mask
#         gt_mask = (gt_depth > 0).astype(np.uint8)
#
#         # Apply mask to predicted depth to only compare accurate data
#         pred_depth_mask = pred_depth * gt_mask
#
# #         mask = np.logical_and(gt_depth > args.min_depth,
# #                               gt_depth < args.max_depth)
# #
# #         # crop used by Garg ECCV16 to reprocude Eigen NIPS14 results
# #         # if used on gt_size 370x1224 produces a crop of [-218, -3, 44, 1180]
# #         gt_height, gt_width = gt_depth.shape
# #         crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
# #                          0.03594771 * gt_width, 0.96405229 * gt_width]).astype(np.int32)
# #
# #         crop_mask = np.zeros(mask.shape)
# #         crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
# #         mask = np.logical_and(mask, crop_mask)
# #
# #         # Scale matching
# #         scalor = np.median(gt_depth[mask]) / np.median(pred_depth[mask])
# #         pred_depth[mask] *= scalor
# #
# #         # Thresholding
# #         pred_depth[pred_depth < args.min_depth] = args.min_depth
# #         pred_depth[pred_depth > args.max_depth] = args.max_depth
# #
# #         # Compute errors
#         abs_rel[i], sq_rel[i], rms[i], log_rms[i], a1[i], a2[i], a3[i] = \
#             compute_errors(gt_depth, pred_depth)
# #
#     print("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(abs_rel[i].mean(),
#                                                                                                   sq_rel[i].mean(),
#                                                                                                   rms[i].mean(),
#                                                                                                   log_rms[i].mean(),
#                                                                                                   d1_all[i].mean(),
#                                                                                                   a1[i].mean(),
#                                                                                                   a2[i].mean(),
#                                                                                                   a3[i].mean()))
# #
#
main()
