from __future__ import division
import sys
import cv2
import os
import numpy as np
import argparse
from depth_evaluation_utils import *

# Command line flags
parser = argparse.ArgumentParser()
parser.add_argument("--kitti_dir", type=str, help='Path to the KITTI dataset directory')
parser.add_argument("--pred_files", type=str, help="Path to prediction files, separated by commas")
parser.add_argument("--test_file_list", type=str, default='./data/kitti/test_files_eigen.txt',
                    help="Path to the list of test files")
parser.add_argument('--min_depth', type=float, default=1e-3, help="Threshold for minimum depth")
parser.add_argument('--max_depth', type=float, default=80, help="Threshold for maximum depth")
args = parser.parse_args()


def main():
    # Split into separate files
    pred_depths_files = args.pred_files.replace(" ", "").split(',')

    # Load predicted depths
    pred_depths = []
    for i in range(len(pred_depths_files)):
        pred_depths.append(np.load(pred_depths_files[i]))

    # Load standard test files
    test_files = read_text_lines(args.test_file_list)

    # Ground truth and images
    gt_files, gt_calib, im_sizes, im_files, cams = \
        read_file_data(test_files, args.kitti_dir)

    # Resize predicted depths and generate GT depth maps
    num_test = len(im_files)
    gt_depths = []
    for i in range(len(pred_depths)):
        pred_depths_resized = []
        for t_id in range(num_test):
            camera_id = cams[t_id]  # 2 is left, 3 is right
            pred_depths_resized.append(
                cv2.resize(pred_depths[i][t_id],
                           (im_sizes[t_id][1], im_sizes[t_id][0]),
                           interpolation=cv2.INTER_LINEAR))
            depth = generate_depth_map(gt_calib[t_id],
                                       gt_files[t_id],
                                       im_sizes[t_id],
                                       camera_id,
                                       False,
                                       True)
            gt_depths.append(depth.astype(np.float32))
        pred_depths[i] = pred_depths_resized

    # Define Comparison metrics
    rms = np.zeros(len(pred_depths), num_test, np.float32)
    log_rms = np.zeros(len(pred_depths), num_test, np.float32)
    abs_rel = np.zeros(len(pred_depths), num_test, np.float32)
    sq_rel = np.zeros(len(pred_depths), num_test, np.float32)
    d1_all = np.zeros(len(pred_depths), num_test, np.float32)
    a1 = np.zeros(len(pred_depths), num_test, np.float32)
    a2 = np.zeros(len(pred_depths), num_test, np.float32)
    a3 = np.zeros(len(pred_depths), num_test, np.float32)

    print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('abs_rel',
                                                                                  'sq_rel',
                                                                                  'rms',
                                                                                  'log_rms',
                                                                                  'd1_all',
                                                                                  'a1',
                                                                                  'a2',
                                                                                  'a3'))
    for i in range(len(pred_depths)):
        for j in range(num_test):
            gt_depth = gt_depths[j]
            pred_depth = np.copy(pred_depths[i][j])

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
            abs_rel[i][j], sq_rel[i][j], rms[i][j], log_rms[i][j], a1[i][j], a2[i][j], a3[i][j] = \
                compute_errors(gt_depth[mask], pred_depth[mask])

        print("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(abs_rel[i].mean(),
                                                                                                      sq_rel[i].mean(),
                                                                                                      rms[i].mean(),
                                                                                                      log_rms[i].mean(),
                                                                                                      d1_all[i].mean(),
                                                                                                      a1[i].mean(),
                                                                                                      a2[i].mean(),
                                                                                                      a3[i].mean()))


main()
