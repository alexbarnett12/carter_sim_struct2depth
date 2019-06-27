import os
import tensorflow as tf
from random import shuffle
import shutil
gfile = tf.gfile

DATA_DIR = 'kitti_processed_data'
OUTPUT_DIR = 'kitti_test_set'
DATA_FILE = 'train'
FILE_EXT = 'png'


def main():

    # Create lists of input files
    with gfile.Open(os.path.join(DATA_DIR, '%s.txt' % DATA_FILE), 'r') as f:
        frames = f.readlines()
        frames = [k.rstrip() for k in frames]
    subfolders = [x.split(' ')[0] for x in frames]
    frame_ids = [x.split(' ')[1] for x in frames]

    # Camera images
    image_file_list = [
        os.path.join(DATA_DIR, subfolders[i], frame_ids[i] + '.' + FILE_EXT)
        for i in range(len(frames))]

    # Segmentation masks
    segment_file_list = [
        os.path.join(DATA_DIR, subfolders[i], frame_ids[i] + '-fseg.' +
                     FILE_EXT)
        for i in range(len(frames))]

    # Camera intrinsics
    cam_file_list = [
        os.path.join(DATA_DIR, subfolders[i], frame_ids[i] + '_cam.txt')
        for i in range(len(frames))]

    # Shuffle list
    shuffle(image_file_list)



    # Test set file lists
    image_test = []
    seg_test = []
    cam_test = []

    # Extract 1 in every 10 images; 10% test set
    i = 0
    while i < len(image_file_list):

        # Extract files
        image_file = image_file_list.pop(i)

        seqname = image_file.split('/')[1]
        imgnum_w_ext = image_file.split('/')[2]
        imgnum = imgnum_w_ext.split('.')[0]

        # Move files to new directory
        shutil.move(image_file, 'kitti_test_set')
        shutil.move(DATA_DIR + '/' + seqname + '/' + imgnum + '-fseg.png', 'kitti_test_set')
        shutil.move(DATA_DIR + '/' + seqname + '/' + imgnum + '_cam.txt', 'kitti_test_set')

        i += 10


if __name__ == '__main__':
  main()