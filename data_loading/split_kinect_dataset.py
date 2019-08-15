import os
import glob
from random import shuffle
import shutil

DATA_DIR = '/mnt/train_data/warehouse_8_15'
OUTPUT_DIR = '/mnt/test_data/warehouse_8_15'
DATA_FILE = 'train'
FILE_EXT = 'png'
TEST_PERCENTAGE = 0.1


def main():

    paths = [file for file in glob.glob(DATA_DIR + '/*.png') if 'fseg' not in file]

    # Shuffle list
    shuffle(paths)

    # Extract a percentage of all samples for test set
    for i in range(int(len(paths) * 0.1)):

        # Extract files
        image_file = paths[i]

        image_num = image_file.split('/')[4].split('.')[0]

        x = DATA_DIR + '/' + image_num + '_cam.csv'
        y = DATA_DIR + '/' + image_num + '-fseg.png'

        # Move files to new directory
        shutil.move(image_file, OUTPUT_DIR + '/' + image_num + '.png')
        shutil.move(DATA_DIR + '/' + image_num + '-fseg.png', OUTPUT_DIR + '/' + image_num + '-fseg.png')
        shutil.move(DATA_DIR + '/' + image_num + '_cam.csv', OUTPUT_DIR + '/' + image_num + '_cam.csv')


if __name__ == '__main__':
  main()