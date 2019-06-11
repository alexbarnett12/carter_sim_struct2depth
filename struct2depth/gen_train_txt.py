import os
from absl import app

DATA_DIR = 'kitti_processed_data'
OUTPUT_DIR = 'kitti_processed_data'
OUTPUT_NAME = 'train'
FILE_EXT = 'png'


def generate_train_txt():
    with open(OUTPUT_DIR + '/' + OUTPUT_NAME + '.txt', 'w') as f:
        for dirpath, dirnames, files in os.walk('./' + DATA_DIR):
            # Skip root
            if dirpath != './' + DATA_DIR:
                # Split directory path
                subdir = dirpath.split('/')[2]
                for file_name in files:
                    frame_id = file_name.split('.')[0]
                    file_ext = file_name.split('.')[1]

                    # Only process specific files to avoid overlap
                    if file_ext == FILE_EXT and frame_id.find("fseg") == -1:
                        text = subdir + ' ' + frame_id + '\n'
                        f.write(text)


def main():
    generate_train_txt()


if __name__ == '__main__':
  main()