import os

DATA_DIR = '/mnt/train_data/warehouse_8_15'
OUTPUT_DIR = '/mnt/train_data/warehouse_8_15'
OUTPUT_NAME = 'train'
FILE_EXT = 'png'


def generate_train_txt_kinect():
    with open(OUTPUT_DIR + '/' + OUTPUT_NAME + '.txt', 'w') as f:
        for dirpath, dirnames, files in os.walk(DATA_DIR):
            # Skip root
            if dirpath != DATA_DIR:
                # Split directory path
                for file_name in files:
                    frame_id = file_name.split('.')[0]
                    file_ext = file_name.split('.')[1]

                    # Only process specific files to avoid overlap
                    if file_ext == FILE_EXT and frame_id.find("fseg") == -1:
                        text = dirpath + ' ' + frame_id + '\n'
                        f.write(text)


def main():
    generate_train_txt_kinect()


if __name__ == '__main__':
  main()