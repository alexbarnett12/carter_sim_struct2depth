
""" Takes 3 images in a sequence and creates one big image with all 3 aligned. """

import numpy as np
import cv2
import tensorflow as tf

# Segmentation mask generation
from .gen_masks_kitti import MaskGenerator
from .alignment import align


SEQ_LENGTH = 3
WIDTH = 416
HEIGHT = 128

class ImageProcessor:
    def __init__(self):

        self.mask_generator = MaskGenerator()


    def create_triplet(self, images):

        ORIGINAL_HEIGHT, ORIGINAL_WIDTH, _, = images[0].shape
        zoom_x = WIDTH / ORIGINAL_WIDTH
        zoom_y = HEIGHT / ORIGINAL_HEIGHT


        big_img = np.zeros(shape=(HEIGHT, WIDTH*SEQ_LENGTH, 3))
        wct = 0


        for img in images:

            img = cv2.resize(img, (WIDTH, HEIGHT))
            big_img[:, wct * WIDTH:(wct + 1) * WIDTH] = img
            wct += 1

        return big_img


    def create_mask(self, img):
        img = cv2.resize(img, (WIDTH, HEIGHT))
        # mask_generator.visualize()
        return self.mask_generator.generate_seg_img(img)


    def create_mask_triplet(self, images):

        # Define list of seg_mask images
        seg_list = []

        for img in images:

            seg_list.append(self.create_mask(img))

        # Align seg_masks
        seg_list[0], seg_list[1], seg_list[2] = align(seg_list[0], seg_list[1], seg_list[2])
        big_seg_img = np.zeros(shape=(HEIGHT, WIDTH*SEQ_LENGTH, 3))

        # Create seg_mask triplet
        for k in range(0, 3):
            big_seg_img[:, k * WIDTH:(k + 1) * WIDTH] = seg_list[k]

        return big_seg_img



    def process_image(self, images):
        return self.create_triplet(images), self.create_mask_triplet(images)