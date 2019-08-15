import os
import sys
import numpy as np
import tensorflow as tf

# Mask RCNN modules
from struct2depth.Mask_RCNN.mrcnn import utils
import struct2depth.Mask_RCNN.mrcnn.model as modellib
from struct2depth.Mask_RCNN.mrcnn import visualize
from struct2depth.Mask_RCNN.samples.coco import coco

# Directory to save logs and trained model
ROOT_DIR = os.path.abspath("./Mask_RCNN")
MODEL_DIR = os.path.join(ROOT_DIR, "mrcnn_logs")

# Local path to trained weights file
COCO_MODEL_PATH = "/usr/local/lib/isaac/apps/carter_sim_struct2depth/struct2depth/Mask_RCNN/mask_rcnn_coco.h5"
# os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")


class MaskGenerator:
    def __init__(self):
        # Fix GPU memory issue
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        session = tf.InteractiveSession(config=config)
        # self.graph = tf.Graph()

        # Download COCO trained weights from Releases if needed
        if not os.path.exists(COCO_MODEL_PATH):
            utils.download_trained_weights(COCO_MODEL_PATH)

        class InferenceConfig(coco.CocoConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1

        self.config = InferenceConfig()
        self.config.display()

        # Create model object in inference mode.
        # with self.graph.as_default():
        self.model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=self.config)

            # Load weights trained on MS-COCO
        self.model.load_weights(COCO_MODEL_PATH, by_name=True)
        self.model.keras_model._make_predict_function()

        # COCO Class names
        # Index of the class in the list is its ID. For example, to get ID of
        # the teddy bear class, use: class_names.index('teddy bear')
        self.class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                            'bus', 'train', 'truck', 'boat', 'traffic light',
                            'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                            'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                            'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                            'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                            'kite', 'baseball bat', 'baseball glove', 'skateboard',
                            'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                            'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                            'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                            'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                            'teddy bear', 'hair drier', 'toothbrush']

        self.image = None
        self.results = None
        self.color_code_scale = 15

    # # Load a random image from the images folder
    # file_names = next(os.walk(IMAGE_DIR))[2]
    # image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))

    # Run detection
    def detect(self, image):
        self.image = image
        # with self.graph.as_default():
        self.results = self.model.detect([image], verbose=1)
        self.results = self.results[0]
        return self.results

    # Generate a segmented image
    # Each instance has a different color ID; background is 0
    # Three channels all with same color code
    def generate_seg_img(self, image):
        self.results = self.detect(image)

        # Generate instance masks with all channel values equal to the instance ID
        masks = self.results['masks']
        class_ids = self.results['class_ids']
        seg_img = np.zeros(shape=image.shape, dtype=np.uint8)

        # seg_img = np.copy(masks[:, :, 0]) * 0

        for i in range(0, masks.shape[2]):
            mask = masks[:, :, i]
            class_id = class_ids[i]
            for j in range(0, seg_img.shape[2]):
                seg_img[:, :, j] += np.uint8(mask * class_id)

        # Visualize seg img
        # imgplot = plt.imshow(seg_img)

        return seg_img

    def visualize(self):
        if self.results is not None:
            visualize.display_instances(self.image, self.results['rois'], self.results['masks'],
                                        self.results['class_ids'], self.class_names, self.results['scores'])
