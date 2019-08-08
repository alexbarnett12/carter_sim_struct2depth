from __future__ import absolute_import, division, print_function
import os
import sys
import numpy as np

# Root directory of the Isaac
ROOT_DIR = os.path.abspath("/mnt/isaac_2019_2")
sys.path.append(ROOT_DIR)

from engine.pyalice import *


class SegmentationEncoder(Codelet):

    # Initialization run once at the beginning of the program
    def start(self):

        # Input and output messages for the Codelet
        self.rx = self.isaac_proto_rx("SegmentationCameraProto", "segmentation")

        # Tick every time we receive a differential base state
        self.tick_on_message(self.rx)

    def tick(self):

        # seg_image_proto = self.rx.get_proto().labelImage
        # element_type = seg_image_proto.elementType
        # rows = seg_image_proto.rows
        # cols = seg_image_proto.cols
        # channels = seg_image_proto.channels
        # image_buffer_id = seg_image_proto.dataBufferIndex
        #
        labels = self.rx.get_proto().labels
        print(labels)

        seg_instance_proto = self.rx.get_proto().instanceImage
        instance_element_type = seg_instance_proto.elementType
        instance_rows = seg_instance_proto.rows
        instance_cols = seg_instance_proto.cols
        instance_channels = seg_instance_proto.channels
        instance_buffer_id = seg_instance_proto.dataBufferIndex
        # print("Element type: {}".format(element_type))
        # print("Rows: {}".format(rows))
        # print("Cols: {}".format(cols))
        # print("Channels: {}".format(channels))
        # print("Image buffer id: {}\n".format(image_buffer_id))


        # image_buffer = self.rx.get_buffer_content(image_buffer_id)
        instance_buffer = self.rx.get_buffer_content(instance_buffer_id)

        # Transform into image
        # image = np.frombuffer(image_buffer, dtype=np.uint8)
        # image = np.reshape(image, (rows, cols, channels))
        instance = np.frombuffer(instance_buffer, dtype=np.uint16)
        instance = np.reshape(instance, (instance_rows, instance_cols, instance_channels))
        # print(instance)
