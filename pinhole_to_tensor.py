from __future__ import absolute_import, division, print_function
import os
import sys
import numpy as np

# Root directory of the Isaac
ROOT_DIR = os.path.abspath("/usr/local/lib/isaac")
sys.path.append(ROOT_DIR)
import struct

from engine.pyalice import *
# import packages.ml
# import apps.pinhole_to_tensor

class PinholeToTensor(Codelet):
    def start(self):
        # This part will be run once in the beginning of the program
        print("Running initialization")

        # Input and output messages for the Codelet
        self.rx = self.isaac_proto_rx("ColorCameraProto", "rgb_image")
        self.tx = self.isaac_proto_tx("TensorListProto", "tensor_list")
        self.tx_inner = self.isaac_proto_tx("TensorProto", "tensor")

        # Tick every time we receive an image
        self.tick_on_message(self.rx)

    def tick(self):
        # print("Ticking")

        # Extract pinhole proto from rx
        pinhole = self.rx.get_proto().pinhole

        # Extract proto attributes
        focal_x = pinhole.focal.x
        focal_y = pinhole.focal.y
        center_x = pinhole.center.x
        center_y = pinhole.center.y

        # Create 1D camera matrix
        camera_mat = [int(focal_x), 0, int(center_x),
                      0, int(focal_y), int(center_y),
                      0, 0, 1]
        print(camera_mat)

        # Initialize TensorProto and TensorListProto for the camera matrix
        tensor_list = self.tx.init_proto()
        tensor = self.tx_inner.init_proto()

        # Set element type
        tensor.elementType = 'float32'

        # Set tensor dimensions
        size = tensor.init('sizes', 2) # = [1, 9]
        size[0] = 9
        size[1] = 1

        # Add data to TensorProto
        tensor.data = np.getbuffer(camera_mat)

        # Create TensorList
        tensor_list = tensor_list.init('tensors', 1)

        # Add tensor to TensorListProto
        tensor_list[0] = tensor

        # Publish TensorListProto
        self.tx.publish()
