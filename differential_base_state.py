from __future__ import absolute_import, division, print_function
import os
import sys
import numpy as np

# Root directory of the Isaac
ROOT_DIR = os.path.abspath("/data/repositories/isaac")
sys.path.append(ROOT_DIR)

from engine.pyalice import *


class DifferentialBaseState(Codelet):

    def start(self):
        # This part will be run once in the beginning of the program
        print("Running initialization")

        # Input and output messages for the Codelet
        self.rx = self.isaac_proto_rx("DifferentialBaseStateProto", "differential_base_state")

        # Tick every time we receive a differential base state
        self.tick_on_message(self.rx)

    def tick(self):
        # Extract current linear velocity from rx
        speed = self.rx.get_proto().linearSpeed
        angular_speed = self.rx.get_proto().angularSpeed

        if speed <= 1e-3:
            speed = 0
        if angular_speed <= 1e-3:
            angular_speed = 0

        # Combine into a string format
        combined_speed = "{}, {}".format(speed, angular_speed)

        # Write to a file. Overwritten for each new speed change
        f = open('/data/repositories/isaac/apps/carter_sim_struct2depth/differential_base_speed/speed.csv', 'w')
        f.write(combined_speed)
        f.close()
