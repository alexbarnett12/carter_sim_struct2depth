from __future__ import absolute_import, division, print_function
import os
import sys

# Root directory of the Isaac
ROOT_DIR = os.path.abspath("/mnt/isaac_2019_2")
sys.path.append(ROOT_DIR)

from engine.pyalice import *


class DifferentialBaseState(Codelet):

    # Initialization run once at the beginning of the program
    def start(self):

        # Input and output messages for the Codelet
        self.rx = self.isaac_proto_rx("DifferentialState", "differential_state")

        # Tick every time we receive a differential base state
        self.tick_on_message(self.rx)

    def tick(self):
        # Extract velocity from rx
        speed_x = self.rx.get_proto().speedX
        speed_y = self.rx.get_proto().speedY
        angular_speed = self.rx.get_proto().angularSpeed

        if speed_x < 1e-3:
            speed_x = 0
        if speed_y < 1e-3:
            speed_y = 0
        if angular_speed <= 1e-3:
            angular_speed = 0

        try:
            float(speed_x)
        except ValueError:
            speed_x = 0
        try:
            float(speed_y)
        except ValueError:
            speed_y = 0
        try:
            float(angular_speed)
        except ValueError:
            angular_speed = 0

        # Combine into a string format
        combined_speed = "{}, {}, {}".format(speed_x, speed_y, angular_speed)

        # Write to a file. Overwritten for each new speed change
        f = open('/mnt/isaac_2019_2/apps/carter_sim_struct2depth/differential_base_speed/speed.csv', 'w')
        f.write(combined_speed)
        f.close()
