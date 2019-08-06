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
        self.rx = self.isaac_proto_rx("Odometry2Proto", "differential_state")

        # Tick every time we receive a differential base state
        self.tick_on_message(self.rx)

    def tick(self):

        # Extract speed from differential base
        speed = self.rx.get_proto().speed.x
        angular_speed = self.rx.get_proto().angularSpeed

        try:
            float(speed)
            speed = float(speed)
        except ValueError:
            speed = 0
        try:
            float(angular_speed)
            angular_speed = float(angular_speed)
        except ValueError:
            angular_speed = 0

        # Combine into a string format
        combined_speed = "{}, {}".format(speed, angular_speed)

        # Write to a file. Overwritten for each new speed change
        f = open('/mnt/isaac_2019_2/apps/carter_sim_struct2depth/differential_base_speed/speed.csv', 'w')
        f.write(combined_speed)
        f.close()
