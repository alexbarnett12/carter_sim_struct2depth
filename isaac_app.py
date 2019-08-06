import os
import sys

# Isaac SDK imports
ROOT_DIR = os.path.abspath("/mnt/isaac_2019_2/")  # Root directory of the Isaac
sys.path.append(ROOT_DIR)
from engine.pyalice import *
from differential_base_state import DifferentialBaseState

APP_FILENAME = "/mnt/isaac_2019_2/apps/carter_sim_struct2depth/carter_sim.app.json"

def create_isaac_app():
    isaac_app = Application(app_filename=APP_FILENAME)

    isaac_app.register({"differential_base_state": DifferentialBaseState})

    return isaac_app
#
# class IsaacApplication:
#     def __init__(self):
#         self.isaac_app = Application(app_filename=APP_FILENAME)
#
#         # Register custom Isaac codelets
#         self.isaac_app.register({"differential_base_state": DifferentialBaseState})
#
#     def get_app(self):
#         return self.isaac_app

def start_isaac_app(isaac_app):
    isaac_app.start()
