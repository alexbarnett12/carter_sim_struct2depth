import os
import sys

# Isaac SDK imports
ROOT_DIR = os.path.abspath("/mnt/isaac_2019_2/")  # Root directory of the Isaac
sys.path.append(ROOT_DIR)
from engine.pyalice import *
import packages.ml
from differential_base_state import DifferentialBaseState

def create_isaac_app(filename):
    isaac_app = Application(app_filename=filename)

    isaac_app.register({"differential_base_state": DifferentialBaseState})

    return isaac_app


def start_isaac_app(isaac_app):
    isaac_app.start()

def create_sample_bridge(isaac_app):
    node = isaac_app.find_node_by_name("CarterTrainingSamples")
    return packages.ml.SampleAccumulator(node)