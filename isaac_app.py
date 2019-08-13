import os
import sys

# Isaac SDK imports
ROOT_DIR = os.path.abspath("/mnt/isaac_2019_2/")  # Root directory of the Isaac
sys.path.append(ROOT_DIR)
from engine.pyalice import *
import packages.ml
from differential_base_state import DifferentialBaseState

''' Util functions for creating and running an Isaac Application within a Python script.'''

# Creates an Isaac Application and registers any custom PyCodelets defined in the graph.json
def create_isaac_app(filename):
    isaac_app = Application(app_filename=filename)

    isaac_app.register({"differential_base_state": DifferentialBaseState})

    return isaac_app

# Starts the isaac app. Should not be called until ready for runtime.
def start_isaac_app(isaac_app):
    isaac_app.start()

# Creates a bridge to Isaac Sim that allows data samples to be retrieved. Make sure that
# find_node_by_name() has the correct graph node passed as a parameter
def create_sample_bridge(isaac_app):
    node = isaac_app.find_node_by_name("CarterTrainingSamples")
    return packages.ml.SampleAccumulator(node)