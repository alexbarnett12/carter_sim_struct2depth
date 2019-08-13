from struct2depth import inference
import os
import glob
import numpy as np

INPUT_DIR = "/mnt/sim_images/sim_images_40_delay"
DATA_DIR = "/mnt/sim_images/sim_images_40_delay"
OUTPUT_DIR = "/mnt/results_inference/saved_images/warehouse/40_delay_pretrained_lr_0002_8_9"
CKPT_DIR = "/mnt/ckpts/saved_images/warehouse/25_delay_7_20"

# Epochs to test
EPOCHS = [1, 2, 3, 4, 5, 10, 25, 50, 100]

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

# Retrieve all ckpt paths
ckpts = glob.glob(CKPT_DIR + "/model-*")

# Extract total training step numbers
training_steps = []
for i in range(len(ckpts)):
    training_steps.append(int(ckpts[i].split('-')[1].split('.')[0]))

# Sort numbers
training_steps = sorted(training_steps)

max_steps = np.amax(training_steps)

# Total size of training dataset
dataset_size = len(glob.glob(DATA_DIR + "/*"))

# Total number of epochs
total_epochs = int(max_steps / dataset_size)

processed_dirs = []
for i in range(len(EPOCHS)):
    if EPOCHS[i] < total_epochs:
        output_dir = os.path.join(OUTPUT_DIR, str(EPOCHS[i]) + "_epochs")

        # Find ckpt dir nearest to desired epoch
        model_ckpt = os.path.join(CKPT_DIR, "model-{}".format(find_nearest(training_steps, EPOCHS[i] * dataset_size)))

        # Check for overlap
        if model_ckpt not in processed_dirs:
            print(model_ckpt)
            processed_dirs.append(model_ckpt)

            # Run inference and save images
            inference.run_inference_experiment(INPUT_DIR, output_dir, model_ckpt)
