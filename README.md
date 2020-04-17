# carter_sim_struct2depth
This is an application that deploys a virtual autonomous robot using NVIDIA Isaac SDK and the Unreal Engine to 
train a neural network that predicts monocular depth, egomotion, and object motion from image sequences. The goal is to investigate how well monocular depth can be learned from photo-realistic virtual data and how well the model will generalize to the real world. The algorithm also incorporates online refinement to train in real-time and improve accuracy in unknown environments. Ideally, the model can be trained solely on virtual data and then online refinement would be used to adjust the model in real-time.

![Models trained on simulation and real data](inference.gif)

## Struct2Depth   
Struct2depth is a state-of-the-art unsupervised monocular depth network created by Google. The original work can be 
found here: https://sites.google.com/view/struct2depth. The network takes a time sequence of three images (taken with motion between them) and learns depth and egomotion by reprojecting the middle image in the sequence from the two outer images and calculating photometric error as a loss function. The model can also predict individual object egomotion in the image sequence when trained along with prior segmentation masks of moving objects. 

This project uses the same CNN model but the input pipeline has been updated to use the more efficient TensorFlow Dataset API
over the deprecated feed dict pipeline. 

## Prerequisites
First make sure that you have downloaded and followed the instructions to install NVIDIA Isaac SDK 2019.2 and Isaac Sim 1.2 at these two links: 

**Isaac SDK:** https://docs.nvidia.com/isaac/isaac/doc/setup.html

**Isaac Sim:** https://docs.nvidia.com/isaac/isaac_sim/setup.html

Once you have installed NVIDIA Isaac and followed their basic tutorials, you can run all the programs in this repo the same way with Bazel. 

## Training
The model can be trained either straight from the simulation or with saved data. All training parameters can be 
modified in `configs/training_parameters.json`. To train on saved data, set `using_saved_images = true` and 
specify the image, segmentation mask, and camera intrinsics directories. It is essential that all the data is sorted, 
since the model requires corresponding seg masks and intrinsics to correctly train. All Isaac Sim training parameters
can be modified in `configs/isaac_parameters.json`. Do not change `num_samples = 1`, since increasing the number will
cause image sequences to not be in consecutive time. 

If training on saved images, the data is expected to be formatted into three separate folders for image triplets,
segmentation masks, and intrinsics. Refer to `save_image_triplets.py` for an example. 

To train, run this in the top directory of this repo:

`bazel run train`

Then, in a separate terminal in the Isaac Sim top-level directory, run:

`./Engine/Binaries/Linux/UE4Editor IsaacSimProject <MAP> -vulkan -isaac_sim_config_json="<ABSOLUTE_PATH_TO_ISAAC_SDK>/apps/carter_sim_struct2depth/bridge_config/carter_full.json"`

Refer to the "Isaac Sim Environments" section for configuring the MAP flag.

## Data Generation
Simulation data can be generated and saved to disk using `save_image_triplets.py` and `save_images.py`. The first script
saves images in sequences of 3 so they can be directly fed into the model. The time delay between each image in a sequence 
can be modified in the script. The time delays ensures that there is a high disparity between images so the model 
can predict egomotion. The model has been trained well with a 0.4 second time delay. To run these scripts:

`bazel run save_image_triplets`

`bazel run save_images`

## Isaac Sim Environments
There are three generic maps that come with Isaac Sim: warehouse, office, and hospital. Each has static and dynamic 
objects and very clunky roaming humans. To change the map Carter drives in, do three things:

1. Edit `apps/carter_sim.app.json` by changing the config and graph files corresponding to maps:

    Warehouse: `apps/assets/maps/carter_warehouse_p.config.json`, `apps/assets/maps/carter_warehouse_p.graph.json`

    Office: `apps/assets/maps/carter_office.config.json`, `apps/assets/maps/carter_office.graph.json`

    Hospital: `apps/assets/maps/hospital.config.json`, `apps/assets/maps/hospital.graph.json`
    
2. Edit the robot starting pose near the top of `bridge_config/carter_full_config.json`:

    Warehouse: [1, 0, 0, 0, -11.01, 63.47, 0.92]
    
    Office: [1, 0, 0, 0, 1.50, -43.10, 0.5]
    
    Hospital: [1, 0, 0, 0, 0.0, -11.0, 0.9]
    
3. When launching Isaac Sim, change the <MAP> flag referenced in the Training section to the correct map name:

    Warehouse: CarterWarehouse_P
    
    Office: Carter_Office
    
    Hospital: Hospital
    
    For example:
    
    `./Engine/Binaries/Linux/UE4Editor IsaacSimProject Carter_Warehouse_P -vulkan -isaac_sim_config_json="<ABSOLUTE_PATH_TO_ISAAC_SDK>/apps/carter_sim_struct2depth/bridge_config/carter_full.json"`

Custom maps can also be created, but none have been created for this project. In the future, custom maps would be very
beneficial since they could be much more photorealistic and have a lot more scene variety.
    
## Inference
Inference can be run on saved images or directly from the simulation. The script currently uses TensorFlow flags as 
parameters but will be updated to user a config JSON like the training script. To run inference on the simulation, use:

`bazel run inference`

To run inference on saved images, use:

`python3 struct2depth/inference.py --input_dir /path/to/input --output_dir /path/to/output --model_ckpt /path/to/ckpt`

Alternatively, the script can be run on multiple models consecutively to allow for easy comparison between models at 
different epochs. To do so, edit the `EPOCHS` global array and `TRAINING_BATCH_SIZE` parameter in `struct2depth/inference.py` with the epochs you would like 
to analyze and the batch size used for training. Then run:

`python3 struct2depth/inference.py --multiple_ckpts True --ckpt_dir /path/to/ckpts --data_dir /path/to/training/data 
--input_dir /path/to/input --output_dir /path/to/output --model_ckpt /path/to/ckpt`

`data_dir` is your original training data, so that the script knows the dataset size and can find the model checkpoint
closest to the desired epoch.


Images just need to be saved into one folder for input_dir. 

## Online Refinement
Online refinement can currently only be used on saved images. To do so, save image triplets, seg masks, and intrinsics in a single
directory. All names should be paired, with seg masks ending in "-fseg.png" and intrinsics ending in "_cam.txt." Edit 
`configs/optimize_parameters.json` and modify these parameters:

`output_dir`: Directory to save inference results.

`data_dir`: Input directory with images. Names must be paired as numbers w/ zero-padding.

`model_ckpt`: Model checkpoint used for inference.

`checkpoint_dir`: Directory to save refined models.

Finally, run:

`bazel run optimize`


## Running Isaac Sim on a Server
It is possible to use a remote server to train with Isaac Sim. Since Isaac Sim has specific CUDA and NVIDIA driver 
requirements, it is easiest to run Isaac Sim locally and run the Carter application on the server. All the required ports
are specified in corresponding config files, such as `configs/carter_server.config.json` and `bridge_config/carter_full_config_server.json`.
To run Carter on a server, choose a set of server config files by running `bazel run train_server_X`, where X represents the number in the config file, such as `carter_server_2.config.json`. 
Make sure to reference the correct bridge_config file when starting Isaac Sim. For example:

`./Engine/Binaries/Linux/UE4Editor IsaacSimProject <MAP> -vulkan -isaac_sim_config_json="<ABSOLUTE_PATH_TO_ISAAC_SDK>/apps/carter_sim_struct2depth/bridge_config/carter_full_server.json"`

You can also edit the ports manually. Make sure that the publisher/subscriber ports in `configs/carter.config.json` match with 
corresponding ports in `bridge_config/carter_full_config.json`


