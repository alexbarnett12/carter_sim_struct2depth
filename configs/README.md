## training_parameters.json
data_dir: Directory with training data. Images, segmentation masks, and camera intrinsics must all be saved in the same directory with corresponding numbers in their names. 
Segmentation masks must be saved with '-fseg' and intrinics with '_cam',e.g.: 1.png, 1-fseg.png, 1_cam.csv.
using_saved_images: Signify if you will be training with Isaac Sim or off of data in a local directory. If training with the sim, you 
do not need to put anything in data_dir.
pretrained_ckpt: 
imagenet_ckpt:
checkpoint_dir:
cuda_device: 
file_extension: 
batch_size: 
save_ckpt_every: 
learning_rate: 
beta1": 
reconstr_weight":
ssim_weight: 
smooth_weight":
icp_weight:
size_constraint_weight: 
img_height:
img_width: 
seq_length":
architecture": 
imagenet_norm:
weight_reg: 
exhaustive_mode: 
random_scale_crop: 
flipping_mode: 
train_steps: 
summary_freq: 
depth_upsampling: 
depth_normalization:
compute_minimum_loss: 
use_skip: 
equal_weighting:
joint_encoder:
handle_motion:
master: 
shuffle: 
max_ckpts_to_keep: 

## optimize_parameters.json

## inference_parameters.json

## carter.config.json

