# Record camera trajectory and generate frames
import numpy as np
import os
from PIL import Image
import torch
import time

import omnigibson as og
from omnigibson.utils.asset_utils import download_key
from omnigibson.macros import gm

# Set headless mode and download key
gm.HEADLESS = True
download_key()

# Don't use GPU dynamics and use flatcache for performance boost
gm.USE_GPU_DYNAMICS = False
gm.ENABLE_FLATCACHE = True

# Choose robot to create
scene_cfg = {"type": "InteractiveTraversableScene", "scene_model": "Rs_int"}

# Add the robot we want to load
robot0_cfg = {
  "type": "Fetch",
  "obs_modalities": ["rgb"],
  "action_type": "continuous",
  "action_normalize": True,
}

# Compile config
cfg = {
    "scene": scene_cfg,
    "robots": [robot0_cfg],
    "env": {"action_timestep": 1 / 10., "physics_timestep": 1 / 120.},
    "render": {"viewer_width": 1024, "viewer_height": 1024}
}

# Create output directories if they don't exist
output_dir = "trajectory_dataset"
frames_dir = os.path.join(output_dir, "frames")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(frames_dir, exist_ok=True)

# Create the environment
env = og.Environment(configs=cfg)

# Initialize the camera
og.sim.viewer_camera.add_modality("rgb")

# Function to get camera extrinsic matrix (world to camera transform)
def get_camera_extrinsic_matrix(camera):
    # Get camera position and orientation
    position = camera.get_position()
    orientation_quat = camera.get_orientation()
    
    # Convert quaternion to rotation matrix
    qw, qx, qy, qz = orientation_quat[3], orientation_quat[0], orientation_quat[1], orientation_quat[2]
    
    # Construct rotation matrix from quaternion
    rot_matrix = np.array([
        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
    ])
    
    # Create 4x4 transformation matrix
    extrinsic_matrix = np.eye(4)
    extrinsic_matrix[:3, :3] = rot_matrix
    extrinsic_matrix[:3, 3] = position
    
    return extrinsic_matrix

# Number of frames to capture
num_frames = 10

# List to store camera trajectories
trajectories = []

print(f"Generating {num_frames} frames with camera trajectory tracking...")

# Initial camera position (can be modified to match your desired starting point)
initial_position = np.array([4.46949, -3.97358, 2.21529])
initial_orientation = np.array([0.56829048, 0.09569975, 0.13571846, 0.80589577])
og.sim.viewer_camera.set_position_orientation(position=initial_position, orientation=initial_orientation)

# Generate frames and record camera trajectory
for frame_idx in range(num_frames):
    # Take a random action to change the scene slightly
    action = env.action_space.sample()
    env.step(action=action)
    
    # Optionally move the camera (can be modified to create desired camera motion)
    # For example, moving in a small circle around the initial point
    angle = frame_idx * (2 * np.pi / num_frames)
    radius = 0.2  # Small radius for circular motion
    x_offset = radius * np.cos(angle)
    y_offset = radius * np.sin(angle)
    
    new_position = initial_position + np.array([x_offset, y_offset, 0.0])
    # You could also modify orientation for more complex trajectories
    
    og.sim.viewer_camera.set_position_orientation(position=new_position, orientation=initial_orientation)
    
    # Get camera extrinsic matrix
    extrinsic_matrix = get_camera_extrinsic_matrix(og.sim.viewer_camera)
    trajectories.append(extrinsic_matrix.flatten())
    
    # Capture RGB image
    obs_dict = og.sim.viewer_camera.get_obs()[0]
    rgb_image = obs_dict["rgb"]
    rgb_np = rgb_image.cpu().detach().numpy()
    
    # Make sure we're getting 3-channel RGB data
    if len(rgb_np.shape) == 3 and rgb_np.shape[2] >= 3:
        rgb_np = rgb_np[:, :, :3]
        if rgb_np.dtype == np.float32 or rgb_np.dtype == np.float64:
            rgb_np = (rgb_np * 255).astype(np.uint8)
        
        # Save the frame
        frame_path = os.path.join(frames_dir, f"frame_{frame_idx:04d}.png")
        Image.fromarray(rgb_np).save(frame_path)
        print(f"Saved frame {frame_idx}")
    else:
        print(f"Error: Unexpected RGB image format - shape: {rgb_np.shape}")

# Save trajectory data to file
trajectory_path = os.path.join(output_dir, "traj.txt")
np.savetxt(trajectory_path, np.array(trajectories), fmt="%.18e", delimiter=" ")

print(f"Dataset generation complete!")
print(f"Frames saved to: {frames_dir}")
print(f"Trajectory data saved to: {trajectory_path}")

# Close the environment
env.close()