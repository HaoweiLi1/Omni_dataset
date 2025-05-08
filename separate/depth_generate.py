# Load a scene with a robot in it and collect depth data
import numpy as np
import os
from PIL import Image
import torch

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
  "obs_modalities": ["rgb", "depth"],  # Include depth in observation modalities
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

# Create the environment
env = og.Environment(configs=cfg)

# Update the simulator's viewer camera's pose so it points towards the robot
og.sim.viewer_camera.set_position_orientation(
    position=np.array([1.46949, -3.97358, 2.21529]),
    orientation=np.array([0.56829048, 0.09569975, 0.13571846, 0.80589577]),
)

# Important: Add depth modality to the viewer camera
# This is the key fix to make depth data accessible
og.sim.viewer_camera.add_modality("depth")

# Create output directory if it doesn't exist
os.makedirs("depth_output", exist_ok=True)

# Take a random action and capture a depth image
# Run more steps to make sure the scene is properly initialized
print("Initializing scene...")
for _ in range(10):  # Run 10 steps to ensure scene is fully loaded
    action = env.action_space.sample()
    env.step(action=action)

print("Getting depth observation...")
# Get observations from the viewer camera
obs_dict = og.sim.viewer_camera.get_obs()[0]

# Debug: Print available modalities in the observation dictionary
print(f"Available modalities: {list(obs_dict.keys())}")

if "depth" not in obs_dict:
    print("Depth modality not available. Trying with 'depth_linear'...")
    og.sim.viewer_camera.add_modality("depth_linear")
    # Run a few more steps
    for _ in range(5):
        env.step(env.action_space.sample())
    obs_dict = og.sim.viewer_camera.get_obs()[0]
    print(f"Available modalities after adding depth_linear: {list(obs_dict.keys())}")
    if "depth_linear" in obs_dict:
        depth_obs = obs_dict["depth_linear"]
    else:
        raise ValueError("Neither 'depth' nor 'depth_linear' modalities are available!")
else:
    depth_obs = obs_dict["depth"]

# Convert PyTorch tensor to NumPy array
depth_np = depth_obs.cpu().detach().numpy()
print(f"Depth array shape: {depth_np.shape}")
print(f"Depth array dtype: {depth_np.dtype}")
print(f"Depth array min: {depth_np.min() if depth_np.size > 0 else 'N/A'}")
print(f"Depth array max: {depth_np.max() if depth_np.size > 0 else 'N/A'}")

# Check if the array is empty or contains only NaN values
if depth_np.size == 0:
    print("Error: Depth array is empty!")
    # Create a dummy depth image for visualization
    depth_np = np.zeros((1024, 1024), dtype=np.float32)
    depth_vis = np.zeros((1024, 1024), dtype=np.uint8)
elif np.all(np.isnan(depth_np)):
    print("Warning: Depth array contains only NaN values!")
    # Create a dummy depth image for visualization
    depth_vis = np.zeros_like(depth_np, dtype=np.uint8)
else:
    # Replace NaN or inf values with 0
    depth_np_clean = np.copy(depth_np)
    depth_np_clean[~np.isfinite(depth_np_clean)] = 0
    
    # Find min and max values in the depth map, excluding NaN and infinity
    depth_min = np.min(depth_np_clean)
    depth_max = np.max(depth_np_clean)
    
    print(f"Clean depth range: min={depth_min}, max={depth_max}")
    
    # Normalize to 0-255 range
    if depth_max > depth_min:
        depth_vis = ((depth_np_clean - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
    else:
        depth_vis = np.zeros_like(depth_np_clean, dtype=np.uint8)

# Save raw depth data as numpy array
np.save("depth_output/depth_raw.npy", depth_np)

# Save as image
Image.fromarray(depth_vis).save("depth_output/depth_frame.png")

print(f"Depth frame saved to depth_output/depth_frame.png")
print(f"Raw depth data saved to depth_output/depth_raw.npy")

# Close the environment
env.close()