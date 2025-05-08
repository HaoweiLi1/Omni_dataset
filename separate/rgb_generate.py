# # Load a scene with a robot in it
# import numpy as np
# import os
# from PIL import Image
# import torch

# import omnigibson as og
# from omnigibson.utils.asset_utils import download_key
# from omnigibson.macros import gm

# # Set headless mode and download key
# gm.HEADLESS = True
# download_key()

# # Don't use GPU dynamics and use flatcache for performance boost
# gm.USE_GPU_DYNAMICS = False
# gm.ENABLE_FLATCACHE = True

# # Choose robot to create
# scene_cfg = {"type": "InteractiveTraversableScene", "scene_model": "Rs_int"}

# # Add the robot we want to load
# robot0_cfg = {
#   "type": "Fetch",
#   "obs_modalities": ["rgb"],
#   "action_type": "continuous",
#   "action_normalize": True,
# }

# # Compile config
# cfg = {
#     "scene": scene_cfg,
#     "robots": [robot0_cfg],
#     "env": {"action_timestep": 1 / 10., "physics_timestep": 1 / 120.},
#     "render": {"viewer_width": 1024, "viewer_height": 1024}
# }

# # Create the environment
# env = og.Environment(configs=cfg)

# # Update the simulator's viewer camera's pose so it points towards the robot
# og.sim.viewer_camera.set_position_orientation(
#     position=np.array([1.46949, -3.97358, 2.21529]),
#     orientation=np.array([0.56829048, 0.09569975, 0.13571846, 0.80589577]),
# )

# # Create output directory if it doesn't exist
# os.makedirs("rgb_output", exist_ok=True)

# # Take a random action and capture an RGB image
# action = env.action_space.sample()
# env.step(action=action)

# # Grab an image observation from the viewer camera
# image = og.sim.viewer_camera.get_obs()[0]["rgb"]

# # Convert PyTorch tensor to NumPy array before saving
# # First ensure the tensor is on CPU and detached from computation graph
# image_np = image.cpu().detach().numpy()[:, :, :3]  # Remove alpha channel if present

# # Save the image
# Image.fromarray((image_np * 255).astype(np.uint8)).save("rgb_output/rgb_frame.png")

# print("RGB frame saved to rgb_output/rgb_frame.png")

# # Close the environment
# env.close()
# Load a scene with a robot in it and capture an RGB image
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

# Create the environment
env = og.Environment(configs=cfg)

# Update the simulator's viewer camera to match the angle used in depth and segmentation
# These position and orientation values should match your depth_generate.py
og.sim.viewer_camera.set_position_orientation(
    position=np.array([1.46949, -3.97358, 2.21529]),
    orientation=np.array([0.56829048, 0.09569975, 0.13571846, 0.80589577]),
)

# Create output directory if it doesn't exist
os.makedirs("rgb_output", exist_ok=True)

# Run several steps to ensure proper initialization
print("Initializing scene...")
for _ in range(10):
    action = env.action_space.sample()
    env.step(action=action)

print("Capturing RGB image...")
# Make sure to add the rgb modality to the camera
og.sim.viewer_camera.add_modality("rgb")

# Capture the RGB image
obs_dict = og.sim.viewer_camera.get_obs()[0]

# Verify that rgb is in the available modalities
print(f"Available modalities: {list(obs_dict.keys())}")

if "rgb" in obs_dict:
    # Get the RGB image (this should be a tensor)
    rgb_image = obs_dict["rgb"]
    
    # Convert from tensor to numpy array
    rgb_np = rgb_image.cpu().detach().numpy()
    
    # Print RGB image shape and type for debugging
    print(f"RGB image shape: {rgb_np.shape}")
    print(f"RGB image dtype: {rgb_np.dtype}")
    
    # Make sure we're getting actual color data (check if we have 3 channels)
    if len(rgb_np.shape) == 3 and rgb_np.shape[2] >= 3:
        # Extract the RGB channels (in case there's an alpha channel)
        rgb_np = rgb_np[:, :, :3]
        
        # Convert from float [0-1] to uint8 [0-255] for proper image saving
        if rgb_np.dtype == np.float32 or rgb_np.dtype == np.float64:
            rgb_np = (rgb_np * 255).astype(np.uint8)
        
        # Save the image
        Image.fromarray(rgb_np).save("rgb_output/rgb_frame.png")
        print("RGB frame saved to rgb_output/rgb_frame.png")
    else:
        print(f"Error: Unexpected RGB image format - shape: {rgb_np.shape}")
else:
    print("Error: RGB modality not available in observations")

# Close the environment
env.close()