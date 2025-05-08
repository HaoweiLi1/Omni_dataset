# Load a scene with a robot, add an apple, place it on the breakfast_table_skczfi_0, and capture RGB image
import numpy as np
import os
from PIL import Image
import torch
import time

import omnigibson as og
from omnigibson.utils.asset_utils import download_key
from omnigibson.macros import gm
from omnigibson.objects.primitive_object import PrimitiveObject
from omnigibson.object_states import OnTop
from omnigibson.utils import object_state_utils

# Set headless mode and download key
gm.HEADLESS = True
download_key()

# Enable object states - required for OnTop to work
gm.ENABLE_OBJECT_STATES = True

# Don't use GPU dynamics and use flatcache for performance boost
gm.USE_GPU_DYNAMICS = False
gm.ENABLE_FLATCACHE = True

# Choose robot to create with the InteractiveTraversableScene scene0
scene_cfg = {"type": "InteractiveTraversableScene", "scene_model": "Rs_int"}

# Add the robot we want to load
robot0_cfg = {
  "type": "Fetch",
  "obs_modalities": ["rgb"],
  "action_type": "continuous",
  "action_normalize": True,
}

# Define an apple to add to the scene
# For this example, we'll create a primitive red sphere to represent an apple
apple_cfg = {
    "type": "PrimitiveObject",
    "name": "apple",
    "primitive_type": "Sphere",
    "rgba": [1.0, 0.0, 0.0, 1.0],  # Red color
    "radius": 0.05,                # Apple-sized
    "position": [1.0, 0.0, 1.0],   # Initial position (will be changed to place on table)
}

# Compile the complete config
cfg = {
    "scene": scene_cfg,
    "robots": [robot0_cfg],
    "objects": [apple_cfg],  # We'll add only our custom apple
    "env": {"action_timestep": 1 / 10., "physics_timestep": 1 / 120.},
    "render": {"viewer_width": 1024, "viewer_height": 1024}
}

# Create output directory if it doesn't exist
os.makedirs("scene_output", exist_ok=True)

# Create the environment with the complete configuration
print("Creating environment with a robot and apple...")
env = og.Environment(configs=cfg)

# Update the simulator's viewer camera to match the angle used in other scripts
og.sim.viewer_camera.set_position_orientation(
    position=np.array([1.46949, -3.97358, 2.21529]),
    orientation=np.array([0.56829048, 0.09569975, 0.13571846, 0.80589577]),
)

# Run several steps to ensure proper initialization
print("Initializing scene...")
for _ in range(10):
    action = env.action_space.sample()
    env.step(action=action)

# Print the names of all objects in the scene to find the breakfast table
all_objs = env.scene.objects
print("Objects in scene:")
table_obj = None

# Look for the specific breakfast table by its name or part of the name
for obj in all_objs:
    print(f"  - {obj.name}")
    # Check if this is the breakfast table we're looking for
    if "breakfast_table_skczfi_0" == obj.name:
        table_obj = obj
        print(f"Found the target table: {obj.name}")
        break
    # As a fallback, also try to match any breakfast table
    elif "breakfast_table" in obj.name:
        table_obj = obj
        print(f"Found a breakfast table: {obj.name}")
        
if table_obj is None:
    # If no breakfast table found, try using object registry with category
    try:
        print("Trying to find breakfast table by category...")
        table_objs = env.scene.object_registry("category", "breakfast_table")
        if len(table_objs) > 0:
            table_obj = table_objs[0]
            print(f"Found breakfast table by category: {table_obj.name}")
    except Exception as e:
        print(f"Error finding table by category: {e}")

    # If still no breakfast table found, look for any table-like object
    if table_obj is None:
        for obj in all_objs:
            if "table" in obj.name.lower():
                table_obj = obj
                print(f"Found table-like object: {obj.name}")
                break

# If still no table found, we'll need to exit
if table_obj is None:
    print("Could not find any table object in the scene. Exiting.")
    env.close()
    exit(1)

# Get reference to our apple
apple = env.scene.object_registry("name", "apple")
print(f"Apple initial position: {apple.get_position()}")
print(f"Table position: {table_obj.get_position()}")

# Place the apple on the table using the OnTop object state
if OnTop in apple.states:
    print("Setting OnTop state for apple...")
    # Try to set the apple on top of the table
    success = apple.states[OnTop].set_value(table_obj, True)
    if success:
        print("Successfully placed apple on the table!")
    else:
        print("Failed to place apple on table using OnTop state. Using manual positioning...")
        # Position apple manually on the table if OnTop fails
        table_pos = table_obj.get_position()
        table_height = table_obj.aabb_center[2] + table_obj.aabb_extent[2]/2  # Top of the table
        new_apple_pos = np.array([table_pos[0], table_pos[1], table_height + 0.05])  # Apple radius
        apple.set_position(new_apple_pos)
        print(f"Manually set apple position to: {new_apple_pos}")
else:
    print("Apple does not have OnTop state. Using manual positioning...")
    # Position apple manually on the table
    table_pos = table_obj.get_position()
    table_height = table_obj.aabb_center[2] + table_obj.aabb_extent[2]/2  # Top of the table
    new_apple_pos = np.array([table_pos[0], table_pos[1], table_height + 0.05])  # Apple radius
    apple.set_position(new_apple_pos)
    print(f"Manually set apple position to: {new_apple_pos}")

# Run a few more steps to let physics settle
print("Letting physics settle...")
for _ in range(20):
    action = env.action_space.sample()
    env.step(action=action)

# Check final positions
print(f"Table final position: {table_obj.get_position()}")
print(f"Apple final position: {apple.get_position()}")

# Check if the apple is actually on top of the table
if OnTop in apple.states:
    on_top = apple.states[OnTop].get_value(table_obj)
    print(f"Is apple on table according to OnTop state? {on_top}")

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
        Image.fromarray(rgb_np).save("scene_output/scene_with_apple_on_breakfast_table.png")
        print("Scene image saved to scene_output/scene_with_apple_on_breakfast_table.png")
    else:
        print(f"Error: Unexpected RGB image format - shape: {rgb_np.shape}")
else:
    print("Error: RGB modality not available in observations")

# Close the environment
env.close()