# OmniGibson Multi-Modal Scene Generator
# Captures RGB, Depth, and Segmentation data from simulated scenes with objects
import numpy as np
import os
import torch
import json

import omnigibson as og
from omnigibson.utils.asset_utils import download_key
from omnigibson.macros import gm
from omnigibson.objects.primitive_object import PrimitiveObject
from omnigibson.object_states import OnTop

# Import our utilities
import utils

# ======== Configuration ========
# Set headless mode and download key
gm.HEADLESS = True
download_key()

# Enable object states - required for OnTop to work
gm.ENABLE_OBJECT_STATES = True

# Don't use GPU dynamics and use flatcache for performance boost
gm.USE_GPU_DYNAMICS = False
gm.ENABLE_FLATCACHE = True

# Create a base scene with a robot
scene_cfg = {"type": "InteractiveTraversableScene", "scene_model": "Rs_int"}

# Add the robot with all observation modalities we need - including seg_instance_id
robot0_cfg = {
  "type": "Fetch",
  "obs_modalities": ["rgb", "depth", "seg_semantic", "seg_instance", "seg_instance_id"],
  "action_type": "continuous",
  "action_normalize": True,
}

# Define an apple object to add to the scene
apple_cfg = {
    "type": "PrimitiveObject",
    "name": "apple",
    "primitive_type": "Sphere",
    "rgba": [1.0, 0.0, 0.0, 1.0],  # Red color
    "radius": 0.05,                # Apple-sized
    "position": [1.0, 0.0, 1.0],   # Initial position (will be changed to place on table)
}

# Compile config
cfg = {
    "scene": scene_cfg,
    "robots": [robot0_cfg],
    "objects": [apple_cfg],
    "env": {"action_timestep": 1 / 10., "physics_timestep": 1 / 120.},
    "render": {"viewer_width": 1024, "viewer_height": 1024}
}

# ======== Main Function ========
def main():
    # Create output directories using utils function
    output_dir = "multimodal_output"
    utils.create_dataset_dirs(output_dir)
    
    # Create the environment
    print("Creating environment with a robot and apple...")
    env = og.Environment(configs=cfg)
    
    # Update the simulator's viewer camera to match the angle used in other scripts
    og.sim.viewer_camera.set_position_orientation(
        position=np.array([1.46949, -3.97358, 2.21529]),
        orientation=np.array([0.56829048, 0.09569975, 0.13571846, 0.80589577]),
    )
    
    # Add all required modalities to the viewer camera - including seg_instance_id
    og.sim.viewer_camera.add_modality("rgb")
    og.sim.viewer_camera.add_modality("depth")
    og.sim.viewer_camera.add_modality("depth_linear")
    og.sim.viewer_camera.add_modality("seg_semantic")
    og.sim.viewer_camera.add_modality("seg_instance")
    og.sim.viewer_camera.add_modality("seg_instance_id")  # Added instance_id modality
    
    # Run several steps to ensure proper initialization
    print("Initializing scene...")
    for _ in range(10):
        action = env.action_space.sample()
        env.step(action=action)
    
    # Find the breakfast table
    print("Looking for breakfast table...")
    table_obj = None
    all_objs = env.scene.objects
    
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
    
    # Get reference to apple
    apple = env.scene.object_registry("name", "apple")
    print(f"Apple initial position: {apple.get_position()}")
    print(f"Table position: {table_obj.get_position()}")
    
    # Place the apple on the table
    if OnTop in apple.states:
        print("Setting OnTop state for apple...")
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
    
    # Capture all modalities
    print("Capturing multi-modal scene...")
    
    # Get observations from the viewer camera
    obs_dict, info_dict = og.sim.viewer_camera.get_obs()
    
    # Debug: Print available modalities in observation dictionary
    print(f"Available modalities: {list(obs_dict.keys())}")
    
    # Process and save all modalities using the utils functions
    frame_name = "scene_with_apple"
    
    try:
        # Check if the comprehensive process_observations function exists in utils.py
        if hasattr(utils, 'process_observations'):
            # Use the comprehensive wrapper function - create dummy observation structure
            # that mimics the standard OmniGibson structure but with viewer camera data
            viewer_obs = {"viewer_camera": {"camera": obs_dict}}
            viewer_info = {"viewer_camera": {"camera": info_dict}}
            
            results = utils.process_observations(
                viewer_obs, viewer_info, output_dir, "viewer_camera", 
                frame_idx=None, frame_name=frame_name
            )
            print(f"Successfully processed all modalities using process_observations")
            
        else:
            # Fallback to individual processing functions
            print("Using individual processing functions from utils.py")
            
            # Process each modality using utils functions
            rgb_success = utils.process_rgb_frame(obs_dict, output_dir, frame_idx=None, frame_name=frame_name)
            depth_success = utils.process_depth_frame(obs_dict, output_dir, frame_idx=None, frame_name=frame_name)
            sem_results = utils.process_segmentation_frame(
                obs_dict, info_dict, output_dir, frame_idx=None, frame_name=frame_name
            )
            
            # Check if we captured all required data 
            sem_success = sem_results[0] is not None
            inst_success = sem_results[1] is not None
            inst_id_success = sem_results[2] is not None
            seg_success = sem_success or inst_success or inst_id_success
            
            if rgb_success and depth_success and seg_success:
                print(f"Successfully captured all modalities")
            else:
                print(f"Warning: Some modalities may be missing")
            
    except Exception as e:
        print(f"Error processing scene: {e}")
        
        # If utils.py functions fail, we could fall back to the original functions in this file
        # but that's not included here to encourage using the improved utils.py
    
    # Create a video from the single frame (optional)
    try:
        # Since this script only captures one frame, creating videos is optional
        # We can still call create_videos, and it will handle the case of a single frame
        modalities = ["rgb", "depth", "semantic", "instance", "instance_id"]
        utils.create_videos(output_dir, modalities, 1)
        print("Created videos from single frame")
    except Exception as e:
        print(f"Note: Video creation skipped (usually makes sense only with multiple frames): {e}")
    
    # Close the environment
    env.close()

if __name__ == "__main__":
    main()