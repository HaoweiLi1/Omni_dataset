import numpy as np
import os
import json

import omnigibson as og
from omnigibson.utils.asset_utils import download_key
from omnigibson.macros import gm

# Import our utilities
import utils

# ======== Configuration ========
gm.HEADLESS = True
download_key()
gm.USE_GPU_DYNAMICS = False
gm.ENABLE_FLATCACHE = True

NUM_FRAMES = 60
HIGH_RES_WIDTH = 1024
HIGH_RES_HEIGHT = 1024

# Choose robot to create
scene_cfg = {"type": "InteractiveTraversableScene", "scene_model": "Rs_int"}

# Add the robot we want to load with all modalities, and high-resolution camera
robot_cfg = {
    "type": "Fetch",
    "name": "fetch",
    "visible": False,
    # Provide whatever modalities you need - including seg_instance_id
    "obs_modalities": ["rgb", "depth", "depth_linear", "seg_semantic", "seg_instance", "seg_instance_id"],
    # Correct sensor configuration
    "sensor_config": {
        "VisionSensor": {                   # sensor class name
            "sensor_kwargs": {              # kwargs forwarded to VisionSensor.__init__
                "image_width": HIGH_RES_WIDTH,
                "image_height": HIGH_RES_HEIGHT,
            }
        }
    },
}

# Compile config
cfg = {
    "scene": scene_cfg,
    "robots": [robot_cfg],
    "env": {"action_timestep": 1 / 10., "physics_timestep": 1 / 120.},
    "render": {"viewer_width": HIGH_RES_WIDTH, "viewer_height": HIGH_RES_HEIGHT}
}

# ======== Main Script ========
def main():    
    # Create output directories using utils function
    output_dir = "robot_dataset_output"
    utils.create_dataset_dirs(output_dir)
    
    # Create the environment
    env = og.Environment(configs=cfg)
    # Get the robot 
    robot = env.robots[0]
    
    # Configure the viewer camera with high resolution too (as a fallback)
    og.sim.viewer_camera.width = HIGH_RES_WIDTH
    og.sim.viewer_camera.height = HIGH_RES_HEIGHT
    
    # Enable modalities on viewer camera (as a fallback) - now including seg_instance_id
    for modality in ["rgb", "depth", "depth_linear", "seg_semantic", "seg_instance", "seg_instance_id"]:
        try:
            og.sim.viewer_camera.add_modality(modality)
            print(f"Added {modality} to viewer camera")
        except Exception as e:
            print(f"Could not add {modality} to viewer camera: {e}")
    
    # List to store robot trajectories
    trajectories = []
    
    # Initialize scene with a few random actions
    print("Initializing scene...")
    for _ in range(10):
        action = env.action_space.sample()
        # Run step and get observations
        obs, reward, terminated, truncated, info = env.step(action=action)
    
    # Print basic info about observations to debug
    print("Observation structure:", type(obs))
    print("Observation keys:", obs.keys())
    
    # Create trajectory directory
    os.makedirs(os.path.join(output_dir, "trajectory"), exist_ok=True)
    
    # Generate dataset
    print("Generating dataset frames...")
    for frame_idx in range(NUM_FRAMES):
        print(f"\n--- Processing frame {frame_idx + 1}/{NUM_FRAMES} ---")
        
        # Take a random action (this will move the robot)
        action = env.action_space.sample()
        
        # Take the action and get the step results
        obs, reward, terminated, truncated, info = env.step(action=action)
        
        # Capture robot pose for trajectory - using utils function
        pose_matrix = utils.get_pose_matrix(robot)
        trajectories.append(pose_matrix.flatten())
        
        # Debug: print the structure of observations in the first frame
        if frame_idx == 0:
            print("Observation structure:")
            print(f"  Keys: {list(obs.keys())}")
            if "fetch" in obs:
                print(f"  Robot obs keys: {list(obs['fetch'].keys())}")
                
                # Check what's inside each key
                for key, value in obs["fetch"].items():
                    if isinstance(value, dict):
                        print(f"    {key}: {list(value.keys())}")
                    else:
                        print(f"    {key}: {type(value)}")
        
        # Use the comprehensive process_observations function to handle all modalities at once
        try:
            # Check if the process_observations function exists (added in the updated utils.py)
            if hasattr(utils, 'process_observations'):
                # Use the comprehensive wrapper function
                results = utils.process_observations(
                    obs, info, output_dir, "fetch", frame_idx, "frame"
                )
                print(f"Successfully processed all modalities for frame {frame_idx} with process_observations")
                
            else:
                # Fallback to the original approach if process_observations doesn't exist
                print("Warning: Using legacy processing functions (utils.process_observations not found)")
                
                # Extract robot observations using utils function - only for "fetch"
                robot_obs, env_info = utils.extract_robot_observations(obs, info, "fetch")
                
                # Process each modality using utils functions
                rgb_success = utils.process_rgb_frame(robot_obs, output_dir, frame_idx)
                depth_success = utils.process_depth_frame(robot_obs, output_dir, frame_idx)
                sem_results = utils.process_segmentation_frame(
                    robot_obs, env_info, output_dir, frame_idx
                )
                
                # Check if we captured all required data 
                # Check if at least one segmentation modality was successful
                sem_success = sem_results[0] is not None
                inst_success = sem_results[1] is not None
                inst_id_success = sem_results[2] is not None
                seg_success = sem_success or inst_success or inst_id_success
                
                if rgb_success and depth_success and seg_success:
                    print(f"Successfully captured all modalities for frame {frame_idx}")
                else:
                    print(f"Warning: Some modalities may be missing for frame {frame_idx}")
                
        except Exception as e:
            print(f"Error processing frame {frame_idx}: {e}")
    
    # Save trajectory data
    trajectory_path = os.path.join(output_dir, "trajectory", "robot_trajectory.txt")
    np.savetxt(trajectory_path, np.array(trajectories), fmt="%.18e", delimiter=" ")
    print(f"Robot trajectory data saved to: {trajectory_path}")
    
    # Create videos from frames using the updated create_videos function
    # This will handle the different filename patterns for each modality
    modalities = ["rgb", "depth", "semantic", "instance", "instance_id"]
    utils.create_videos(output_dir, modalities, NUM_FRAMES)
    
    # Close the environment
    env.close()

if __name__ == "__main__":
    main()