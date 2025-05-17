#!/usr/bin/env python3
# OmniGibson Fetch Robot Navigation Dataset Generator
# This script implements random navigation for the Fetch robot and collects a dataset of frames

import numpy as np
import os
import time
from PIL import Image
import torch

import omnigibson as og
from omnigibson.utils.asset_utils import download_key
from omnigibson.macros import gm
from omnigibson.utils.constants import semantic_class_id_to_name
from omnigibson.tasks.point_navigation_task import PointNavigationTask
from omnigibson.scenes.traversable_scene import TraversableScene
import json

# ======== Configuration ========
# Set headless mode and download key
gm.HEADLESS = True
download_key()

# Don't use GPU dynamics and use flatcache for performance boost
gm.USE_GPU_DYNAMICS = False
gm.ENABLE_FLATCACHE = True

# Configuration parameters
NUM_EPISODES = 5  # Number of navigation episodes
MAX_STEPS_PER_EPISODE = 200  # Maximum steps per episode
HIGH_RES_WIDTH = 1024
HIGH_RES_HEIGHT = 1024
PATH_RANGE = (2.0, 10.0)  # Min and max path length for navigation (in meters)
GOAL_TOLERANCE = 0.5  # Distance in meters to consider goal reached

# Output directory
OUTPUT_DIR = "navigation_dataset"

# ======== Helper Functions ========
def create_output_dirs():
    """Create all necessary output directories"""
    dirs = [
        OUTPUT_DIR,
        f"{OUTPUT_DIR}/rgb",
        f"{OUTPUT_DIR}/depth",
        f"{OUTPUT_DIR}/segmentation",
        f"{OUTPUT_DIR}/trajectory",
        f"{OUTPUT_DIR}/metadata"
    ]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    print("Created output directories.")

def get_visualization_colors():
    """
    Generate a color palette for visualization.
    """
    colors = [
        (0, 0, 0),          # Black (background)
        (255, 0, 0),        # Red
        (0, 255, 0),        # Green
        (0, 0, 255),        # Blue
        (255, 255, 0),      # Yellow
        (255, 0, 255),      # Magenta
        (0, 255, 255),      # Cyan
    ]
    
    # Add more colors by using intermediate shades
    for r in [64, 128, 192]:
        for g in [64, 128, 192]:
            for b in [64, 128, 192]:
                if (r, g, b) not in colors:
                    colors.append((r, g, b))
    
    # Convert to numpy array and normalize to [0, 1]
    colors = np.array(colors, dtype=np.float32) / 255.0
    return colors

def get_robot_pose_matrix(robot):
    """Get robot pose matrix (world to robot transform)"""
    # Get robot position and orientation
    position = robot.get_position()
    orientation_quat = robot.get_orientation()
    
    # Convert quaternion to rotation matrix
    qw, qx, qy, qz = orientation_quat[3], orientation_quat[0], orientation_quat[1], orientation_quat[2]
    
    # Construct rotation matrix from quaternion
    rot_matrix = np.array([
        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
    ])
    
    # Create 4x4 transformation matrix
    pose_matrix = np.eye(4)
    pose_matrix[:3, :3] = rot_matrix
    pose_matrix[:3, 3] = position
    
    return pose_matrix

def process_rgb_frame(robot_obs, frame_idx):
    """Process and save RGB data from robot camera"""
    if "rgb" in robot_obs:
        # Get the RGB image tensor
        rgb_image = robot_obs["rgb"]
        
        # Convert from tensor to numpy array
        rgb_np = rgb_image.cpu().detach().numpy()
        
        # Make sure we're getting actual color data
        if len(rgb_np.shape) == 3 and rgb_np.shape[2] >= 3:
            # Extract the RGB channels (in case there's an alpha channel)
            rgb_np = rgb_np[:, :, :3]
            
            # Convert from float [0-1] to uint8 [0-255] for proper image saving
            if rgb_np.dtype == np.float32 or rgb_np.dtype == np.float64:
                rgb_np = (rgb_np * 255).astype(np.uint8)
            
            # Save the image
            rgb_path = f"{OUTPUT_DIR}/rgb/frame_{frame_idx:06d}.png"
            Image.fromarray(rgb_np).save(rgb_path)
            print(f"RGB frame {frame_idx} saved with shape {rgb_np.shape}.")
            return True
        else:
            print(f"Error: Unexpected RGB image format - shape: {rgb_np.shape}")
    else:
        print("Error: RGB modality not available in observations")
    return False

def process_depth_frame(robot_obs, frame_idx):
    """Process and save depth data from robot camera"""
    depth_key = None
    if "depth" in robot_obs:
        depth_key = "depth"
    elif "depth_linear" in robot_obs:
        depth_key = "depth_linear"
    
    if depth_key:
        # Get depth observation
        depth_obs = robot_obs[depth_key]
        # Convert PyTorch tensor to NumPy array
        depth_np = depth_obs.cpu().detach().numpy()
        
        # Save raw depth data
        np.save(f"{OUTPUT_DIR}/depth/depth_raw_{frame_idx:06d}.npy", depth_np)
        
        # Check if the array is empty or contains only NaN values
        if depth_np.size == 0:
            print(f"Warning: Depth array for frame {frame_idx} is empty!")
            depth_vis = np.zeros((HIGH_RES_HEIGHT, HIGH_RES_WIDTH), dtype=np.uint8)
        elif np.all(np.isnan(depth_np)):
            print(f"Warning: Depth array for frame {frame_idx} contains only NaN values!")
            depth_vis = np.zeros_like(depth_np, dtype=np.uint8)
        else:
            # Replace NaN or inf values with 0
            depth_np_clean = np.copy(depth_np)
            depth_np_clean[~np.isfinite(depth_np_clean)] = 0
            
            # Find min and max values in the depth map, excluding NaN and infinity
            depth_min = np.min(depth_np_clean)
            depth_max = np.max(depth_np_clean)
            
            # Normalize to 0-255 range
            if depth_max > depth_min:
                depth_vis = ((depth_np_clean - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
            else:
                depth_vis = np.zeros_like(depth_np_clean, dtype=np.uint8)
        
        # Save as image
        depth_path = f"{OUTPUT_DIR}/depth/depth_frame_{frame_idx:06d}.png"
        Image.fromarray(depth_vis).save(depth_path)
        print(f"Depth frame {frame_idx} saved with shape {depth_vis.shape}.")
        return True
    else:
        print("Error: Depth modality not available in observations")
    return False

def process_segmentation_frame(robot_obs, env_info, frame_idx):
    """Process and save segmentation data from robot camera"""
    colors = get_visualization_colors()
    seg_success = False
    
    # Process semantic segmentation
    if "seg_semantic" in robot_obs:
        sem_seg = robot_obs["seg_semantic"]
        # Convert PyTorch tensor to NumPy array
        sem_seg_np = sem_seg.cpu().detach().numpy()
        
        # Save raw semantic segmentation data
        np.save(f"{OUTPUT_DIR}/segmentation/semantic_raw_{frame_idx:06d}.npy", sem_seg_np)
        
        # Get all unique semantic IDs
        unique_ids = np.unique(sem_seg_np)
        
        # Create an RGB image for visualization
        sem_vis = np.zeros((sem_seg_np.shape[0], sem_seg_np.shape[1], 3), dtype=np.uint8)
        
        # Map each semantic ID to a color
        for i, id_val in enumerate(unique_ids):
            mask = (sem_seg_np == id_val)
            color_idx = i % len(colors)
            sem_vis[mask] = (colors[color_idx] * 255).astype(np.uint8)
        
        # Save semantic segmentation visualization
        sem_path = f"{OUTPUT_DIR}/segmentation/semantic_{frame_idx:06d}.png"
        Image.fromarray(sem_vis).save(sem_path)
        
        # Get semantic mapping from info - if available
        sem_mapping = {}
        if "seg_semantic" in env_info:
            sem_mapping = env_info["seg_semantic"]
        
        # Save semantic segmentation mapping if this is the first frame
        if frame_idx == 0 and sem_mapping:
            with open(f"{OUTPUT_DIR}/segmentation/semantic_mapping.json", "w") as f:
                json.dump(sem_mapping, f, indent=2)
            
            # Also save a human-readable mapping with class names
            readable_mapping = {}
            for id_str, category in sem_mapping.items():
                try:
                    id_int = int(id_str)
                    class_name = semantic_class_id_to_name.get(id_int, "unknown")
                    readable_mapping[id_str] = f"{category} ({class_name})"
                except:
                    readable_mapping[id_str] = category
                    
            with open(f"{OUTPUT_DIR}/segmentation/semantic_mapping_readable.json", "w") as f:
                json.dump(readable_mapping, f, indent=2)
        
        print(f"Semantic segmentation frame {frame_idx} saved.")
        seg_success = True
    else:
        print("Semantic segmentation not available!")
    
    # Process instance segmentation
    if "seg_instance" in robot_obs:
        inst_seg = robot_obs["seg_instance"]
        # Convert PyTorch tensor to NumPy array
        inst_seg_np = inst_seg.cpu().detach().numpy()
        
        # Save raw instance segmentation data
        np.save(f"{OUTPUT_DIR}/segmentation/instance_raw_{frame_idx:06d}.npy", inst_seg_np)
        
        # Get all unique instance IDs
        unique_ids = np.unique(inst_seg_np)
        
        # Create an RGB image for visualization
        inst_vis = np.zeros((inst_seg_np.shape[0], inst_seg_np.shape[1], 3), dtype=np.uint8)
        
        # Map each instance ID to a color
        for i, id_val in enumerate(unique_ids):
            mask = (inst_seg_np == id_val)
            color_idx = i % len(colors)
            inst_vis[mask] = (colors[color_idx] * 255).astype(np.uint8)
        
        # Save instance segmentation visualization
        inst_path = f"{OUTPUT_DIR}/segmentation/instance_{frame_idx:06d}.png"
        Image.fromarray(inst_vis).save(inst_path)
        
        # Get instance mapping from info - if available
        inst_mapping = {}
        if "seg_instance" in env_info:
            inst_mapping = env_info["seg_instance"]
        
        # Save instance segmentation mapping if this is the first frame
        if frame_idx == 0 and inst_mapping:
            with open(f"{OUTPUT_DIR}/segmentation/instance_mapping.json", "w") as f:
                json.dump(inst_mapping, f, indent=2)
        
        print(f"Instance segmentation frame {frame_idx} saved.")
        seg_success = True
    else:
        print("Instance segmentation not available!")
    
    return seg_success

def create_navigation_env():
    """Create the navigation environment with a Fetch robot and navigation task"""
    # Create scene configuration
    scene_cfg = {
        "type": "InteractiveTraversableScene",
        "scene_model": "Rs_int"  # Rs_int is a good default interior scene
    }
    
    # Create Fetch robot configuration with all modalities
    robot_cfg = {
        "type": "Fetch",
        "name": "fetch",
        "visible": False,
        # Include all observation modalities we want to capture
        "obs_modalities": ["rgb", "depth", "depth_linear", "seg_semantic", "seg_instance"],
        # Set high-resolution camera
        "sensor_config": {
            "VisionSensor": {
                "sensor_kwargs": {
                    "image_width": HIGH_RES_WIDTH,
                    "image_height": HIGH_RES_HEIGHT,
                }
            }
        },
    }
    
    # Configure the point navigation task
    task_cfg = {
        "type": "PointNavigationTask",
        # Let's use random initial and goal positions within a range
        "path_range": PATH_RANGE,
        # Distance tolerance to consider goal reached
        "goal_tolerance": GOAL_TOLERANCE,
        # Set to visualize the goal and path
        "visualize_goal": True,
        "visualize_path": True,
    }
    
    # Compile environment configuration
    env_cfg = {
        "scene": scene_cfg,
        "robots": [robot_cfg],
        "task": task_cfg
    }
    
    # Create environment
    env = og.Environment(configs=env_cfg)
    
    return env

def main():
    # Create output directories
    create_output_dirs()
    
    # Create the environment
    print("Creating navigation environment with Fetch robot...")
    env = create_navigation_env()
    
    # Get reference to the robot
    robot = env.robots[0]
    
    # Get reference to the task
    task = env.task
    
    # Dataset metadata
    metadata = {
        "dataset_info": {
            "name": "Fetch Navigation Dataset",
            "description": "Dataset of Fetch robot navigating to random goals",
            "date_created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "num_episodes": NUM_EPISODES,
            "max_steps_per_episode": MAX_STEPS_PER_EPISODE,
            "path_range": PATH_RANGE,
            "goal_tolerance": GOAL_TOLERANCE,
            "image_width": HIGH_RES_WIDTH,
            "image_height": HIGH_RES_HEIGHT
        },
        "episodes": []
    }
    
    # Run multiple episodes
    frame_idx = 0
    for episode in range(NUM_EPISODES):
        print(f"\n===== Starting Episode {episode+1}/{NUM_EPISODES} =====")
        
        # Reset the environment to get a new random start and goal
        obs, info = env.reset()
        
        # Extract information about the navigation goal
        start_pos = robot.get_position().tolist()
        goal_pos = task.get_goal_pos().tolist()
        
        # Add episode information to metadata
        episode_info = {
            "episode_id": episode,
            "start_position": start_pos,
            "goal_position": goal_pos,
            "frames": []
        }
        
        print(f"Start position: {start_pos}")
        print(f"Goal position: {goal_pos}")
        
        # Lists to track trajectory
        trajectory = []
        
        # Run this episode
        done = False
        success = False
        step_count = 0
        
        while not done and step_count < MAX_STEPS_PER_EPISODE:
            # Get current robot position for trajectory tracking
            current_pos = robot.get_position().tolist()
            current_ori = robot.get_orientation().tolist()
            pose_matrix = get_robot_pose_matrix(robot)
            
            # Add to trajectory
            trajectory.append({
                "frame_idx": frame_idx,
                "position": current_pos,
                "orientation": current_ori,
                "pose_matrix": pose_matrix.flatten().tolist(),
                "step": step_count
            })
            
            # Record the current observation
            # Get observations from robot sensors
            robot_obs = {}
            if "fetch" in obs:
                robot_data = obs["fetch"]
                
                # Process each sensor's observations
                for sensor_name, sensor_data in robot_data.items():
                    if isinstance(sensor_data, dict):
                        # Add all modalities from the sensor
                        for modality, data in sensor_data.items():
                            robot_obs[modality] = data
                            
                print(f"Found modalities from fetch robot: {list(robot_obs.keys())}")
            
            # Process each modality and save to dataset
            rgb_success = process_rgb_frame(robot_obs, frame_idx)
            depth_success = process_depth_frame(robot_obs, frame_idx)
            seg_success = process_segmentation_frame(robot_obs, info, frame_idx)
            
            # Add frame info to episode metadata
            episode_info["frames"].append({
                "frame_idx": frame_idx,
                "step": step_count,
                "position": current_pos,
                "orientation": current_ori,
                "distance_to_goal": np.linalg.norm(np.array(current_pos[:2]) - np.array(goal_pos[:2]))
            })
            
            # Increment frame index
            frame_idx += 1
            
            # Take a navigation step - implement a simple navigation policy
            # For a simple demonstration, we'll use random actions
            # In a real implementation, you might use a trained policy or path planner
            action = env.action_space.sample()
            
            # Take a step in the environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Check if we've reached the goal
            done = terminated or truncated
            success = info.get("success", False)
            
            print(f"Step {step_count}: Distance to goal: {info.get('distance_to_goal', 'N/A')}")
            
            # Increment step counter
            step_count += 1
        
        # Add episode result information
        episode_info["steps_taken"] = step_count
        episode_info["success"] = success
        episode_info["trajectory"] = trajectory
        episode_info["final_distance_to_goal"] = info.get("distance_to_goal", -1)
        
        # Add to metadata
        metadata["episodes"].append(episode_info)
        
        # Save trajectory data for this episode
        trajectory_path = f"{OUTPUT_DIR}/trajectory/trajectory_episode_{episode:02d}.json"
        with open(trajectory_path, "w") as f:
            json.dump(trajectory, f, indent=2)
            
        print(f"Episode {episode+1} completed in {step_count} steps. Success: {success}")
        print(f"Final distance to goal: {info.get('distance_to_goal', 'N/A')}")
        
    # Save metadata
    with open(f"{OUTPUT_DIR}/metadata/dataset_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Close the environment
    env.close()
    
    print("\nNavigation dataset generation complete!")
    print(f"Generated {frame_idx} frames across {NUM_EPISODES} episodes")
    print(f"All data saved to {OUTPUT_DIR}/ directory")

if __name__ == "__main__":
    main()