# OmniGibson Dataset Generator - Combined Script with Random Trajectory
# Collects multiple modalities: RGB, Depth, Semantic Segmentation, Instance Segmentation, and Camera Trajectory
import numpy as np
import os
from PIL import Image
import torch
import json
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cv2
import random
import time

import omnigibson as og
from omnigibson.utils.asset_utils import download_key
from omnigibson.macros import gm
from omnigibson.utils.constants import semantic_class_id_to_name

# ======== Configuration ========
# Set headless mode and download key
gm.HEADLESS = True
download_key()

# Don't use GPU dynamics and use flatcache for performance boost
gm.USE_GPU_DYNAMICS = False
gm.ENABLE_FLATCACHE = True

# Number of frames to capture
NUM_FRAMES = 1

# Initial camera position and orientation
CAMERA_POSITION = np.array([1.46949, -3.97358, 2.21529])
CAMERA_ORIENTATION = np.array([0.56829048, 0.09569975, 0.13571846, 0.80589577])

# Choose robot to create
scene_cfg = {"type": "InteractiveTraversableScene", "scene_model": "Rs_int"}

# Add the robot we want to load with all modalities
robot0_cfg = {
  "type": "Fetch",
  "obs_modalities": ["rgb", "depth", "seg_semantic", "seg_instance"],
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

# ======== Helper Functions ========
def create_output_dirs():
    """Create all necessary output directories"""
    dirs = [
        "dataset_output",
        "dataset_output/rgb",
        "dataset_output/depth",
        "dataset_output/segmentation/semantic",
        "dataset_output/segmentation/instance",
        "dataset_output/segmentation/instance_id",
        "dataset_output/trajectory",
        "dataset_output/videos"
    ]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    print("Created output directories.")

def get_visualization_colors():
    """
    Generate a color palette for visualization.
    Uses predefined distinct colors to visually separate different segments.
    """
    colors = [
        (0, 0, 0),          # Black (background)
        (255, 0, 0),        # Red
        (0, 255, 0),        # Green
        (0, 0, 255),        # Blue
        (255, 255, 0),      # Yellow
        (255, 0, 255),      # Magenta
        (0, 255, 255),      # Cyan
        (128, 0, 0),        # Maroon
        (0, 128, 0),        # Dark Green
        (0, 0, 128),        # Navy
        (128, 128, 0),      # Olive
        (128, 0, 128),      # Purple
        (0, 128, 128),      # Teal
        (192, 0, 0),        # Dark Red
        (0, 192, 0),        # Dark Lime
        (0, 0, 192),        # Dark Blue
        (192, 192, 0),      # Dark Yellow
        (192, 0, 192),      # Dark Magenta
        (0, 192, 192),      # Dark Cyan
        (64, 0, 0),         # Very Dark Red
        (0, 64, 0),         # Very Dark Green
        (0, 0, 64),         # Very Dark Blue
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

def get_camera_extrinsic_matrix(camera):
    """Get camera extrinsic matrix (world to camera transform)"""
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

def generate_random_trajectory(num_frames, base_position, exploration_radius=0.5):
    """
    Generate a random trajectory for the camera.
    
    Args:
        num_frames: Number of frames to generate trajectory points for
        base_position: Starting position (numpy array [x, y, z])
        exploration_radius: Maximum distance from base position
        
    Returns:
        List of (position, orientation) tuples
    """
    trajectory = []
    
    # Generate a random walk trajectory
    current_position = base_position.copy()
    base_orientation = CAMERA_ORIENTATION.copy()
    
    for i in range(num_frames):
        # Random movement
        delta_x = random.uniform(-0.01, 0.01)
        delta_y = random.uniform(-0.01, 0.01)
        delta_z = random.uniform(-0.01, 0.01)
        
        # Apply movement
        current_position[0] += delta_x
        current_position[1] += delta_y
        current_position[2] += delta_z
        
        # Limit the distance from base position
        displacement = current_position - base_position
        distance = np.linalg.norm(displacement)
        if distance > exploration_radius:
            # Scale back to keep within radius
            current_position = base_position + (displacement / distance) * exploration_radius
        
        # Small random changes in orientation
        delta_orientation = np.array([
            random.uniform(-0.01, 0.01),  # x
            random.uniform(-0.01, 0.01),  # y
            random.uniform(-0.01, 0.01),  # z
            random.uniform(-0.01, 0.01)   # w
        ])
        current_orientation = base_orientation + delta_orientation
        
        # Normalize quaternion
        norm = np.linalg.norm(current_orientation)
        current_orientation = current_orientation / norm
        
        trajectory.append((current_position.copy(), current_orientation.copy()))
        
    return trajectory

def process_rgb_frame(obs_dict, frame_idx):
    """Process and save RGB data"""
    if "rgb" in obs_dict:
        # Get the RGB image (this should be a tensor)
        rgb_image = obs_dict["rgb"]
        
        # Convert from tensor to numpy array
        rgb_np = rgb_image.cpu().detach().numpy()
        
        # Make sure we're getting actual color data (check if we have 3 channels)
        if len(rgb_np.shape) == 3 and rgb_np.shape[2] >= 3:
            # Extract the RGB channels (in case there's an alpha channel)
            rgb_np = rgb_np[:, :, :3]
            
            # Convert from float [0-1] to uint8 [0-255] for proper image saving
            if rgb_np.dtype == np.float32 or rgb_np.dtype == np.float64:
                rgb_np = (rgb_np * 255).astype(np.uint8)
            
            # Save the image
            rgb_path = f"dataset_output/rgb/frame_{frame_idx:04d}.png"
            Image.fromarray(rgb_np).save(rgb_path)
            print(f"RGB frame {frame_idx} saved.")
            return rgb_np
        else:
            print(f"Error: Unexpected RGB image format - shape: {rgb_np.shape}")
    else:
        print("Error: RGB modality not available in observations")
    return None

def process_depth_frame(obs_dict, frame_idx):
    """Process and save depth data"""
    depth_key = None
    if "depth" in obs_dict:
        depth_key = "depth"
    elif "depth_linear" in obs_dict:
        depth_key = "depth_linear"
    
    if depth_key:
        # Get depth observation
        depth_obs = obs_dict[depth_key]
        
        # Convert PyTorch tensor to NumPy array
        depth_np = depth_obs.cpu().detach().numpy()
        
        # Save raw depth data
        np.save(f"dataset_output/depth/depth_raw_{frame_idx:04d}.npy", depth_np)
        
        # Check if the array is empty or contains only NaN values
        if depth_np.size == 0:
            print(f"Warning: Depth array for frame {frame_idx} is empty!")
            depth_vis = np.zeros((1024, 1024), dtype=np.uint8)
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
        depth_path = f"dataset_output/depth/depth_frame_{frame_idx:04d}.png"
        Image.fromarray(depth_vis).save(depth_path)
        print(f"Depth frame {frame_idx} saved.")
        return depth_vis
    else:
        print("Error: Depth modality not available in observations")
    return None

def process_segmentation_frame(obs_dict, info_dict, frame_idx):
    """Process and save segmentation data, returns (semantic_vis, instance_vis, instance_id_vis)"""
    colors = get_visualization_colors()
    seg_results = [None, None, None]  # semantic, instance, instance_id
    
    # Process semantic segmentation
    if "seg_semantic" in obs_dict:
        sem_seg = obs_dict["seg_semantic"]
        # Convert PyTorch tensor to NumPy array
        sem_seg_np = sem_seg.cpu().detach().numpy()
        
        # Save raw semantic segmentation data
        np.save(f"dataset_output/segmentation/semantic/semantic_raw_{frame_idx:04d}.npy", sem_seg_np)
        
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
        sem_path = f"dataset_output/segmentation/semantic/semantic_{frame_idx:04d}.png"
        Image.fromarray(sem_vis).save(sem_path)
        
        # Save semantic segmentation mapping if available (only for first frame)
        if frame_idx == 0 and isinstance(info_dict, dict) and "seg_semantic" in info_dict:
            with open("dataset_output/segmentation/semantic/semantic_mapping.json", "w") as f:
                json.dump(info_dict["seg_semantic"], f, indent=2)
            
            # Also save a human-readable mapping with class names
            readable_mapping = {}
            for id_str, category in info_dict["seg_semantic"].items():
                try:
                    id_int = int(id_str)
                    class_name = semantic_class_id_to_name.get(id_int, "unknown")
                    readable_mapping[id_str] = f"{category} ({class_name})"
                except:
                    readable_mapping[id_str] = category
                    
            with open("dataset_output/segmentation/semantic/semantic_mapping_readable.json", "w") as f:
                json.dump(readable_mapping, f, indent=2)
        
        print(f"Semantic segmentation frame {frame_idx} saved.")
        seg_results[0] = sem_vis
    else:
        print("Semantic segmentation not available!")
    
    # Process instance segmentation
    if "seg_instance" in obs_dict:
        inst_seg = obs_dict["seg_instance"]
        # Convert PyTorch tensor to NumPy array
        inst_seg_np = inst_seg.cpu().detach().numpy()
        
        # Save raw instance segmentation data
        np.save(f"dataset_output/segmentation/instance/instance_raw_{frame_idx:04d}.npy", inst_seg_np)
        
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
        inst_path = f"dataset_output/segmentation/instance/instance_{frame_idx:04d}.png"
        Image.fromarray(inst_vis).save(inst_path)
        
        # Save instance segmentation mapping if available (only for first frame)
        if isinstance(info_dict, dict) and "seg_instance" in info_dict:
            if frame_idx == 0:
                with open("dataset_output/segmentation/instance/instance_mapping.json", "w") as f:
                    json.dump(info_dict["seg_instance"], f, indent=2)
            
            # Extract and save instance segmentation ID information
            instance_id_mapping = {}
            for inst_id, inst_info in info_dict["seg_instance"].items():
                # Some OmniGibson versions provide different format info
                if isinstance(inst_info, str):
                    # Direct path
                    instance_id_mapping[inst_id] = inst_info
                elif isinstance(inst_info, dict) and "path" in inst_info:
                    # Dictionary with path key
                    instance_id_mapping[inst_id] = inst_info["path"]
                else:
                    instance_id_mapping[inst_id] = str(inst_info)
            
            # Create instance segmentation ID visualization
            inst_id_vis = np.zeros((inst_seg_np.shape[0], inst_seg_np.shape[1], 3), dtype=np.uint8)
            
            # For instance IDs, we'll use a different hash function to ensure different coloring
            for i, (inst_id, inst_path) in enumerate(instance_id_mapping.items()):
                try:
                    id_int = int(inst_id)
                    # Create a hash value from the path string for more distinct colors
                    path_hash = sum(ord(c) for c in inst_path) % len(colors)
                    mask = (inst_seg_np == id_int)
                    inst_id_vis[mask] = (colors[path_hash] * 255).astype(np.uint8)
                except:
                    continue
            
            # Save instance ID visualization
            id_path = f"dataset_output/segmentation/instance_id/instance_id_{frame_idx:04d}.png"
            Image.fromarray(inst_id_vis).save(id_path)
            
            if frame_idx == 0:
                # Save instance ID mapping information
                with open("dataset_output/segmentation/instance_id/instance_id_mapping.json", "w") as f:
                    json.dump(instance_id_mapping, f, indent=2)
            
            seg_results[2] = inst_id_vis
        
        if frame_idx == 0:
            print("Instance segmentation ID mapping saved.")
        print(f"Instance segmentation frame {frame_idx} saved.")
        seg_results[1] = inst_vis
    else:
        print("Instance segmentation not available!")
    
    return tuple(seg_results)

def create_videos(num_frames, fps=10):
    """Create videos from saved frames"""
    print("Creating videos...")
    
    # Define video paths
    video_paths = {
        "rgb": "dataset_output/videos/rgb_video.mp4",
        "depth": "dataset_output/videos/depth_video.mp4",
        "semantic": "dataset_output/videos/semantic_video.mp4",
        "instance": "dataset_output/videos/instance_video.mp4",
        "instance_id": "dataset_output/videos/instance_id_video.mp4"
    }
    
    # Get first frame dimensions for video configuration
    first_rgb = cv2.imread(f"dataset_output/rgb/frame_0000.png")
    if first_rgb is None:
        print("Error: Cannot read first frame. Skipping video creation.")
        return
    
    h, w, _ = first_rgb.shape
    
    # Initialize video writers
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writers = {}
    
    for modality, path in video_paths.items():
        video_writers[modality] = cv2.VideoWriter(path, fourcc, fps, (w, h))
    
    # Add frames to videos
    for i in range(num_frames):
        frame_idx = f"{i:04d}"
        
        # RGB video
        rgb_frame = cv2.imread(f"dataset_output/rgb/frame_{frame_idx}.png")
        if rgb_frame is not None:
            video_writers["rgb"].write(rgb_frame)
        
        # Depth video
        depth_frame = cv2.imread(f"dataset_output/depth/depth_frame_{frame_idx}.png")
        if depth_frame is not None:
            video_writers["depth"].write(depth_frame)
        else:
            # If grayscale, convert to BGR
            depth_gray = cv2.imread(f"dataset_output/depth/depth_frame_{frame_idx}.png", cv2.IMREAD_GRAYSCALE)
            if depth_gray is not None:
                depth_bgr = cv2.cvtColor(depth_gray, cv2.COLOR_GRAY2BGR)
                video_writers["depth"].write(depth_bgr)
        
        # Semantic segmentation video
        sem_frame = cv2.imread(f"dataset_output/segmentation/semantic/semantic_{frame_idx}.png")
        if sem_frame is not None:
            video_writers["semantic"].write(sem_frame)
        
        # Instance segmentation video
        inst_frame = cv2.imread(f"dataset_output/segmentation/instance/instance_{frame_idx}.png")
        if inst_frame is not None:
            video_writers["instance"].write(inst_frame)
        
        # Instance ID video
        inst_id_frame = cv2.imread(f"dataset_output/segmentation/instance_id/instance_id_{frame_idx}.png")
        if inst_id_frame is not None:
            video_writers["instance_id"].write(inst_id_frame)
    
    # Release all video writers
    for writer in video_writers.values():
        writer.release()
    
    print("Videos created successfully!")
    for modality, path in video_paths.items():
        print(f"- {modality}: {path}")

# ======== Main Script ========
def main():
    print("Starting OmniGibson Dataset Generator with Random Trajectory...")
    print(f"Will generate {NUM_FRAMES} frames of multiple modalities")
    
    # Create output directories
    create_output_dirs()
    
    # Create the environment
    env = og.Environment(configs=cfg)
    
    # Initialize the camera with all modalities
    og.sim.viewer_camera.set_position_orientation(
        position=CAMERA_POSITION,
        orientation=CAMERA_ORIENTATION,
    )
    
    # Add all required modalities to the viewer camera
    og.sim.viewer_camera.add_modality("rgb")
    og.sim.viewer_camera.add_modality("depth")
    og.sim.viewer_camera.add_modality("depth_linear")
    og.sim.viewer_camera.add_modality("seg_semantic")
    og.sim.viewer_camera.add_modality("seg_instance")
    
    # List to store camera trajectories
    trajectories = []
    
    # Initialize scene with a few random actions
    print("Initializing scene...")
    for _ in range(10):
        action = env.action_space.sample()
        env.step(action=action)
    
    # Generate random trajectory
    print("Generating random trajectory...")
    camera_trajectory = generate_random_trajectory(NUM_FRAMES, CAMERA_POSITION)
    
    # Lists to collect frames for combined visualization
    all_rgb_frames = []
    all_depth_frames = []
    all_semantic_frames = []
    all_instance_frames = []
    all_instance_id_frames = []
    
    # Generate dataset
    print("Generating dataset frames...")
    for frame_idx in range(NUM_FRAMES):
        print(f"\n--- Processing frame {frame_idx + 1}/{NUM_FRAMES} ---")
        
        # Take a random action for the robot
        action = env.action_space.sample()
        env.step(action=action)
        
        # Move camera according to trajectory
        position, orientation = camera_trajectory[frame_idx]
        og.sim.viewer_camera.set_position_orientation(
            position=position, 
            orientation=orientation
        )
        
        # Capture camera extrinsic matrix
        extrinsic_matrix = get_camera_extrinsic_matrix(og.sim.viewer_camera)
        trajectories.append(extrinsic_matrix.flatten())
        
        # Get all observations
        obs_dict, info_dict = og.sim.viewer_camera.get_obs()
        
        # Debug: Print available modalities
        print(f"Available modalities: {list(obs_dict.keys())}")
        
        # Process each modality
        rgb_frame = process_rgb_frame(obs_dict, frame_idx)
        depth_frame = process_depth_frame(obs_dict, frame_idx)
        semantic_frame, instance_frame, instance_id_frame = process_segmentation_frame(obs_dict, info_dict, frame_idx)
        
        # Collect frames for videos
        if rgb_frame is not None:
            all_rgb_frames.append(rgb_frame)
        if depth_frame is not None:
            all_depth_frames.append(depth_frame)
        if semantic_frame is not None:
            all_semantic_frames.append(semantic_frame)
        if instance_frame is not None:
            all_instance_frames.append(instance_frame)
        if instance_id_frame is not None:
            all_instance_id_frames.append(instance_id_frame)
        
        # Check if we captured all required data
        if rgb_frame is not None and depth_frame is not None and semantic_frame is not None and instance_frame is not None and instance_id_frame is not None:
            print(f"Successfully captured all modalities for frame {frame_idx}")
        else:
            print(f"Warning: Some modalities may be missing for frame {frame_idx}")
    
    # Save trajectory data
    trajectory_path = "dataset_output/trajectory/camera_trajectory.txt"
    np.savetxt(trajectory_path, np.array(trajectories), fmt="%.18e", delimiter=" ")
    print(f"Camera trajectory data saved to: {trajectory_path}")
    
    # Create videos from frames
    create_videos(NUM_FRAMES)
    
    # Close the environment
    env.close()
    
    print("\nDataset generation complete!")
    print("All files saved to dataset_output/ directory")
    print("Videos saved to dataset_output/videos/ directory")

if __name__ == "__main__":
    main()