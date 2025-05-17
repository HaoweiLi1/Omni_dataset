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
from omnigibson.objects.primitive_object import PrimitiveObject
from omnigibson.object_states import OnTop

# ======== Environment Setup Functions ========

def initialize_omnigibson(
    headless=True, 
    use_gpu_dynamics=False, 
    enable_flatcache=True, 
    enable_object_states=True
):
    """
    Initialize OmniGibson settings
    
    Args:
        headless (bool): Whether to run in headless mode
        use_gpu_dynamics (bool): Whether to use GPU for physics simulation
        enable_flatcache (bool): Whether to enable flatcache for performance boost
        enable_object_states (bool): Whether to enable object states (required for OnTop)
    """
    gm.HEADLESS = headless
    download_key()
    gm.USE_GPU_DYNAMICS = use_gpu_dynamics
    gm.ENABLE_FLATCACHE = enable_flatcache
    
    if enable_object_states:
        gm.ENABLE_OBJECT_STATES = True
    
    print("OmniGibson initialized with settings:")
    print(f"  Headless: {headless}")
    print(f"  GPU Dynamics: {use_gpu_dynamics}")
    print(f"  Flatcache: {enable_flatcache}")
    print(f"  Object States: {enable_object_states}")


def create_env_config(
    scene_type="InteractiveTraversableScene", 
    scene_model="Rs_int",
    robot_type="Fetch",
    obs_modalities=None,
    action_type="continuous",
    action_normalize=True,
    render_width=1024,
    render_height=1024,
    objects=None
):
    """
    Create environment configuration for OmniGibson
    
    Args:
        scene_type (str): Type of scene to create
        scene_model (str): Model of scene to use
        robot_type (str): Type of robot to create
        obs_modalities (list): Observation modalities for the robot
        action_type (str): Type of actions (continuous or discrete)
        action_normalize (bool): Whether to normalize actions
        render_width (int): Width of rendering
        render_height (int): Height of rendering
        objects (list): List of object configurations to add to scene
        
    Returns:
        dict: Environment configuration
    """
    if obs_modalities is None:
        obs_modalities = ["rgb", "depth", "seg_semantic", "seg_instance"]
    
    # Setup scene configuration
    scene_cfg = {"type": scene_type, "scene_model": scene_model}
    
    # Setup robot configuration
    robot_cfg = {
        "type": robot_type,
        "obs_modalities": obs_modalities,
        "action_type": action_type,
        "action_normalize": action_normalize,
    }
    
    # Create environment configuration
    env_config = {
        "scene": scene_cfg,
        "robots": [robot_cfg],
        "env": {"action_timestep": 1 / 10., "physics_timestep": 1 / 120.},
        "render": {"viewer_width": render_width, "viewer_height": render_height}
    }
    
    # Add objects if specified
    if objects:
        env_config["objects"] = objects
    
    return env_config


def create_high_res_robot_config(
    robot_type="Fetch",
    obs_modalities=None,
    width=1024,
    height=1024,
    visible=False,
    name="fetch"
):
    """
    Create a robot configuration with high-resolution camera
    
    Args:
        robot_type (str): Type of robot to create
        obs_modalities (list): Observation modalities for the robot
        width (int): Width of camera sensor
        height (int): Height of camera sensor
        visible (bool): Whether robot is visible
        name (str): Name of the robot
        
    Returns:
        dict: Robot configuration
    """
    if obs_modalities is None:
        obs_modalities = ["rgb", "depth", "depth_linear", "seg_semantic", "seg_instance"]
    
    robot_cfg = {
        "type": robot_type,
        "name": name,
        "visible": visible,
        "obs_modalities": obs_modalities,
        "sensor_config": {
            "VisionSensor": {                   
                "sensor_kwargs": {             
                    "image_width": width,
                    "image_height": height,
                }
            }
        },
    }
    
    return robot_cfg

def setup_camera(
    position=np.array([1.46949, -3.97358, 2.21529]),
    orientation=np.array([0.56829048, 0.09569975, 0.13571846, 0.80589577]),
    modalities=None
):
    """
    Set up the viewer camera with position, orientation, and modalities
    
    Args:
        position (numpy.ndarray): Camera position
        orientation (numpy.ndarray): Camera orientation as quaternion [x,y,z,w]
        modalities (list): List of modalities to add to camera
    """
    og.sim.viewer_camera.set_position_orientation(
        position=position,
        orientation=orientation,
    )
    
    if modalities is None:
        modalities = ["rgb", "depth", "depth_linear", "seg_semantic", "seg_instance"]
    
    for modality in modalities:
        try:
            og.sim.viewer_camera.add_modality(modality)
            print(f"Added {modality} to viewer camera")
        except Exception as e:
            print(f"Could not add {modality} to viewer camera: {e}")


def initialize_scene(env, num_steps=10):
    """
    Initialize scene with a few random actions
    
    Args:
        env (og.Environment): OmniGibson environment
        num_steps (int): Number of random actions to take
        
    Returns:
        dict: Last observation received
    """
    print("Initializing scene...")
    obs = None
    
    for _ in range(num_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action=action)
    
    return obs, info


# ======== Directory and File Management ========

def create_output_dirs(base_dir, subdirs=None):
    """
    Create all necessary output directories
    
    Args:
        base_dir (str): Base directory to create
        subdirs (list): List of subdirectories to create
        
    Returns:
        str: Path to base directory
    """
    # Create base directory
    os.makedirs(base_dir, exist_ok=True)
    
    # Create subdirectories if specified
    if subdirs:
        for subdir in subdirs:
            dir_path = os.path.join(base_dir, subdir)
            os.makedirs(dir_path, exist_ok=True)
    
    print(f"Created output directories in {base_dir}")
    return base_dir


def create_dataset_dirs():
    """Create standard dataset output directory structure"""
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
    print("Created dataset output directories.")


def create_robot_dataset_dirs():
    """Create robot dataset output directory structure"""
    dirs = [
        "robot_dataset_output",
        "robot_dataset_output/rgb",
        "robot_dataset_output/depth",
        "robot_dataset_output/segmentation",
        "robot_dataset_output/trajectory"
    ]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    print("Created robot dataset output directories.")


def create_multimodal_dirs():
    """Create multimodal output directory structure"""
    dirs = [
        "multimodal_output",
        "multimodal_output/rgb",
        "multimodal_output/depth",
        "multimodal_output/segmentation"
    ]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    print("Created multimodal output directories.")


# ======== Camera and Trajectory Functions ========

def get_camera_extrinsic_matrix(camera):
    """
    Get camera extrinsic matrix (world to camera transform)
    
    Args:
        camera: OmniGibson camera object
        
    Returns:
        numpy.ndarray: 4x4 extrinsic matrix
    """
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


def get_robot_pose_matrix(robot):
    """
    Get robot pose matrix (world to robot transform)
    
    Args:
        robot: OmniGibson robot object
        
    Returns:
        numpy.ndarray: 4x4 pose matrix
    """
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

def save_trajectory_data(trajectory_data, output_path):
    """
    Save trajectory data to file
    
    Args:
        trajectory_data (list or numpy.ndarray): Trajectory data to save
        output_path (str): Path to save trajectory data
    """
    np.savetxt(output_path, np.array(trajectory_data), fmt="%.18e", delimiter=" ")
    print(f"Trajectory data saved to: {output_path}")


# ======== Visualization Utilities ========

def get_visualization_colors():
    """
    Generate a color palette for visualization.
    Uses predefined distinct colors to visually separate different segments.
    
    Returns:
        numpy.ndarray: Array of colors normalized to [0-1] range
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


# ======== Data Processing Functions ========

def process_rgb_frame(obs_dict, output_dir, frame_idx=None, frame_name="frame"):
    """
    Process and save RGB data
    
    Args:
        obs_dict (dict): Observation dictionary containing RGB data
        output_dir (str): Directory to save RGB data
        frame_idx (int, optional): Frame index for sequential data
        frame_name (str): Base name for the output file
        
    Returns:
        numpy.ndarray or None: RGB data if successful, None otherwise
    """
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
            
            # Determine filename based on whether frame_idx is provided
            if frame_idx is not None:
                filename = f"{frame_name}_{frame_idx:04d}.png"
            else:
                filename = f"{frame_name}.png"
            
            # Save the image
            rgb_path = os.path.join(output_dir, filename)
            Image.fromarray(rgb_np).save(rgb_path)
            print(f"RGB frame saved to {rgb_path}")
            return rgb_np
        else:
            print(f"Error: Unexpected RGB image format - shape: {rgb_np.shape}")
    else:
        print("Error: RGB modality not available in observations")
    return None


def process_depth_frame(obs_dict, output_dir, frame_idx=None, frame_name="frame"):
    """
    Process and save depth data
    
    Args:
        obs_dict (dict): Observation dictionary containing depth data
        output_dir (str): Directory to save depth data
        frame_idx (int, optional): Frame index for sequential data
        frame_name (str): Base name for the output file
        
    Returns:
        numpy.ndarray or None: Depth visualization if successful, None otherwise
    """
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
        
        # Determine filename based on whether frame_idx is provided
        if frame_idx is not None:
            raw_filename = f"depth_raw_{frame_idx:04d}.npy"
            vis_filename = f"depth_frame_{frame_idx:04d}.png"
        else:
            raw_filename = f"{frame_name}_raw.npy"
            vis_filename = f"{frame_name}.png"
        
        # Save raw depth data
        raw_path = os.path.join(output_dir, raw_filename)
        np.save(raw_path, depth_np)
        
        # Check if the array is empty or contains only NaN values
        if depth_np.size == 0:
            print(f"Warning: Depth array is empty!")
            depth_vis = np.zeros((1024, 1024), dtype=np.uint8)
        elif np.all(np.isnan(depth_np)):
            print(f"Warning: Depth array contains only NaN values!")
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
        vis_path = os.path.join(output_dir, vis_filename)
        Image.fromarray(depth_vis).save(vis_path)
        print(f"Depth frame saved to {vis_path}")
        return depth_vis
    else:
        print("Error: Depth modality not available in observations")
    return None


def process_segmentation_frame(obs_dict, info_dict, output_dir, frame_idx=None, frame_name="frame"):
    """
    Process and save segmentation data
    
    Args:
        obs_dict (dict): Observation dictionary containing segmentation data
        info_dict (dict): Information dictionary with segmentation mappings
        output_dir (str): Directory to save segmentation data
        frame_idx (int, optional): Frame index for sequential data
        frame_name (str): Base name for the output files
        
    Returns:
        tuple: (semantic_vis, instance_vis, instance_id_vis) visualization arrays
    """
    colors = get_visualization_colors()
    seg_results = [None, None, None]  # semantic, instance, instance_id
    
    # Process semantic segmentation
    if "seg_semantic" in obs_dict:
        sem_seg = obs_dict["seg_semantic"]
        # Convert PyTorch tensor to NumPy array
        sem_seg_np = sem_seg.cpu().detach().numpy()
        
        # Determine filenames based on whether frame_idx is provided
        if frame_idx is not None:
            raw_filename = f"semantic_raw_{frame_idx:04d}.npy"
            vis_filename = f"semantic_{frame_idx:04d}.png"
            mapping_filename = "semantic_mapping.json"
            readable_mapping_filename = "semantic_mapping_readable.json"
        else:
            raw_filename = f"{frame_name}_semantic_raw.npy"
            vis_filename = f"{frame_name}_semantic.png"
            mapping_filename = f"{frame_name}_semantic_mapping.json"
            readable_mapping_filename = f"{frame_name}_semantic_mapping_readable.json"
        
        # Create subdirectory for semantic if not using sequential frames
        if frame_idx is None:
            sem_dir = output_dir
        else:
            sem_dir = os.path.join(output_dir, "semantic")
            os.makedirs(sem_dir, exist_ok=True)
        
        # Save raw semantic segmentation data
        raw_path = os.path.join(sem_dir, raw_filename)
        np.save(raw_path, sem_seg_np)
        
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
        vis_path = os.path.join(sem_dir, vis_filename)
        Image.fromarray(sem_vis).save(vis_path)
        
        # Save semantic segmentation mapping if available
        if isinstance(info_dict, dict) and "seg_semantic" in info_dict:
            # Only save mappings for first frame in sequence or standalone frames
            if frame_idx is None or frame_idx == 0:
                mapping_path = os.path.join(sem_dir, mapping_filename)
                with open(mapping_path, "w") as f:
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
                
                readable_mapping_path = os.path.join(sem_dir, readable_mapping_filename)
                with open(readable_mapping_path, "w") as f:
                    json.dump(readable_mapping, f, indent=2)
        
        print(f"Semantic segmentation saved to {vis_path}")
        seg_results[0] = sem_vis
    else:
        print("Semantic segmentation not available!")
    
    # Process instance segmentation
    if "seg_instance" in obs_dict:
        inst_seg = obs_dict["seg_instance"]
        # Convert PyTorch tensor to NumPy array
        inst_seg_np = inst_seg.cpu().detach().numpy()
        
        # Determine filenames based on whether frame_idx is provided
        if frame_idx is not None:
            raw_filename = f"instance_raw_{frame_idx:04d}.npy"
            vis_filename = f"instance_{frame_idx:04d}.png"
            mapping_filename = "instance_mapping.json"
            id_vis_filename = f"instance_id_{frame_idx:04d}.png"
            id_mapping_filename = "instance_id_mapping.json"
        else:
            raw_filename = f"{frame_name}_instance_raw.npy"
            vis_filename = f"{frame_name}_instance.png"
            mapping_filename = f"{frame_name}_instance_mapping.json"
            id_vis_filename = f"{frame_name}_instance_id.png"
            id_mapping_filename = f"{frame_name}_instance_id_mapping.json"
        
        # Create subdirectories if using sequential frames
        if frame_idx is None:
            inst_dir = output_dir
            inst_id_dir = output_dir
        else:
            inst_dir = os.path.join(output_dir, "instance")
            inst_id_dir = os.path.join(output_dir, "instance_id")
            os.makedirs(inst_dir, exist_ok=True)
            os.makedirs(inst_id_dir, exist_ok=True)
        
        # Save raw instance segmentation data
        raw_path = os.path.join(inst_dir, raw_filename)
        np.save(raw_path, inst_seg_np)
        
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
        vis_path = os.path.join(inst_dir, vis_filename)
        Image.fromarray(inst_vis).save(vis_path)
        
        # Save instance segmentation mapping if available
        if isinstance(info_dict, dict) and "seg_instance" in info_dict:
            # Only save mappings for first frame in sequence or standalone frames
            if frame_idx is None or frame_idx == 0:
                mapping_path = os.path.join(inst_dir, mapping_filename)
                with open(mapping_path, "w") as f:
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
            id_vis_path = os.path.join(inst_id_dir, id_vis_filename)
            Image.fromarray(inst_id_vis).save(id_vis_path)
            
            # Save instance ID mapping information (only for first frame or standalone)
            if frame_idx is None or frame_idx == 0:
                id_mapping_path = os.path.join(inst_id_dir, id_mapping_filename)
                with open(id_mapping_path, "w") as f:
                    json.dump(instance_id_mapping, f, indent=2)
            
            seg_results[2] = inst_id_vis
        
        print(f"Instance segmentation saved to {vis_path}")
        seg_results[1] = inst_vis
    else:
        print("Instance segmentation not available!")
    
    return tuple(seg_results)


# ======== Scene Manipulation Functions ========
def settle_physics(env, num_steps=20):
    """
    Run simulation steps to let physics settle
    
    Args:
        env (og.Environment): OmniGibson environment
        num_steps (int): Number of steps to run
    """
    print(f"Letting physics settle with {num_steps} steps...")
    for _ in range(num_steps):
        action = env.action_space.sample()
        env.step(action=action)


# ======== Video Creation Functions ========

def create_videos(base_dir, modalities, num_frames, fps=10):
    """
    Create videos from saved frames
    
    Args:
        base_dir (str): Base directory containing frame data
        modalities (list): List of modalities to convert to video
        num_frames (int): Number of frames in sequence
        fps (int): Frames per second for output video
    """
    print("Creating videos...")
    
    # Define video paths and input frame directories
    video_paths = {}
    frame_dirs = {}
    
    for modality in modalities:
        if modality == "rgb":
            video_paths[modality] = os.path.join(base_dir, "videos", "rgb_video.mp4")
            frame_dirs[modality] = os.path.join(base_dir, "rgb")
        elif modality == "depth":
            video_paths[modality] = os.path.join(base_dir, "videos", "depth_video.mp4")
            frame_dirs[modality] = os.path.join(base_dir, "depth")
        elif modality == "semantic":
            video_paths[modality] = os.path.join(base_dir, "videos", "semantic_video.mp4")
            frame_dirs[modality] = os.path.join(base_dir, "segmentation", "semantic")
        elif modality == "instance":
            video_paths[modality] = os.path.join(base_dir, "videos", "instance_video.mp4")
            frame_dirs[modality] = os.path.join(base_dir, "segmentation", "instance")
        elif modality == "instance_id":
            video_paths[modality] = os.path.join(base_dir, "videos", "instance_id_video.mp4")
            frame_dirs[modality] = os.path.join(base_dir, "segmentation", "instance_id")
    
    # Create videos directory if it doesn't exist
    os.makedirs(os.path.join(base_dir, "videos"), exist_ok=True)
    
    # Get first frame dimensions for video configuration
    for modality in modalities:
        first_frame_path = os.path.join(frame_dirs[modality], f"frame_0000.png")
        if os.path.exists(first_frame_path):
            first_frame = cv2.imread(first_frame_path)
            if first_frame is not None:
                h, w = first_frame.shape[:2]
                break
    else:
        print("Error: Cannot find any valid first frame. Skipping video creation.")
        return
    
    # Initialize video writers
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writers = {}
    
    for modality in modalities:
        video_writers[modality] = cv2.VideoWriter(video_paths[modality], fourcc, fps, (w, h))
    
    # Add frames to videos
    for i in range(num_frames):
        frame_idx = f"{i:04d}"
        
        for modality in modalities:
            frame_path = os.path.join(frame_dirs[modality], f"frame_{frame_idx}.png")
            if modality == "depth":
                depth_path = os.path.join(frame_dirs[modality], f"depth_frame_{frame_idx}.png")
                if os.path.exists(depth_path):
                    frame_path = depth_path
            
            # Try to read frame
            frame = cv2.imread(frame_path)
            
            if frame is not None:
                video_writers[modality].write(frame)
            else:
                # If color read fails, try grayscale (for depth images)
                gray_frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
                if gray_frame is not None:
                    color_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
                    video_writers[modality].write(color_frame)
                else:
                    print(f"Warning: Could not read frame {frame_idx} for {modality}")
    
    # Release all video writers
    for writer in video_writers.values():
        writer.release()
    
    print("Videos created successfully!")
    for modality in modalities:
        print(f"- {modality}: {video_paths[modality]}")


# ======== Robot Observation Processing ========

def extract_robot_observations(obs, info):
    """
    Extract observations from robot sensors
    
    Args:
        obs (dict): Observation dictionary from environment step
        info (dict): Info dictionary from environment step
        
    Returns:
        tuple: (robot_obs, env_info) containing robot observations and mapping info
    """
    robot_obs = {}
    env_info = {}
    
    # Try to get from named robot
    if "fetch" in obs:
        robot_data = obs["fetch"]
        
        # Process each sensor's observations
        for sensor_name, sensor_data in robot_data.items():
            if isinstance(sensor_data, dict):
                # Add all modalities from the sensor
                for modality, data in sensor_data.items():
                    robot_obs[modality] = data
                    
        print(f"Found modalities from fetch robot: {list(robot_obs.keys())}")
    
    # If no robot observations, try with generic "robot0" key
    if not robot_obs and "robot0" in obs:
        robot_data = obs["robot0"]
        
        # Process each sensor's observations
        for sensor_name, sensor_data in robot_data.items():
            if isinstance(sensor_data, dict):
                # Add all modalities from the sensor
                for modality, data in sensor_data.items():
                    robot_obs[modality] = data
                    
        print(f"Found modalities from robot0: {list(robot_obs.keys())}")
    
    # Extract environment info for segmentation mappings
    # Try to get from named robot
    if "fetch" in info:
        for sensor_name, sensor_info in info["fetch"].items():
            if isinstance(sensor_info, dict):
                # Merge all sensor info
                env_info.update(sensor_info)
    # Try with generic "robot0" key
    elif "robot0" in info:
        for sensor_name, sensor_info in info["robot0"].items():
            if isinstance(sensor_info, dict):
                # Merge all sensor info
                env_info.update(sensor_info)
    
    # If still no observations, return empty dict
    if not robot_obs:
        print("No robot observations found")
    
    return robot_obs, env_info