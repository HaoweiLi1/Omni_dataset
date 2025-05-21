import numpy as np
import os
from PIL import Image
import json
import cv2

from omnigibson.utils.constants import semantic_class_id_to_name
from omnigibson.objects.primitive_object import PrimitiveObject
from omnigibson.object_states import OnTop

# ======== Directory and File Management ========

def create_dataset_dirs(base_dir):
    """
    Create a standard dataset directory structure with the required subdirectories
    
    Args:
        base_dir (str): Base directory path to create
        
    Returns:
        str: Path to the created base directory
    """
    # Create base directory
    os.makedirs(base_dir, exist_ok=True)
    
    # Create standard subdirectories
    subdirs = [
        "rgb",
        "depth",
        "semantic",
        "instance",
        "instance_id"
    ]
    
    # Create each subdirectory
    for subdir in subdirs:
        dir_path = os.path.join(base_dir, subdir)
        os.makedirs(dir_path, exist_ok=True)
    
    print(f"Created dataset directory structure in {base_dir}")
    return base_dir


# ======== Camera and Trajectory Function ========

def get_pose_matrix(obj):
    """
    Get pose matrix (world to object transform) for a camera or robot
    
    Args:
        obj: OmniGibson object with get_position and get_orientation methods
        
    Returns:
        numpy.ndarray: 4x4 transformation matrix
    """
    # Get position and orientation
    position = obj.get_position()
    orientation_quat = obj.get_orientation()
    
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
            rgb_path = os.path.join(output_dir, "rgb", filename)
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
        
        # Determine filenames based on whether frame_idx is provided
        if frame_idx is not None:
            raw_filename = f"depth_raw_{frame_idx:04d}.npy"
            vis_filename = f"depth_frame_{frame_idx:04d}.png"
        else:
            raw_filename = f"{frame_name}_raw.npy"
            vis_filename = f"{frame_name}.png"
        
        # Save raw depth data
        raw_path = os.path.join(output_dir, "depth", raw_filename)
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
        vis_path = os.path.join(output_dir, "depth", vis_filename)
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
        
        # Directory for semantic segmentation
        sem_dir = os.path.join(output_dir, "semantic")
        
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
        else:
            raw_filename = f"{frame_name}_instance_raw.npy"
            vis_filename = f"{frame_name}_instance.png"
            mapping_filename = f"{frame_name}_instance_mapping.json"
        
        # Directory for instance segmentation
        inst_dir = os.path.join(output_dir, "instance")
        
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
        
        print(f"Instance segmentation saved to {vis_path}")
        seg_results[1] = inst_vis
    else:
        print("Instance segmentation not available!")
    
    # Process instance ID segmentation directly from the dedicated modality
    if "seg_instance_id" in obs_dict:
        inst_id_seg = obs_dict["seg_instance_id"]
        # Convert PyTorch tensor to NumPy array
        inst_id_seg_np = inst_id_seg.cpu().detach().numpy()
        
        # Determine filenames based on whether frame_idx is provided
        if frame_idx is not None:
            raw_filename = f"instance_id_raw_{frame_idx:04d}.npy"
            vis_filename = f"instance_id_{frame_idx:04d}.png"
            mapping_filename = "instance_id_mapping.json"
        else:
            raw_filename = f"{frame_name}_instance_id_raw.npy"
            vis_filename = f"{frame_name}_instance_id.png"
            mapping_filename = f"{frame_name}_instance_id_mapping.json"
        
        # Directory for instance ID segmentation
        inst_id_dir = os.path.join(output_dir, "instance_id")
        
        # Save raw instance ID segmentation data
        raw_path = os.path.join(inst_id_dir, raw_filename)
        np.save(raw_path, inst_id_seg_np)
        
        # Get all unique instance ID values
        unique_ids = np.unique(inst_id_seg_np)
        
        # Create an RGB image for visualization
        inst_id_vis = np.zeros((inst_id_seg_np.shape[0], inst_id_seg_np.shape[1], 3), dtype=np.uint8)
        
        # Map each instance ID value to a color
        for i, id_val in enumerate(unique_ids):
            mask = (inst_id_seg_np == id_val)
            color_idx = i % len(colors)
            inst_id_vis[mask] = (colors[color_idx] * 255).astype(np.uint8)
        
        # Save instance ID segmentation visualization
        vis_path = os.path.join(inst_id_dir, vis_filename)
        Image.fromarray(inst_id_vis).save(vis_path)
        
        # Save instance ID mapping if available
        if isinstance(info_dict, dict) and "seg_instance_id" in info_dict:
            # Only save mappings for first frame in sequence or standalone frames
            if frame_idx is None or frame_idx == 0:
                mapping_path = os.path.join(inst_id_dir, mapping_filename)
                with open(mapping_path, "w") as f:
                    json.dump(info_dict["seg_instance_id"], f, indent=2)
        
        print(f"Instance ID segmentation saved to {vis_path}")
        seg_results[2] = inst_id_vis
    else:
        print("Instance ID segmentation not available!")
    
    return tuple(seg_results)

# ======== Video Creation Functions ========

def create_videos(base_dir, modalities, num_frames, fps=10):
    """
    Create videos from saved frames, saving directly to the base directory
    
    Args:
        base_dir (str): Base directory containing frame data
        modalities (list): List of modalities to convert to video (rgb, depth, semantic, instance, instance_id)
        num_frames (int): Number of frames in sequence
        fps (int): Frames per second for output video
    """
    print("Creating videos...")
    
    # Define video paths and input frame directories
    video_paths = {}
    frame_dirs = {}
    
    for modality in modalities:
        if modality == "rgb":
            video_paths[modality] = os.path.join(base_dir, "rgb_video.mp4")
            frame_dirs[modality] = os.path.join(base_dir, "rgb")
        elif modality == "depth":
            video_paths[modality] = os.path.join(base_dir, "depth_video.mp4")
            frame_dirs[modality] = os.path.join(base_dir, "depth")
        elif modality == "semantic":
            video_paths[modality] = os.path.join(base_dir, "semantic_video.mp4")
            frame_dirs[modality] = os.path.join(base_dir, "semantic")
        elif modality == "instance":
            video_paths[modality] = os.path.join(base_dir, "instance_video.mp4")
            frame_dirs[modality] = os.path.join(base_dir, "instance")
        elif modality == "instance_id":
            video_paths[modality] = os.path.join(base_dir, "instance_id_video.mp4")
            frame_dirs[modality] = os.path.join(base_dir, "instance_id")
    
    # Get first frame dimensions for video configuration
    for modality in modalities:
        # Build potential frame paths based on modality
        potential_paths = []
        if modality == "rgb":
            potential_paths = [os.path.join(frame_dirs[modality], f"frame_0000.png")]
        elif modality == "depth":
            potential_paths = [
                os.path.join(frame_dirs[modality], f"depth_frame_0000.png"),
                os.path.join(frame_dirs[modality], f"frame_0000.png")
            ]
        elif modality == "semantic":
            potential_paths = [
                os.path.join(frame_dirs[modality], f"semantic_0000.png"),
                os.path.join(frame_dirs[modality], f"frame_0000.png")
            ]
        elif modality == "instance":
            potential_paths = [
                os.path.join(frame_dirs[modality], f"instance_0000.png"),
                os.path.join(frame_dirs[modality], f"frame_0000.png")
            ]
        elif modality == "instance_id":
            potential_paths = [
                os.path.join(frame_dirs[modality], f"instance_id_0000.png"),
                os.path.join(frame_dirs[modality], f"frame_0000.png")
            ]
        
        # Try each potential path
        for path in potential_paths:
            if os.path.exists(path):
                first_frame = cv2.imread(path)
                if first_frame is not None:
                    h, w = first_frame.shape[:2]
                    print(f"Found first frame for {modality} at {path}")
                    break
        else:
            # If no frame found for this modality, continue to next modality
            continue
        
        # If dimensions were found, break the outer loop
        if 'h' in locals() and 'w' in locals():
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
            frame_path = None
            
            # Choose the correct frame path based on modality
            if modality == "rgb":
                frame_path = os.path.join(frame_dirs[modality], f"frame_{frame_idx}.png")
            elif modality == "depth":
                # Try depth-specific filename first, then fallback
                depth_path = os.path.join(frame_dirs[modality], f"depth_frame_{frame_idx}.png")
                if os.path.exists(depth_path):
                    frame_path = depth_path
                else:
                    frame_path = os.path.join(frame_dirs[modality], f"frame_{frame_idx}.png")
            elif modality == "semantic":
                # Try semantic-specific filename first, then fallback
                semantic_path = os.path.join(frame_dirs[modality], f"semantic_{frame_idx}.png")
                if os.path.exists(semantic_path):
                    frame_path = semantic_path
                else:
                    frame_path = os.path.join(frame_dirs[modality], f"frame_{frame_idx}.png")
            elif modality == "instance":
                # Try instance-specific filename first, then fallback
                instance_path = os.path.join(frame_dirs[modality], f"instance_{frame_idx}.png")
                if os.path.exists(instance_path):
                    frame_path = instance_path
                else:
                    frame_path = os.path.join(frame_dirs[modality], f"frame_{frame_idx}.png")
            elif modality == "instance_id":
                # Try instance_id-specific filename first, then fallback
                instance_id_path = os.path.join(frame_dirs[modality], f"instance_id_{frame_idx}.png")
                if os.path.exists(instance_id_path):
                    frame_path = instance_id_path
                else:
                    frame_path = os.path.join(frame_dirs[modality], f"frame_{frame_idx}.png")
            
            if frame_path and os.path.exists(frame_path):
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
            else:
                print(f"Warning: Frame file not found for {modality} at index {frame_idx}")
    
    # Release all video writers
    for writer in video_writers.values():
        writer.release()
    
    print("Videos created successfully!")
    for modality in modalities:
        print(f"- {modality}: {video_paths[modality]}")


# ======== Robot Observation Processing ========

def extract_robot_observations(obs, info, robot_name="fetch"):
    """
    Extract observations from a specific robot's sensors
    
    Args:
        obs (dict): Observation dictionary from environment step
        info (dict): Info dictionary from environment step
        robot_name (str): Name of the robot to extract observations for (default: "fetch")
                          Common robot names in OmniGibson include:
                          "fetch", "tiago", "freight", "locobot", "turtlebot", etc.
        
    Returns:
        tuple: (robot_obs, env_info) containing robot observations and mapping info
    """
    robot_obs = {}
    env_info = {}
    
    # Extract robot observations from the specified robot
    if robot_name in obs:
        robot_data = obs[robot_name]
        
        # Process each sensor's observations
        for sensor_name, sensor_data in robot_data.items():
            if isinstance(sensor_data, dict):
                # Add all modalities from the sensor
                for modality, data in sensor_data.items():
                    robot_obs[modality] = data
        
        print(f"Found modalities from {robot_name} robot: {list(robot_obs.keys())}")
    else:
        print(f"No observations found for robot '{robot_name}'")
    
    # Extract environment info for segmentation mappings
    if robot_name in info:
        for sensor_name, sensor_info in info[robot_name].items():
            if isinstance(sensor_info, dict):
                # Merge all sensor info
                env_info.update(sensor_info)
    
    return robot_obs, env_info