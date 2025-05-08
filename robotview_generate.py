import numpy as np
import os
from PIL import Image
import json

import omnigibson as og
from omnigibson.utils.asset_utils import download_key
from omnigibson.macros import gm
from omnigibson.utils.constants import semantic_class_id_to_name

# ======== Configuration ========
gm.HEADLESS = True
download_key()
gm.USE_GPU_DYNAMICS = False
gm.ENABLE_FLATCACHE = True

NUM_FRAMES = 10
HIGH_RES_WIDTH = 1024
HIGH_RES_HEIGHT = 1024

# Choose robot to create
scene_cfg = {"type": "InteractiveTraversableScene", "scene_model": "Rs_int"}

# Add the robot we want to load with all modalities, and high-resolution camera
robot_cfg = {
    "type": "Fetch",
    "name": "fetch",
    "visible": False,
    # Provide whatever modalities you need
    "obs_modalities": ["rgb", "depth", "depth_linear", "seg_semantic", "seg_instance"],
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

# ======== Helper Functions ========
def create_output_dirs():
    """Create all necessary output directories"""
    dirs = [
        "robot_dataset_output",
        "robot_dataset_output/rgb",
        "robot_dataset_output/depth",
        "robot_dataset_output/segmentation",
        "robot_dataset_output/trajectory"
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
        # Get the RGB image (this should be a tensor)
        rgb_image = robot_obs["rgb"]
        
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
            rgb_path = f"robot_dataset_output/rgb/frame_{frame_idx:04d}.png"
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
        np.save(f"robot_dataset_output/depth/depth_raw_{frame_idx:04d}.npy", depth_np)
        
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
        depth_path = f"robot_dataset_output/depth/depth_frame_{frame_idx:04d}.png"
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
        np.save(f"robot_dataset_output/segmentation/semantic_raw_{frame_idx:04d}.npy", sem_seg_np)
        
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
        sem_path = f"robot_dataset_output/segmentation/semantic_{frame_idx:04d}.png"
        Image.fromarray(sem_vis).save(sem_path)
        
        # Get semantic mapping from info - if available
        sem_mapping = {}
        if "seg_semantic" in env_info:
            sem_mapping = env_info["seg_semantic"]
        
        # Save semantic segmentation mapping if available (only for first frame)
        if frame_idx == 0 and sem_mapping:
            with open("robot_dataset_output/segmentation/semantic_mapping.json", "w") as f:
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
                    
            with open("robot_dataset_output/segmentation/semantic_mapping_readable.json", "w") as f:
                json.dump(readable_mapping, f, indent=2)
        
        print(f"Semantic segmentation frame {frame_idx} saved with shape {sem_vis.shape}.")
        seg_success = True
    else:
        print("Semantic segmentation not available!")
    
    # Process instance segmentation
    if "seg_instance" in robot_obs:
        inst_seg = robot_obs["seg_instance"]
        # Convert PyTorch tensor to NumPy array
        inst_seg_np = inst_seg.cpu().detach().numpy()
        
        # Save raw instance segmentation data
        np.save(f"robot_dataset_output/segmentation/instance_raw_{frame_idx:04d}.npy", inst_seg_np)
        
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
        inst_path = f"robot_dataset_output/segmentation/instance_{frame_idx:04d}.png"
        Image.fromarray(inst_vis).save(inst_path)
        
        # Get instance mapping from info - if available
        inst_mapping = {}
        if "seg_instance" in env_info:
            inst_mapping = env_info["seg_instance"]
        
        # Save instance segmentation mapping if available (only for first frame)
        if frame_idx == 0 and inst_mapping:
            with open("robot_dataset_output/segmentation/instance_mapping.json", "w") as f:
                json.dump(inst_mapping, f, indent=2)
            
            # Extract and save instance segmentation ID information
            instance_id_mapping = {}
            for inst_id, inst_info in inst_mapping.items():
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
            
            # Save instance ID visualization (first frame only)
            id_path = f"robot_dataset_output/segmentation/instance_id_{frame_idx:04d}.png"
            Image.fromarray(inst_id_vis).save(id_path)
            
            # Save instance ID mapping information
            with open("robot_dataset_output/segmentation/instance_id_mapping.json", "w") as f:
                json.dump(instance_id_mapping, f, indent=2)
        
        if frame_idx == 0:
            print("Instance segmentation ID mapping saved.")
        print(f"Instance segmentation frame {frame_idx} saved with shape {inst_vis.shape}.")
        seg_success = True
    else:
        print("Instance segmentation not available!")
    
    return seg_success

# ======== Main Script ========
def main():    
    # Create output directories
    create_output_dirs()
    
    # Create the environment
    env = og.Environment(configs=cfg)
    # Get the robot 
    robot = env.robots[0]
    
    # Configure the viewer camera with high resolution too (as a fallback)
    og.sim.viewer_camera.width = HIGH_RES_WIDTH
    og.sim.viewer_camera.height = HIGH_RES_HEIGHT
    
    # Enable modalities on viewer camera (as a fallback)
    for modality in ["rgb", "depth", "depth_linear", "seg_semantic", "seg_instance"]:
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
    
    # Generate dataset
    print("Generating dataset frames...")
    for frame_idx in range(NUM_FRAMES):
        print(f"\n--- Processing frame {frame_idx + 1}/{NUM_FRAMES} ---")
        
        # Take a random action (this will move the robot)
        action = env.action_space.sample()
        
        # Take the action and get the step results
        obs, reward, terminated, truncated, info = env.step(action=action)
        
        # Capture robot pose for trajectory
        pose_matrix = get_robot_pose_matrix(robot)
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
        
        # Extract observations from the robot's sensors
        robot_obs = {}
        
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
        
        # If still no observations, fall back to viewer camera
        if not robot_obs:
            print("No robot observations found, falling back to viewer camera")
            viewer_obs = og.sim.viewer_camera.get_obs()[0]
            robot_obs = viewer_obs
            print(f"Viewer camera modalities: {list(robot_obs.keys())}")
        
        # Get environment info for segmentation mappings
        env_info = {}
        
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
        
        # Process each modality
        rgb_success = process_rgb_frame(robot_obs, frame_idx)
        depth_success = process_depth_frame(robot_obs, frame_idx)
        seg_success = process_segmentation_frame(robot_obs, env_info, frame_idx)
        
        # Check if we captured all required data
        if rgb_success and depth_success and seg_success:
            print(f"Successfully captured all modalities for frame {frame_idx}")
        else:
            print(f"Warning: Some modalities may be missing for frame {frame_idx}")
    
    # Save trajectory data
    trajectory_path = "robot_dataset_output/trajectory/robot_trajectory.txt"
    np.savetxt(trajectory_path, np.array(trajectories), fmt="%.18e", delimiter=" ")
    print(f"Robot trajectory data saved to: {trajectory_path}")
    
    # Close the environment
    env.close()

if __name__ == "__main__":
    main()