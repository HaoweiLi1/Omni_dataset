# OmniGibson Multi-Modal Scene Generator
# Captures RGB, Depth, and Segmentation data from simulated scenes with objects
import numpy as np
import os
from PIL import Image
import torch
import json
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import omnigibson as og
from omnigibson.utils.asset_utils import download_key
from omnigibson.macros import gm
from omnigibson.utils.constants import semantic_class_id_to_name
from omnigibson.objects.primitive_object import PrimitiveObject
from omnigibson.object_states import OnTop

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

# Add the robot with all observation modalities we need
robot0_cfg = {
  "type": "Fetch",
  "obs_modalities": ["rgb", "depth", "seg_semantic", "seg_instance"],
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

# ======== Helper Functions ========
def create_output_dirs():
    """Create all necessary output directories"""
    dirs = [
        "multimodal_output",
        "multimodal_output/rgb",
        "multimodal_output/depth",
        "multimodal_output/segmentation"
    ]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    print("Created output directories.")

def get_visualization_colors():
    """
    Generate a color palette for segmentation visualization.
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

def process_rgb_frame(obs_dict, frame_name="frame"):
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
            rgb_path = f"multimodal_output/rgb/{frame_name}.png"
            Image.fromarray(rgb_np).save(rgb_path)
            print(f"RGB frame saved to {rgb_path}")
            return True
        else:
            print(f"Error: Unexpected RGB image format - shape: {rgb_np.shape}")
    else:
        print("Error: RGB modality not available in observations")
    return False

def process_depth_frame(obs_dict, frame_name="frame"):
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
        np.save(f"multimodal_output/depth/{frame_name}_raw.npy", depth_np)
        
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
            
            print(f"Clean depth range: min={depth_min}, max={depth_max}")
            
            # Normalize to 0-255 range
            if depth_max > depth_min:
                depth_vis = ((depth_np_clean - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
            else:
                depth_vis = np.zeros_like(depth_np_clean, dtype=np.uint8)
        
        # Save as image
        depth_path = f"multimodal_output/depth/{frame_name}.png"
        Image.fromarray(depth_vis).save(depth_path)
        print(f"Depth frame saved to {depth_path}")
        return True
    else:
        print("Error: Depth modality not available in observations")
    return False

def process_segmentation_frame(obs_dict, info_dict, frame_name="frame"):
    """Process and save segmentation data"""
    colors = get_visualization_colors()
    seg_success = False
    
    # Process semantic segmentation
    if "seg_semantic" in obs_dict:
        sem_seg = obs_dict["seg_semantic"]
        # Convert PyTorch tensor to NumPy array
        sem_seg_np = sem_seg.cpu().detach().numpy()
        
        # Save raw semantic segmentation data
        np.save(f"multimodal_output/segmentation/{frame_name}_semantic_raw.npy", sem_seg_np)
        
        # Get all unique semantic IDs
        unique_ids = np.unique(sem_seg_np)
        n_classes = len(unique_ids)
        print(f"Number of unique semantic classes: {n_classes}")
        
        # Create an RGB image for visualization
        sem_vis = np.zeros((sem_seg_np.shape[0], sem_seg_np.shape[1], 3), dtype=np.uint8)
        
        # Map each semantic ID to a color
        for i, id_val in enumerate(unique_ids):
            mask = (sem_seg_np == id_val)
            color_idx = i % len(colors)
            sem_vis[mask] = (colors[color_idx] * 255).astype(np.uint8)
        
        # Save semantic segmentation visualization
        sem_path = f"multimodal_output/segmentation/{frame_name}_semantic.png"
        Image.fromarray(sem_vis).save(sem_path)
        
        # Save semantic segmentation mapping if available
        if isinstance(info_dict, dict) and "seg_semantic" in info_dict:
            with open(f"multimodal_output/segmentation/{frame_name}_semantic_mapping.json", "w") as f:
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
                    
            with open(f"multimodal_output/segmentation/{frame_name}_semantic_mapping_readable.json", "w") as f:
                json.dump(readable_mapping, f, indent=2)
                
        print(f"Semantic segmentation saved to {sem_path}")
        seg_success = True
    else:
        print("Semantic segmentation not available!")
    
    # Process instance segmentation
    if "seg_instance" in obs_dict:
        inst_seg = obs_dict["seg_instance"]
        # Convert PyTorch tensor to NumPy array
        inst_seg_np = inst_seg.cpu().detach().numpy()
        
        # Save raw instance segmentation data
        np.save(f"multimodal_output/segmentation/{frame_name}_instance_raw.npy", inst_seg_np)
        
        # Get all unique instance IDs
        unique_ids = np.unique(inst_seg_np)
        n_instances = len(unique_ids)
        print(f"Number of unique instances: {n_instances}")
        
        # Create an RGB image for visualization
        inst_vis = np.zeros((inst_seg_np.shape[0], inst_seg_np.shape[1], 3), dtype=np.uint8)
        
        # Map each instance ID to a color
        for i, id_val in enumerate(unique_ids):
            mask = (inst_seg_np == id_val)
            color_idx = i % len(colors)
            inst_vis[mask] = (colors[color_idx] * 255).astype(np.uint8)
        
        # Save instance segmentation visualization
        inst_path = f"multimodal_output/segmentation/{frame_name}_instance.png"
        Image.fromarray(inst_vis).save(inst_path)
        
        # Save instance segmentation mapping if available
        if isinstance(info_dict, dict) and "seg_instance" in info_dict:
            with open(f"multimodal_output/segmentation/{frame_name}_instance_mapping.json", "w") as f:
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
            
            # For instance IDs, we'll use a different hash function for color mapping
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
            id_path = f"multimodal_output/segmentation/{frame_name}_instance_id.png"
            Image.fromarray(inst_id_vis).save(id_path)
            
            # Save instance ID mapping information
            with open(f"multimodal_output/segmentation/{frame_name}_instance_id_mapping.json", "w") as f:
                json.dump(instance_id_mapping, f, indent=2)
            
            print(f"Instance segmentation ID saved to {id_path}")
        
        print(f"Instance segmentation saved to {inst_path}")
        seg_success = True
    else:
        print("Instance segmentation not available!")
    
    return seg_success

# ======== Main Function ========
def main():
    # Create output directories
    create_output_dirs()
    
    # Create the environment
    print("Creating environment with a robot and apple...")
    env = og.Environment(configs=cfg)
    
    # Update the simulator's viewer camera to match the angle used in other scripts
    og.sim.viewer_camera.set_position_orientation(
        position=np.array([1.46949, -3.97358, 2.21529]),
        orientation=np.array([0.56829048, 0.09569975, 0.13571846, 0.80589577]),
    )
    
    # Add all required modalities to the viewer camera
    og.sim.viewer_camera.add_modality("rgb")
    og.sim.viewer_camera.add_modality("depth")
    og.sim.viewer_camera.add_modality("depth_linear")
    og.sim.viewer_camera.add_modality("seg_semantic")
    og.sim.viewer_camera.add_modality("seg_instance")
    
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
    
    # Process and save all modalities
    frame_name = "scene_with_apple"
    rgb_success = process_rgb_frame(obs_dict, frame_name)
    depth_success = process_depth_frame(obs_dict, frame_name)
    seg_success = process_segmentation_frame(obs_dict, info_dict, frame_name)
    
    # Check if we captured all required data
    if rgb_success and depth_success and seg_success:
        print("Successfully captured all modalities")
    else:
        print("Warning: Some modalities may be missing")
    
    # Close the environment
    env.close()

if __name__ == "__main__":
    main()