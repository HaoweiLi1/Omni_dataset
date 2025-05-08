# Load a scene with a robot in it and collect segmentation data
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

# Set headless mode and download key
gm.HEADLESS = True
download_key()

# Don't use GPU dynamics and use flatcache for performance boost
gm.USE_GPU_DYNAMICS = False
gm.ENABLE_FLATCACHE = True

# Choose robot to create
scene_cfg = {"type": "InteractiveTraversableScene", "scene_model": "Rs_int"}

# Add the robot we want to load with segmentation modalities
robot0_cfg = {
  "type": "Fetch",
  # Include all required segmentation modalities
  "obs_modalities": ["rgb", "seg_semantic", "seg_instance"],
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

# Update the simulator's viewer camera's pose so it points towards the robot
og.sim.viewer_camera.set_position_orientation(
    position=np.array([1.46949, -3.97358, 2.21529]),
    orientation=np.array([0.56829048, 0.09569975, 0.13571846, 0.80589577]),
)

# Important: Add segmentation modalities to the viewer camera
og.sim.viewer_camera.add_modality("seg_semantic")
og.sim.viewer_camera.add_modality("seg_instance")

# Create output directory if it doesn't exist
os.makedirs("segmentation_output", exist_ok=True)

# Run multiple steps to ensure scene is properly initialized
print("Initializing scene...")
for _ in range(10):
    action = env.action_space.sample()
    env.step(action=action)

print("Getting segmentation observations...")
# Get observations from the viewer camera
obs_dict, info_dict = og.sim.viewer_camera.get_obs()

# Debug: Print available modalities in the observation dictionary
print(f"Available modalities: {list(obs_dict.keys())}")
print(f"Available info: {list(info_dict.keys()) if info_dict else 'None'}")

# Function to create colormap similar to the GitHub reference
def get_visualization_colors():
    """
    Generate a color palette based on the GitHub reference.
    Uses predefined distinct colors for better visualization.
    """
    # Define a colormap similar to the referenced GitHub code
    # This uses a mix of distinct colors that will visually separate different segments
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

# Process semantic segmentation
if "seg_semantic" in obs_dict:
    sem_seg = obs_dict["seg_semantic"]
    # Convert PyTorch tensor to NumPy array
    sem_seg_np = sem_seg.cpu().detach().numpy()
    print(f"Semantic segmentation array shape: {sem_seg_np.shape}")
    print(f"Semantic segmentation array dtype: {sem_seg_np.dtype}")
    
    # Save raw semantic segmentation data
    np.save("segmentation_output/semantic_raw.npy", sem_seg_np)
    
    # Get all unique semantic IDs
    unique_ids = np.unique(sem_seg_np)
    n_classes = len(unique_ids)
    print(f"Number of unique semantic classes: {n_classes}")
    
    # Get colormap for visualization
    colors = get_visualization_colors()
    
    # Create an RGB image for visualization
    sem_vis = np.zeros((sem_seg_np.shape[0], sem_seg_np.shape[1], 3), dtype=np.uint8)
    
    # Map each semantic ID to a color
    # Use modulo to handle if we have more IDs than colors
    for i, id_val in enumerate(unique_ids):
        mask = (sem_seg_np == id_val)
        color_idx = i % len(colors)
        sem_vis[mask] = (colors[color_idx] * 255).astype(np.uint8)
    
    # Save semantic segmentation visualization
    Image.fromarray(sem_vis).save("segmentation_output/semantic.png")
    print("Semantic segmentation saved.")
    
    # Save semantic segmentation mapping if available
    if isinstance(info_dict, dict) and "seg_semantic" in info_dict:
        with open("segmentation_output/semantic_mapping.json", "w") as f:
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
                
        with open("segmentation_output/semantic_mapping_readable.json", "w") as f:
            json.dump(readable_mapping, f, indent=2)
            
        print("Semantic mapping saved.")
else:
    print("Semantic segmentation not available!")

# Process instance segmentation
if "seg_instance" in obs_dict:
    inst_seg = obs_dict["seg_instance"]
    # Convert PyTorch tensor to NumPy array
    inst_seg_np = inst_seg.cpu().detach().numpy()
    print(f"Instance segmentation array shape: {inst_seg_np.shape}")
    print(f"Instance segmentation array dtype: {inst_seg_np.dtype}")
    
    # Save raw instance segmentation data
    np.save("segmentation_output/instance_raw.npy", inst_seg_np)
    
    # Get all unique instance IDs
    unique_ids = np.unique(inst_seg_np)
    n_instances = len(unique_ids)
    print(f"Number of unique instances: {n_instances}")
    
    # Get colormap for visualization
    colors = get_visualization_colors()
    
    # Create an RGB image for visualization
    inst_vis = np.zeros((inst_seg_np.shape[0], inst_seg_np.shape[1], 3), dtype=np.uint8)
    
    # Map each instance ID to a color
    # Use modulo to handle if we have more IDs than colors
    for i, id_val in enumerate(unique_ids):
        mask = (inst_seg_np == id_val)
        color_idx = i % len(colors)
        inst_vis[mask] = (colors[color_idx] * 255).astype(np.uint8)
    
    # Save instance segmentation visualization
    Image.fromarray(inst_vis).save("segmentation_output/instance.png")
    print("Instance segmentation saved.")
    
    # Save instance segmentation mapping if available
    if isinstance(info_dict, dict) and "seg_instance" in info_dict:
        with open("segmentation_output/instance_mapping.json", "w") as f:
            json.dump(info_dict["seg_instance"], f, indent=2)
        print("Instance mapping saved.")
        
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
        # Similar coloring approach, but with a different color mapping
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
        Image.fromarray(inst_id_vis).save("segmentation_output/instance_id.png")
        
        # Save instance ID mapping information
        with open("segmentation_output/instance_id_mapping.json", "w") as f:
            json.dump(instance_id_mapping, f, indent=2)
        
        print("Instance segmentation ID saved.")
else:
    print("Instance segmentation not available!")

print("Segmentation data collection complete.")
print("Files saved to segmentation_output/ directory.")

# Close the environment
env.close()