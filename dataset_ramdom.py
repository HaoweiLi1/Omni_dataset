import numpy as np
import os
import random
import omnigibson as og
from omnigibson.utils.asset_utils import download_key
from omnigibson.macros import gm
import utils

# ======== Configuration ========
# Set headless mode and download key
gm.HEADLESS = True
download_key()

# Don't use GPU dynamics and use flatcache for performance boost
gm.USE_GPU_DYNAMICS = False
gm.ENABLE_FLATCACHE = True

# Number of frames to capture
NUM_FRAMES = 60

# Initial camera position and orientation
CAMERA_POSITION = np.array([1.46949, -3.97358, 2.21529])
CAMERA_ORIENTATION = np.array([0.56829048, 0.09569975, 0.13571846, 0.80589577])

# Choose robot to create
scene_cfg = {"type": "InteractiveTraversableScene", "scene_model": "Rs_int"}

# Add the robot we want to load with all modalities
robot0_cfg = {
  "type": "Fetch",
  "obs_modalities": ["rgb", "depth", "seg_semantic", "seg_instance", "seg_instance_id"],  # Added seg_instance_id
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

# ======== Main Script ========
def main():
 
    # Create output directories - using utils function
    output_dir = "dataset_output"
    utils.create_dataset_dirs(output_dir)
    
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
    og.sim.viewer_camera.add_modality("seg_instance_id")  # Added seg_instance_id
    
    # List to store camera trajectories
    trajectories = []
    
    # Initialize scene with a few random actions
    for _ in range(10):
        action = env.action_space.sample()
        env.step(action=action)
    
    # Generate random trajectory
    camera_trajectory = generate_random_trajectory(NUM_FRAMES, CAMERA_POSITION)
    
    # Lists to collect frames for combined visualization
    all_rgb_frames = []
    all_depth_frames = []
    all_semantic_frames = []
    all_instance_frames = []
    all_instance_id_frames = []
    
    # Generate dataset
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
        
        # Capture camera extrinsic matrix - using utils function
        extrinsic_matrix = utils.get_pose_matrix(og.sim.viewer_camera)
        trajectories.append(extrinsic_matrix.flatten())
        
        # Get all observations
        obs_dict, info_dict = og.sim.viewer_camera.get_obs()
        
        # Process each modality - using utils functions
        rgb_frame = utils.process_rgb_frame(obs_dict, output_dir, frame_idx, "frame")
        depth_frame = utils.process_depth_frame(obs_dict, output_dir, frame_idx, "frame")
        semantic_frame, instance_frame, instance_id_frame = utils.process_segmentation_frame(
            obs_dict, info_dict, output_dir, frame_idx, "frame"
        )
        
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
    
    # Save trajectory data
    os.makedirs(os.path.join(output_dir, "trajectory"), exist_ok=True)
    trajectory_path = os.path.join(output_dir, "trajectory", "camera_trajectory.txt")
    np.savetxt(trajectory_path, np.array(trajectories), fmt="%.18e", delimiter=" ")
    print(f"Camera trajectory data saved to: {trajectory_path}")
    
    # Create videos from frames - using utils function
    modalities = ["rgb", "depth", "semantic", "instance", "instance_id"]
    utils.create_videos(output_dir, modalities, NUM_FRAMES)
    
    # Close the environment
    env.close()

if __name__ == "__main__":
    main()