import numpy as np
import os
import json
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

# Number of frames to capture - will be overridden by loaded trajectory length
NUM_FRAMES = 60

# Initial camera values set to None - will be read from trajectory file
CAMERA_POSITION = None
CAMERA_ORIENTATION = None

# Choose robot to create
scene_cfg = {"type": "InteractiveTraversableScene", "scene_model": "Rs_int"}

# Add the robot we want to load with all modalities
robot0_cfg = {
  "type": "Fetch",
  "obs_modalities": ["rgb", "depth", "seg_semantic", "seg_instance", "seg_instance_id"],
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

def load_trajectory_from_json(json_path):
    """
    Load a camera trajectory from a JSON file created by traj_generate.py
    
    Args:
        json_path (str): Path to the JSON file containing the trajectory
        
    Returns:
        tuple: (trajectory, initial_position, initial_orientation)
            - trajectory: List of (position, orientation) tuples
            - initial_position: numpy array of initial camera position
            - initial_orientation: numpy array of initial camera orientation
    """
    # Check if file exists
    if not os.path.exists(json_path):
        print(f"Error: Trajectory file {json_path} not found!")
        return None, None, None
    
    try:
        # Load the JSON file
        with open(json_path, 'r') as f:
            trajectory_data = json.load(f)
        
        # Create a list to store the trajectory
        trajectory = []
        
        # Extract position and orientation for each frame
        for frame in trajectory_data:
            position = np.array(frame["position"])
            orientation = np.array(frame["orientation"])
            trajectory.append((position, orientation))
        
        # Get initial position and orientation from the first frame
        if len(trajectory) > 0:
            initial_position = trajectory[0][0]
            initial_orientation = trajectory[0][1]
        else:
            print("Warning: Trajectory file is empty!")
            initial_position = None
            initial_orientation = None
        
        print(f"Loaded trajectory with {len(trajectory)} frames from {json_path}")
        return trajectory, initial_position, initial_orientation
    
    except Exception as e:
        print(f"Error loading trajectory: {e}")
        return None, None, None

# ======== Main Script ========
def main():
    # Parse command line arguments for trajectory file
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", type=str, default=None, 
                        help="Path to JSON trajectory file")
    parser.add_argument("--output_dir", type=str, default="dataset_output",
                        help="Directory to save output data")
    args = parser.parse_args()
 
    # Create output directories - using utils function
    output_dir = args.output_dir
    utils.create_dataset_dirs(output_dir)
    
    # Load trajectory from JSON file
    camera_trajectory = None
    global CAMERA_POSITION, CAMERA_ORIENTATION
    
    if args.trajectory:
        camera_trajectory, initial_position, initial_orientation = load_trajectory_from_json(args.trajectory)
        # Update initial camera values from the first frame of the trajectory
        if initial_position is not None and initial_orientation is not None:
            CAMERA_POSITION = initial_position
            CAMERA_ORIENTATION = initial_orientation
            print(f"Initial camera position set to: {CAMERA_POSITION}")
            print(f"Initial camera orientation set to: {CAMERA_ORIENTATION}")
    
    # Check if trajectory was loaded successfully
    if camera_trajectory is None:
        print("No valid trajectory provided. Please specify a trajectory file with --trajectory")
        return
    
    # Create the environment
    env = og.Environment(configs=cfg)
    
    # Initialize the camera with position and orientation from the trajectory
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
    og.sim.viewer_camera.add_modality("seg_instance_id")
    
    # Update NUM_FRAMES based on the loaded trajectory
    global NUM_FRAMES
    NUM_FRAMES = len(camera_trajectory)
    print(f"Using trajectory with {NUM_FRAMES} frames")
    
    # List to store camera trajectories
    trajectories = []
    
    # Initialize scene with a few random actions
    for _ in range(10):
        action = env.action_space.sample()
        env.step(action=action)
    
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