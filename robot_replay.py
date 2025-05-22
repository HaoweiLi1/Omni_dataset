"""
Robot Action Replay Data Collection with Consistent Color Mapping
Loads recorded actions from JSON file and replays them while collecting multi-modal sensor data.
Ensures consistent color mapping for segmentation data across frames (critical for 3D reconstruction).
"""

import numpy as np
import os
import json
import torch as th
import argparse
import cv2

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

HIGH_RES_WIDTH = 1024
HIGH_RES_HEIGHT = 1024

def load_recorded_actions(json_path):
    """
    Load recorded actions from JSON file.
    
    Args:
        json_path (str): Path to the JSON file containing recorded actions
        
    Returns:
        tuple: (actions_list, recording_info) where actions_list contains action data
    """
    if not os.path.exists(json_path):
        print(f"Error: Actions file {json_path} not found!")
        return None, None
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        actions_list = data.get("actions", [])
        recording_info = data.get("recording_info", {})
        
        print(f"Loaded {len(actions_list)} actions from {json_path}")
        print(f"Recording info: {recording_info}")
        
        return actions_list, recording_info
        
    except Exception as e:
        print(f"Error loading actions: {e}")
        return None, None

class ActionReplayer:
    """
    Replays recorded actions with proper timing.
    """
    
    def __init__(self, actions_list, env_timestep=1/10.):
        self.actions_list = actions_list
        self.env_timestep = env_timestep
        self.current_index = 0
        self.start_time = None
        
    def reset(self):
        """Reset replayer to beginning."""
        self.current_index = 0
        self.start_time = None
        
    def get_current_action(self, sim_time):
        """
        Get the appropriate action for the current simulation time.
        
        Args:
            sim_time (float): Current simulation time in seconds
            
        Returns:
            numpy.ndarray: Action vector, or None if replay is complete
        """
        if self.current_index >= len(self.actions_list):
            return None
            
        # Find the action that corresponds to current simulation time
        target_time = sim_time
        
        # Look for the closest action in time
        while (self.current_index < len(self.actions_list) - 1 and 
               self.actions_list[self.current_index + 1]["timestamp"] <= target_time):
            self.current_index += 1
            
        if self.current_index < len(self.actions_list):
            action = self.actions_list[self.current_index]["action"]
            return np.array(action)
        else:
            return None
            
    def is_complete(self):
        """Check if replay is complete."""
        return self.current_index >= len(self.actions_list) - 1

class DirectVideoWriter:
    """
    Class to directly write frames to video files with consistent color mapping
    for segmentation data. Adapted from dataset_traj.py
    """
    def __init__(self, output_dir, fps=10):
        """
        Initialize the video writers for different modalities.
        
        Args:
            output_dir (str): Directory to save the output videos
            fps (int): Frames per second for the videos
        """
        self.output_dir = output_dir
        self.fps = fps
        self.video_writers = {}
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.frame_size = None
        
        # Create color mappings for segmentation data - CRITICAL for consistent colors
        self.semantic_color_map = {}
        self.instance_color_map = {}
        self.instance_id_color_map = {}
        
        # Get visualization colors
        self.colors = utils.get_visualization_colors()
        
        # Paths for video files
        self.video_paths = {
            "rgb": os.path.join(output_dir, "rgb_video.mp4"),
            "depth": os.path.join(output_dir, "depth_video.mp4"),
            "semantic": os.path.join(output_dir, "semantic_video.mp4"),
            "instance": os.path.join(output_dir, "instance_video.mp4"),
            "instance_id": os.path.join(output_dir, "instance_id_video.mp4")
        }
        
        # Make sure output directory exists
        os.makedirs(output_dir, exist_ok=True)

    def initialize_writers(self, height, width):
        """
        Initialize video writers with the specified frame size.
        
        Args:
            height (int): Frame height
            width (int): Frame width
        """
        self.frame_size = (width, height)
        
        # Initialize video writers for each modality
        for modality, path in self.video_paths.items():
            self.video_writers[modality] = cv2.VideoWriter(path, self.fourcc, self.fps, self.frame_size)
            print(f"Initialized video writer for {modality} at {path}")
    
    def process_and_write_rgb(self, rgb_tensor):
        """
        Process RGB frame and write directly to video.
        
        Args:
            rgb_tensor: RGB tensor from observation
            
        Returns:
            numpy.ndarray: Processed RGB frame
        """
        if rgb_tensor is None:
            return None
            
        # Convert from tensor to numpy array
        rgb_np = rgb_tensor.cpu().detach().numpy()
        
        # Make sure we're getting actual color data
        if len(rgb_np.shape) == 3 and rgb_np.shape[2] >= 3:
            # Extract the RGB channels (in case there's alpha)
            rgb_np = rgb_np[:, :, :3]
            
            # Convert from float [0-1] to uint8 [0-255]
            if rgb_np.dtype == np.float32 or rgb_np.dtype == np.float64:
                rgb_np = (rgb_np * 255).astype(np.uint8)
            
            # Write to video
            # OpenCV uses BGR format
            bgr_np = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
            self.video_writers["rgb"].write(bgr_np)
            
            return rgb_np
        
        return None
    
    def process_and_write_depth(self, depth_tensor):
        """
        Process depth frame and write directly to video.
        
        Args:
            depth_tensor: Depth tensor from observation
            
        Returns:
            numpy.ndarray: Processed depth visualization
        """
        if depth_tensor is None:
            return None
            
        # Convert PyTorch tensor to NumPy array
        depth_np = depth_tensor.cpu().detach().numpy()
        
        # Check if array is valid
        if depth_np.size == 0 or np.all(np.isnan(depth_np)):
            depth_vis = np.zeros((self.frame_size[1], self.frame_size[0]), dtype=np.uint8)
        else:
            # Replace NaN or inf values with 0
            depth_np_clean = np.copy(depth_np)
            depth_np_clean[~np.isfinite(depth_np_clean)] = 0
            
            # Find min and max values in the depth map
            depth_min = np.min(depth_np_clean)
            depth_max = np.max(depth_np_clean)
            
            # Normalize to 0-255 range
            if depth_max > depth_min:
                depth_vis = ((depth_np_clean - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
            else:
                depth_vis = np.zeros_like(depth_np_clean, dtype=np.uint8)
        
        # Write to video (convert grayscale to BGR)
        depth_bgr = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)
        self.video_writers["depth"].write(depth_bgr)
        
        return depth_vis
    
    def process_and_write_segmentation(self, seg_data, info_dict, modality):
        """
        Process segmentation frame with consistent colors and write directly to video.
        CRITICAL: This ensures the same object maintains the same color throughout the video.
        
        Args:
            seg_data: Segmentation tensor from observation
            info_dict: Information dictionary with mappings
            modality: Type of segmentation (semantic, instance, instance_id)
            
        Returns:
            numpy.ndarray: Segmentation visualization
        """
        if seg_data is None:
            return None
            
        # Convert PyTorch tensor to NumPy array
        seg_np = seg_data.cpu().detach().numpy()
        
        # Get color map for this modality
        if modality == "semantic":
            color_map = self.semantic_color_map
        elif modality == "instance":
            color_map = self.instance_color_map
        else:  # instance_id
            color_map = self.instance_id_color_map
        
        # Get unique IDs in this frame
        unique_ids = np.unique(seg_np)
        
        # Assign colors to new IDs that haven't been seen before
        # This is the KEY to consistent color mapping!
        for id_val in unique_ids:
            if id_val not in color_map:
                # Assign a new color from the palette
                color_idx = len(color_map) % len(self.colors)
                color_map[id_val] = (self.colors[color_idx] * 255).astype(np.uint8)
        
        # Create visualization image with consistent colors
        vis_img = np.zeros((seg_np.shape[0], seg_np.shape[1], 3), dtype=np.uint8)
        
        # Apply color mapping to each pixel
        for id_val in unique_ids:
            mask = (seg_np == id_val)
            vis_img[mask] = color_map[id_val]
        
        # Write to video (OpenCV uses BGR)
        bgr_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
        self.video_writers[modality].write(bgr_img)
        
        # Save mapping if needed (only for first frame)
        if info_dict and modality in info_dict:
            mapping_dir = os.path.join(self.output_dir, modality)
            os.makedirs(mapping_dir, exist_ok=True)
            mapping_path = os.path.join(mapping_dir, f"{modality}_mapping.json")
            
            if not os.path.exists(mapping_path):
                with open(mapping_path, "w") as f:
                    json.dump(info_dict[modality], f, indent=2)
        
        return vis_img
    
    def close(self):
        """
        Close all video writers.
        """
        for writer in self.video_writers.values():
            writer.release()
        print("All video writers closed.")

def main():
    parser = argparse.ArgumentParser(description="Replay recorded robot actions while collecting sensor data")
    parser.add_argument("--actions", type=str, required=True,
                        help="Path to JSON file containing recorded actions")
    parser.add_argument("--output_dir", type=str, default="action_replay_output",
                        help="Directory to save output data")
    parser.add_argument("--save_frames", action="store_true",
                        help="Save individual frames in addition to videos")
    args = parser.parse_args()
    
    # Load recorded actions
    actions_list, recording_info = load_recorded_actions(args.actions)
    if actions_list is None:
        print("Failed to load actions. Exiting.")
        return
    
    # Create output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Save a copy of the original actions file for reference
    action_copy_path = os.path.join(output_dir, "source_actions.json")
    with open(args.actions, 'r') as src, open(action_copy_path, 'w') as dst:
        dst.write(src.read())
    
    # Create frame directories if saving individual frames
    if args.save_frames:
        utils.create_dataset_dirs(output_dir)
    
    # Environment configuration - same as robotview_generate.py
    scene_cfg = {"type": "InteractiveTraversableScene", "scene_model": "Rs_int"}
    
    robot_cfg = {
        "type": "Fetch",
        "name": "fetch",
        "visible": False,
        "obs_modalities": ["rgb", "depth", "depth_linear", "seg_semantic", "seg_instance", "seg_instance_id"],
        "sensor_config": {
            "VisionSensor": {
                "sensor_kwargs": {
                    "image_width": HIGH_RES_WIDTH,
                    "image_height": HIGH_RES_HEIGHT,
                }
            }
        },
    }
    
    cfg = {
        "scene": scene_cfg,
        "robots": [robot_cfg],
        "env": {"action_timestep": 1 / 10., "physics_timestep": 1 / 120.},
        "render": {"viewer_width": HIGH_RES_WIDTH, "viewer_height": HIGH_RES_HEIGHT}
    }
    
    # Create environment
    env = og.Environment(configs=cfg)
    robot = env.robots[0]
    
    # Set up robot controllers (same as the recording script)
    controller_config = {
        "base": {"name": "DifferentialDriveController"},
        "arm_0": {"name": "InverseKinematicsController"}, 
        "gripper_0": {"name": "MultiFingerGripperController"},
        "camera": {"name": "JointController"}
    }
    
    robot.reload_controllers(controller_config=controller_config)
    env.scene.update_initial_state()
    
    # Configure viewer camera as fallback
    og.sim.viewer_camera.width = HIGH_RES_WIDTH
    og.sim.viewer_camera.height = HIGH_RES_HEIGHT
    
    for modality in ["rgb", "depth", "depth_linear", "seg_semantic", "seg_instance", "seg_instance_id"]:
        try:
            og.sim.viewer_camera.add_modality(modality)
            print(f"Added {modality} to viewer camera")
        except Exception as e:
            print(f"Could not add {modality} to viewer camera: {e}")
    
    # Reset environment
    env.reset()
    robot.reset()
    
    # Initialize action replayer
    env_timestep = 1 / 10.  # Fixed timestep from config
    replayer = ActionReplayer(actions_list, env_timestep)
    
    # Data collection setup
    trajectories = []
    os.makedirs(os.path.join(output_dir, "trajectory"), exist_ok=True)
    
    # Initialize scene with a few steps
    print("Initializing scene...")
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action=action)
    
    # Initialize the direct video writer with consistent color mapping
    video_writer = DirectVideoWriter(output_dir, fps=10)
    
    # Sample first frame to get dimensions for initialization
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action=action)
    
    # Extract robot observations to get dimensions
    robot_obs, env_info = utils.extract_robot_observations(obs, info, "fetch")
    
    if "rgb" in robot_obs:
        rgb_np = robot_obs["rgb"].cpu().detach().numpy()
        height, width = rgb_np.shape[:2]
        video_writer.initialize_writers(height, width)
        print(f"Initialized video writers with size: {width}x{height}")
    else:
        print("Error: Could not determine frame size from first frame")
        return
    
    print(f"Starting action replay and data collection...")
    print(f"Total actions to replay: {len(actions_list)}")
    
    frame_idx = 0
    sim_time = 0.0
    
    # Main replay loop
    while not replayer.is_complete():
        # Get action for current time
        action = replayer.get_current_action(sim_time)
        
        if action is None:
            print("Replay complete!")
            break
            
        # Convert to tensor if needed
        if not isinstance(action, th.Tensor):
            action = th.tensor(action, dtype=th.float32)
        
        # Execute action
        obs, reward, terminated, truncated, info = env.step(action=action)
        
        # Record robot pose
        pose_matrix = utils.get_pose_matrix(robot)
        trajectories.append(pose_matrix.flatten())
        
        # Extract robot observations
        robot_obs, env_info = utils.extract_robot_observations(obs, info, "fetch")
        
        # Process each modality and write directly to video with consistent colors
        if "rgb" in robot_obs:
            video_writer.process_and_write_rgb(robot_obs["rgb"])
        
        if "depth" in robot_obs:
            video_writer.process_and_write_depth(robot_obs["depth"])
        elif "depth_linear" in robot_obs:
            video_writer.process_and_write_depth(robot_obs["depth_linear"])
        
        if "seg_semantic" in robot_obs:
            video_writer.process_and_write_segmentation(
                robot_obs["seg_semantic"], env_info, "semantic"
            )
        
        if "seg_instance" in robot_obs:
            video_writer.process_and_write_segmentation(
                robot_obs["seg_instance"], env_info, "instance"
            )
        
        if "seg_instance_id" in robot_obs:
            video_writer.process_and_write_segmentation(
                robot_obs["seg_instance_id"], env_info, "instance_id"
            )
        
        # Additionally save individual frames if requested
        if args.save_frames:
            utils.process_rgb_frame(robot_obs, output_dir, frame_idx, "frame")
            utils.process_depth_frame(robot_obs, output_dir, frame_idx, "frame")
            utils.process_segmentation_frame(robot_obs, env_info, output_dir, frame_idx, "frame")
        
        print(f"Processed frame {frame_idx} (t={sim_time:.2f}s)", end="\r")
        
        frame_idx += 1
        sim_time += env_timestep
        
        # Safety check
        if frame_idx > len(actions_list) * 2:  # Prevent infinite loops
            print("Safety limit reached, stopping replay")
            break
    
    # Close video writers
    video_writer.close()
    
    print(f"Replay completed!")
    print(f"Processed {frame_idx} frames")
    print(f"Simulation time: {sim_time:.2f} seconds")
    
    # Save trajectory data
    trajectory_path = os.path.join(output_dir, "trajectory", "replayed_robot_trajectory.txt")
    np.savetxt(trajectory_path, np.array(trajectories), fmt="%.18e", delimiter=" ")
    print(f"Robot trajectory saved to: {trajectory_path}")
    
    # Create videos from frames if individual frames were saved
    if args.save_frames and frame_idx > 1:
        modalities = ["rgb", "depth", "semantic", "instance", "instance_id"]
        utils.create_videos(output_dir, modalities, frame_idx)
        print("Additional videos created from individual frames")
    
    # Save replay summary
    summary = {
        "replay_info": {
            "source_actions_file": args.actions,
            "total_frames_processed": frame_idx,
            "simulation_duration": sim_time,
            "original_recording_info": recording_info,
            "consistent_color_mapping": True,
            "video_files": list(video_writer.video_paths.values())
        }
    }
    
    summary_path = os.path.join(output_dir, "replay_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Replay summary saved to: {summary_path}")
    print("🎨 Consistent color mapping ensured for 3D reconstruction compatibility")
    
    # Clean up
    env.close()

if __name__ == "__main__":
    main()