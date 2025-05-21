import numpy as np
import os
import json
import omnigibson as og
from omnigibson.utils.asset_utils import download_key
from omnigibson.macros import gm
import cv2
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

class DirectVideoWriter:
    """
    Class to directly write frames to video files with consistent color mapping
    for segmentation data.
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
        
        # Create color mappings for segmentation data
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
            
            # Also save raw data if needed for other purposes
            # You can add code here to save the raw data if necessary
            
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
        
        # Also save raw data if needed
        # np.save(os.path.join(self.output_dir, "depth", f"depth_raw_{frame_idx:04d}.npy"), depth_np)
        
        return depth_vis
    
    def process_and_write_segmentation(self, seg_data, info_dict, modality):
        """
        Process segmentation frame with consistent colors and write directly to video.
        
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

# ======== Main Script ========
def main():
    # Parse command line arguments for trajectory file
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", type=str, default=None, 
                        help="Path to JSON trajectory file")
    parser.add_argument("--output_dir", type=str, default="dataset_output",
                        help="Directory to save output data")
    parser.add_argument("--save_frames", action="store_true",
                        help="Save individual frames in addition to videos")
    args = parser.parse_args()
 
    # Create output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # If saving frames is requested, create the subdirectories
    if args.save_frames:
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
    
    # Initialize the direct video writer
    video_writer = DirectVideoWriter(output_dir, fps=10)
    
    # Sample first frame to get dimensions for initialization
    action = env.action_space.sample()
    env.step(action=action)
    og.sim.viewer_camera.set_position_orientation(
        position=camera_trajectory[0][0], 
        orientation=camera_trajectory[0][1]
    )
    obs_dict, _ = og.sim.viewer_camera.get_obs()
    
    if "rgb" in obs_dict:
        rgb_np = obs_dict["rgb"].cpu().detach().numpy()
        height, width = rgb_np.shape[:2]
        video_writer.initialize_writers(height, width)
    else:
        print("Error: Could not determine frame size from first frame")
        return
    
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
        
        # Process each modality and write directly to video
        if "rgb" in obs_dict:
            video_writer.process_and_write_rgb(obs_dict["rgb"])
        
        if "depth" in obs_dict:
            video_writer.process_and_write_depth(obs_dict["depth"])
        elif "depth_linear" in obs_dict:
            video_writer.process_and_write_depth(obs_dict["depth_linear"])
        
        if "seg_semantic" in obs_dict:
            video_writer.process_and_write_segmentation(
                obs_dict["seg_semantic"], info_dict, "semantic"
            )
        
        if "seg_instance" in obs_dict:
            video_writer.process_and_write_segmentation(
                obs_dict["seg_instance"], info_dict, "instance"
            )
        
        if "seg_instance_id" in obs_dict:
            video_writer.process_and_write_segmentation(
                obs_dict["seg_instance_id"], info_dict, "instance_id"
            )
        
        # Additionally save individual frames if requested
        if args.save_frames:
            utils.process_rgb_frame(obs_dict, output_dir, frame_idx, "frame")
            utils.process_depth_frame(obs_dict, output_dir, frame_idx, "frame")
            utils.process_segmentation_frame(obs_dict, info_dict, output_dir, frame_idx, "frame")
    
    # Close video writers
    video_writer.close()
    
    # Save trajectory data
    os.makedirs(os.path.join(output_dir, "trajectory"), exist_ok=True)
    trajectory_path = os.path.join(output_dir, "trajectory", "camera_trajectory.txt")
    np.savetxt(trajectory_path, np.array(trajectories), fmt="%.18e", delimiter=" ")
    print(f"Camera trajectory data saved to: {trajectory_path}")
    
    # Close the environment
    env.close()

if __name__ == "__main__":
    main()