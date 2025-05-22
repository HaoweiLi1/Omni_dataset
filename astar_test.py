import os
import numpy as np
import time
import matplotlib.pyplot as plt
import math

import omnigibson as og
from omnigibson.macros import gm
from omnigibson.utils.asset_utils import download_key
from omnigibson.utils.motion_planning_utils import astar  # Import the astar function from OmniGibson

# ======== Configuration ========
# Set headless mode and download key
gm.HEADLESS = False  # Set to True for headless mode (no visualization)
download_key()

# Don't use GPU dynamics and use flatcache for performance boost
gm.USE_GPU_DYNAMICS = False
gm.ENABLE_FLATCACHE = True


def create_search_map(scene, resolution=0.1, height=0.1):
    """
    Create a search map from the scene for A* path planning.
    
    Args:
        scene: OmniGibson scene
        resolution: Grid map resolution (cell size in meters)
        height: Height at which to check for obstacles
        
    Returns:
        search_map: 2D numpy array where 0 = free, 1 = obstacle
        bounds: Tuple of (min_x, min_y, max_x, max_y) in world coordinates
    """
    # Use default bounds as OmniGibson scene doesn't support get_floor_bbox
    min_x, min_y = -5, -5
    max_x, max_y = 5, 5
    
    print(f"Using bounds: min_x={min_x}, min_y={min_y}, max_x={max_x}, max_y={max_y}")
    
    # Calculate grid dimensions
    width = int((max_x - min_x) / resolution) + 1
    length = int((max_y - min_y) / resolution) + 1
    
    print(f"Grid dimensions: {width}x{length}")
    
    # Initialize search map (0 = free, 1 = obstacle)
    search_map = np.zeros((width, length), dtype=np.uint8)
    
    print("Creating a default search map with boundary walls")
    
    # Add boundary walls
    wall_thickness = 2  # cells
    
    # Add walls around the perimeter
    search_map[0:wall_thickness, :] = 1  # Left wall
    search_map[-wall_thickness:, :] = 1  # Right wall
    search_map[:, 0:wall_thickness] = 1  # Bottom wall
    search_map[:, -wall_thickness:] = 1  # Top wall
    
    # Add some obstacles in the center for demonstration
    center_x, center_y = width // 2, length // 2
    
    # Add a cross-shaped obstacle in the center
    obstacle_size = 20
    search_map[center_x-obstacle_size//2:center_x+obstacle_size//2, center_y-5:center_y+5] = 1  # Horizontal bar
    search_map[center_x-5:center_x+5, center_y-obstacle_size//2:center_y+obstacle_size//2] = 1  # Vertical bar
    
    return search_map, (min_x, min_y, max_x, max_y)


def world_to_grid(position, bounds, resolution):
    """
    Convert world coordinates to grid coordinates.
    
    Args:
        position: Position in world coordinates (x, y) - can be tensor, numpy array, or list
        bounds: Tuple of (min_x, min_y, max_x, max_y) in world coordinates
        resolution: Grid resolution
        
    Returns:
        grid_pos: Position in grid coordinates as a tuple (x, y)
    """
    # Convert tensor to numpy array if needed
    if hasattr(position, 'cpu'):  # Check if it's a tensor
        position = position.cpu().numpy()
    
    min_x, min_y, _, _ = bounds
    grid_x = int((float(position[0]) - min_x) / resolution)
    grid_y = int((float(position[1]) - min_y) / resolution)
    return (grid_x, grid_y)  # Return as tuple, not numpy array


def grid_to_world(grid_pos, bounds, resolution):
    """
    Convert grid coordinates to world coordinates.
    
    Args:
        grid_pos: Position in grid coordinates (x, y)
        bounds: Tuple of (min_x, min_y, max_x, max_y) in world coordinates
        resolution: Grid resolution
        
    Returns:
        world_pos: Position in world coordinates (x, y)
    """
    min_x, min_y, _, _ = bounds
    world_x = grid_pos[0] * resolution + min_x
    world_y = grid_pos[1] * resolution + min_y
    return np.array([world_x, world_y], dtype=np.float32)


def visualize_path(search_map, path, start_pos, goal_pos):
    """
    Visualize the path on the search map.
    
    Args:
        search_map: 2D grid map where 0 = free, 1 = obstacle
        path: Path from start to goal (list of tuples of grid positions)
        start_pos: Start position in grid coordinates (tuple)
        goal_pos: Goal position in grid coordinates (tuple)
    """
    if path is None or len(path) == 0:
        print("No path to visualize!")
        return
        
    # Transpose the grid map for visualization (matplotlib's (0,0) is bottom-left)
    plt.figure(figsize=(10, 10))
    plt.imshow(search_map.T, cmap='binary', origin='lower')
    
    # Extract x and y coordinates from path
    x_coords = [pos[0] for pos in path]
    y_coords = [pos[1] for pos in path]
    
    # Plot the path
    plt.plot(x_coords, y_coords, 'r-', linewidth=2)
    
    # Plot start and goal
    plt.plot(start_pos[0], start_pos[1], 'go', markersize=10)
    plt.plot(goal_pos[0], goal_pos[1], 'bo', markersize=10)
    
    plt.title("A* Path Planning")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.colorbar(label="Obstacle")
    plt.savefig("turtlebot_astar_path.png")
    print("Path visualization saved to turtlebot_astar_path.png")
    plt.close()


def move_robot_along_path(robot, path, bounds, resolution, env, speed=0.2, turn_speed=0.5):
    """
    Move the robot along the path.
    
    Args:
        robot: OmniGibson robot
        path: Path from start to goal (list of tuples of grid positions)
        bounds: Tuple of (min_x, min_y, max_x, max_y) in world coordinates
        resolution: Grid resolution
        env: OmniGibson environment
        speed: Robot forward speed (m/s)
        turn_speed: Robot turning speed (rad/s)
    """
    if path is None or len(path) < 2:
        print("Path is too short to follow!")
        return
    
    print(f"Following path with {len(path)} waypoints...")
    
    # Get initial robot position
    robot_pos = robot.get_position()[:2]
    
    # Convert path from grid to world coordinates
    world_path = []
    for grid_pos in path:
        world_pos = grid_to_world(grid_pos, bounds, resolution)
        world_path.append(world_pos)
    
    # Follow the path
    for i, waypoint in enumerate(world_path):
        print(f"Moving to waypoint {i+1}/{len(world_path)}: {waypoint}")
        
        # Calculate direction to waypoint
        dx = waypoint[0] - robot_pos[0]
        dy = waypoint[1] - robot_pos[1]
        target_angle = np.arctan2(dy, dx)
        
        # Get current robot orientation
        robot_quat = robot.get_orientation()
        robot_angle = np.arctan2(2.0 * (robot_quat[3] * robot_quat[2] + robot_quat[0] * robot_quat[1]),
                                1.0 - 2.0 * (robot_quat[1] ** 2 + robot_quat[2] ** 2))
        
        # Calculate angle difference (within -π to π)
        angle_diff = np.arctan2(np.sin(target_angle - robot_angle), 
                              np.cos(target_angle - robot_angle))
        
        # Turn to face waypoint
        steps = 0
        while abs(angle_diff) > 0.1 and steps < 100:
            # Calculate angular velocity
            angular_velocity = turn_speed * np.sign(angle_diff) * min(1.0, abs(angle_diff) / np.pi)
            
            # Set action (differential drive with left and right wheel velocity)
            # Left wheel = angular_velocity, Right wheel = -angular_velocity for turning
            env.step(action=[angular_velocity, -angular_velocity])
            
            time.sleep(0.01)
            
            # Update robot angle
            robot_quat = robot.get_orientation()
            robot_angle = np.arctan2(2.0 * (robot_quat[3] * robot_quat[2] + robot_quat[0] * robot_quat[1]),
                                   1.0 - 2.0 * (robot_quat[1] ** 2 + robot_quat[2] ** 2))
            
            # Recalculate angle difference
            angle_diff = np.arctan2(np.sin(target_angle - robot_angle), 
                                  np.cos(target_angle - robot_angle))
            
            steps += 1
        
        # Move to waypoint
        dist = np.hypot(dx, dy)
        steps = 0
        while dist > 0.1 and steps < 500:
            # Calculate direction to waypoint
            robot_pos = robot.get_position()[:2]
            
            # Convert tensor to numpy array if needed
            if hasattr(robot_pos, 'cpu'):  # Check if it's a tensor
                robot_pos = robot_pos.cpu().numpy()
                
            dx = waypoint[0] - robot_pos[0]
            dy = waypoint[1] - robot_pos[1]
            dist = np.hypot(dx, dy)
            
            # Calculate linear velocity (slow down as we approach target)
            linear_velocity = speed * min(1.0, dist)
            
            # Set action (both wheels at same velocity for forward movement)
            env.step(action=[linear_velocity, linear_velocity])
            
            time.sleep(0.01)
            
            steps += 1
        
        # Stop robot
        env.step(action=[0.0, 0.0])
        robot_pos = robot.get_position()[:2]
        
        # Convert tensor to numpy array if needed
        if hasattr(robot_pos, 'cpu'):  # Check if it's a tensor
            robot_pos = robot_pos.cpu().numpy()
            
        print(f"Reached waypoint {i+1}: Current position: {robot_pos}")
    
    print("Path following complete!")


def add_marker(scene, position, radius=0.1, color=(0, 0, 1, 0.7), name="marker"):
    """
    Add a marker to the scene by directly placing a primitive object.
    This is a workaround since import_object is not available.
    """
    # We can't use import_object, so just print that we're marking the position
    print(f"Goal position marked at {position} (visualization only)")


def main():
    """
    Main function for turtlebot A* navigation using OmniGibson's built-in astar function.
    """
    # Create the environment
    print("Creating environment...")
    
    # Choose a simple empty scene
    scene_cfg = {"type": "InteractiveTraversableScene", "scene_model": "Rs_int"}
    
    # Add the Turtlebot robot with all modalities
    robot_cfg = {
        "type": "Turtlebot",
        "obs_modalities": ["rgb", "depth", "seg_semantic", "seg_instance"],
        "action_type": "continuous",
        "action_normalize": True,
    }
    
    # Compile config
    cfg = {
        "scene": scene_cfg,
        "robots": [robot_cfg],
        "env": {"action_timestep": 1 / 10.0, "physics_timestep": 1 / 120.0},
        "render": {"viewer_width": 1024, "viewer_height": 1024}
    }
    
    # Create the environment
    env = og.Environment(configs=cfg)
    
    # Get the robot
    robot = env.robots[0]
    
    # Initialize the camera for viewing
    og.sim.viewer_camera.set_position_orientation(
        position=np.array([3.0, 0.0, 2.0]),
        orientation=np.array([0.7071, 0.0, 0.0, 0.7071]),
    )
    
    # Wait for the scene to initialize
    for _ in range(10):
        env.step(action=[0.0, 0.0])  # Pass zero velocity to both wheels
    
    # Initialize robot starting position and goal position at sensible locations
    start_pos = robot.get_position()[:2]
    
    # Convert tensor to numpy array if needed
    if hasattr(start_pos, 'cpu'):  # Check if it's a tensor
        start_pos = start_pos.cpu().numpy()
    
    print(f"Robot starting position: {start_pos}")
    
    # Define goal position away from the robot
    # Try to place the goal in a clear area
    goal_pos = np.array([1.5, 1.5])
    print(f"Goal position: {goal_pos}")
    
    # Mark the goal position (visualization only)
    add_marker(env.scene, goal_pos, name="goal_marker")
    
    # Create search map
    print("Creating search map...")
    resolution = 0.05  # Grid resolution (5cm)
    search_map, bounds = create_search_map(env.scene, start_pos, resolution)
    
    # Convert start and goal to grid coordinates as tuples
    start_grid = world_to_grid(start_pos, bounds, resolution)
    goal_grid = world_to_grid(goal_pos, bounds, resolution)
    
    print(f"Start grid position: {start_grid}, Goal grid position: {goal_grid}")
    
    # Check if start or goal are outside the map or in collision
    if (start_grid[0] < 0 or start_grid[0] >= search_map.shape[0] or 
        start_grid[1] < 0 or start_grid[1] >= search_map.shape[1]):
        print("Start position is outside the map!")
        env.close()
        return
        
    if (goal_grid[0] < 0 or goal_grid[0] >= search_map.shape[0] or 
        goal_grid[1] < 0 or goal_grid[1] >= search_map.shape[1]):
        print("Goal position is outside the map!")
        env.close()
        return
    
    if search_map[start_grid[0], start_grid[1]] == 1:
        print("Start position is in collision!")
        env.close()
        return
        
    if search_map[goal_grid[0], goal_grid[1]] == 1:
        print("Goal position is in collision!")
        env.close()
        return
    
    # Plan path using OmniGibson's built-in A* algorithm
    # Make sure to pass tuples, not numpy arrays
    print("Planning path using OmniGibson's astar function...")
    path = astar(search_map, start_grid, goal_grid, eight_connected=True)
    
    if path is not None and len(path) > 0:
        print(f"Path found with {len(path)} waypoints!")
        
        # Visualize the path
        visualize_path(search_map, path, start_grid, goal_grid)
        
        # Move the robot along the path
        move_robot_along_path(robot, path, bounds, resolution, env)
    else:
        print("No path found!")
    
    # Keep the environment running for a while
    print("Press Ctrl+C to exit...")
    try:
        while True:
            env.step(action=[0.0, 0.0])  # Pass zero velocity to both wheels
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Exiting...")
    
    # Close the environment
    env.close()


if __name__ == "__main__":
    main()