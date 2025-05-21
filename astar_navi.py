import numpy as np
import time
import heapq
import matplotlib.pyplot as plt

import omnigibson as og
from omnigibson.macros import gm
from omnigibson.utils.asset_utils import download_key

# ======== Configuration ========
# Set headless mode and download key
gm.HEADLESS = False  # Set to True for headless mode (no visualization)
download_key()

# Don't use GPU dynamics and use flatcache for performance boost
gm.USE_GPU_DYNAMICS = False
gm.ENABLE_FLATCACHE = True

# ======== A* Path Planning Implementation ========
class Node:
    """
    Node class for A* path planning
    """
    def __init__(self, x, y, cost, parent_index):
        self.x = x  # grid x position
        self.y = y  # grid y position
        self.cost = cost  # cost from start
        self.parent_index = parent_index  # parent node index
        
    def __str__(self):
        return str(self.x) + "," + str(self.y)
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y


def calculate_heuristic(node1, node2):
    """
    Calculate heuristic distance between two nodes using Euclidean distance
    """
    return np.hypot(node1.x - node2.x, node1.y - node2.y)


def get_motion_model():
    """
    Get motion model for 2D grid navigation
    dx, dy, cost
    """
    # 8-connected grid (including diagonals)
    # fmt: off
    motion = [
        [1, 0, 1],         # right
        [0, 1, 1],         # up
        [-1, 0, 1],        # left
        [0, -1, 1],        # down
        [1, 1, np.sqrt(2)],  # up-right
        [-1, 1, np.sqrt(2)], # up-left
        [-1, -1, np.sqrt(2)], # down-left
        [1, -1, np.sqrt(2)],  # down-right
    ]
    # fmt: on
    return motion


def verify_node(node, grid_map, robot_radius):
    """
    Verify if the node is valid (within bounds and not in collision)
    """
    if node.x < 0 or node.y < 0:
        return False
    
    if node.x >= grid_map.shape[0] or node.y >= grid_map.shape[1]:
        return False
    
    # Check if node is in collision (1 = obstacle)
    # For safety, check a neighborhood around the node
    x_range = range(max(0, int(node.x - robot_radius)), 
                    min(grid_map.shape[0], int(node.x + robot_radius + 1)))
    y_range = range(max(0, int(node.y - robot_radius)), 
                    min(grid_map.shape[1], int(node.y + robot_radius + 1)))
    
    for i in x_range:
        for j in y_range:
            if grid_map[i, j] == 1:
                return False
    
    return True


def astar(start_pos, goal_pos, grid_map, resolution, robot_radius):
    """
    A* path planning algorithm
    
    Args:
        start_pos: Start position (x, y) in world coordinates
        goal_pos: Goal position (x, y) in world coordinates
        grid_map: Grid map (2D numpy array where 0 = free, 1 = obstacle)
        resolution: Grid map resolution (cell size in meters)
        robot_radius: Robot radius (in grid cells)
        
    Returns:
        path: Path from start to goal (list of (x, y) coordinates in world space)
    """
    # Convert start and goal positions to grid coordinates
    start_x = int(start_pos[0] / resolution)
    start_y = int(start_pos[1] / resolution)
    goal_x = int(goal_pos[0] / resolution)
    goal_y = int(goal_pos[1] / resolution)
    
    print(f"Grid coordinates - Start: ({start_x}, {start_y}), Goal: ({goal_x}, {goal_y})")
    
    # Check if start or goal are in collision
    if (start_x < 0 or start_x >= grid_map.shape[0] or 
        start_y < 0 or start_y >= grid_map.shape[1] or
        grid_map[start_x, start_y] == 1):
        print("Start position is in collision or outside map!")
        return None
        
    if (goal_x < 0 or goal_x >= grid_map.shape[0] or 
        goal_y < 0 or goal_y >= grid_map.shape[1] or
        grid_map[goal_x, goal_y] == 1):
        print("Goal position is in collision or outside map!")
        return None
    
    # Create start and goal nodes
    start_node = Node(start_x, start_y, 0.0, -1)
    goal_node = Node(goal_x, goal_y, 0.0, -1)
    
    # Initialize open and closed lists
    open_set = {}
    closed_set = {}
    open_set[start_node.__str__()] = start_node
    
    # Create a priority queue for open nodes
    pq = []
    heapq.heappush(pq, (0, start_node.__str__()))
    
    # Get motion model
    motion = get_motion_model()
    
    # A* search
    while len(open_set) > 0:
        # Get node with lowest f_score
        _, current_key = heapq.heappop(pq)
        
        # Check if current_key is still in open_set (might have been removed)
        if current_key not in open_set:
            continue
            
        current = open_set[current_key]
        
        # Check if we've reached the goal
        if current.x == goal_node.x and current.y == goal_node.y:
            print("Goal reached!")
            # Reconstruct path
            path = []
            while current.parent_index != -1:
                # Convert back to world coordinates for the path
                path.append((current.x * resolution, current.y * resolution))
                current = closed_set[current.parent_index]
            path.append((start_pos[0], start_pos[1]))  # Add start position
            path.reverse()  # Reverse to get path from start to goal
            return path
        
        # Remove current node from open set and add to closed set
        del open_set[current_key]
        closed_set[current_key] = current
        
        # Expand current node
        for dx, dy, cost in motion:
            next_x = current.x + dx
            next_y = current.y + dy
            next_node = Node(next_x, next_y, current.cost + cost, current_key)
            
            # Skip if node is already in closed set
            if next_node.__str__() in closed_set:
                continue
            
            # Skip if node is invalid (outside bounds or in collision)
            if not verify_node(next_node, grid_map, robot_radius):
                continue
            
            # If node is already in open set with lower cost, skip
            if next_node.__str__() in open_set:
                if next_node.cost >= open_set[next_node.__str__()].cost:
                    continue
            
            # Add node to open set
            open_set[next_node.__str__()] = next_node
            f_score = next_node.cost + calculate_heuristic(next_node, goal_node)
            heapq.heappush(pq, (f_score, next_node.__str__()))
    
    print("No path found!")
    return None


def create_occupancy_grid(scene, resolution=0.1, height=0.1):
    """
    Create an occupancy grid map from the scene
    
    Args:
        scene: OmniGibson scene
        resolution: Grid map resolution (cell size in meters)
        height: Height at which to check for obstacles
        
    Returns:
        grid_map: 2D numpy array where 0 = free, 1 = obstacle
        bounds: Tuple of (min_x, min_y, max_x, max_y) in world coordinates
    """
    try:
        # Get scene bounds
        bounds = scene.get_floor_bbox()
        min_x, min_y = bounds[0][:2]
        max_x, max_y = bounds[1][:2]
        
        print(f"Scene bounds: min_x={min_x}, min_y={min_y}, max_x={max_x}, max_y={max_y}")
        
        # Ensure bounds are valid
        if min_x >= max_x or min_y >= max_y:
            print("Warning: Invalid scene bounds! Using default bounds.")
            min_x, min_y = -5, -5
            max_x, max_y = 5, 5
    except Exception as e:
        print(f"Error getting scene bounds: {e}")
        print("Using default bounds.")
        min_x, min_y = -5, -5
        max_x, max_y = 5, 5
    
    # Calculate grid dimensions
    width = int((max_x - min_x) / resolution) + 1
    length = int((max_y - min_y) / resolution) + 1
    
    print(f"Grid dimensions: {width}x{length}")
    
    # Initialize grid map
    grid_map = np.zeros((width, length))
    
    # Mark obstacles in the grid map
    for obj in scene.objects:
        # Skip the robot itself and very small objects
        if "turtlebot" in obj.name.lower() or "goal_marker" in obj.name.lower():
            continue
        
        try:
            # Get object bounds
            obj_bounds = obj.get_aabb()
            obj_min_x, obj_min_y, obj_min_z = obj_bounds[0]
            obj_max_x, obj_max_y, obj_max_z = obj_bounds[1]
            
            # Skip objects that are too high or too low
            if obj_min_z > height + 0.5 or obj_max_z < height - 0.1:
                continue
            
            # Convert to grid coordinates
            grid_min_x = max(0, int((obj_min_x - min_x) / resolution))
            grid_min_y = max(0, int((obj_min_y - min_y) / resolution))
            grid_max_x = min(width - 1, int((obj_max_x - min_x) / resolution))
            grid_max_y = min(length - 1, int((obj_max_y - min_y) / resolution))
            
            # Mark as occupied
            grid_map[grid_min_x:grid_max_x+1, grid_min_y:grid_max_y+1] = 1
            
            print(f"Added obstacle: {obj.name} at ({grid_min_x},{grid_min_y}) to ({grid_max_x},{grid_max_y})")
        except Exception as e:
            print(f"Skipping object {obj.name}: {e}")
    
    return grid_map, (min_x, min_y, max_x, max_y)


def visualize_path(grid_map, path, start_pos, goal_pos, bounds, resolution):
    """
    Visualize the path on the grid map
    """
    if path is None:
        print("No path to visualize!")
        return
        
    # Convert path to grid coordinates
    min_x, min_y, _, _ = bounds
    grid_path = []
    for x, y in path:
        grid_x = int((x - min_x) / resolution)
        grid_y = int((y - min_y) / resolution)
        grid_path.append((grid_x, grid_y))
    
    # Convert start and goal to grid coordinates
    start_grid_x = int((start_pos[0] - min_x) / resolution)
    start_grid_y = int((start_pos[1] - min_y) / resolution)
    goal_grid_x = int((goal_pos[0] - min_x) / resolution)
    goal_grid_y = int((goal_pos[1] - min_y) / resolution)
    
    # Transpose the grid map for visualization (matplotlib's (0,0) is bottom-left)
    plt.figure(figsize=(10, 10))
    plt.imshow(grid_map.T, cmap='binary', origin='lower')
    
    # Plot the path
    if path:
        x_coords = [x for x, y in grid_path]
        y_coords = [y for x, y in grid_path]
        plt.plot(x_coords, y_coords, 'r-', linewidth=2)
    
    # Plot start and goal
    plt.plot(start_grid_x, start_grid_y, 'go', markersize=10)
    plt.plot(goal_grid_x, goal_grid_y, 'bo', markersize=10)
    
    plt.title("A* Path Planning")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.colorbar(label="Obstacle")
    plt.savefig("turtlebot_astar_path.png")
    print("Path visualization saved to turtlebot_astar_path.png")
    plt.close()


def move_robot_along_path(robot, path, env, speed=0.2, turn_speed=0.5):
    """
    Move the robot along the path
    
    Args:
        robot: OmniGibson robot
        path: Path from start to goal (list of (x, y) coordinates)
        env: OmniGibson environment
        speed: Robot forward speed (m/s)
        turn_speed: Robot turning speed (rad/s)
    """
    if not path or len(path) < 2:
        print("Path is too short to follow!")
        return
    
    print(f"Following path with {len(path)} waypoints...")
    
    # Get initial robot position
    robot_pos = robot.get_position()[:2]
    
    # Follow the path
    for i, waypoint in enumerate(path[1:]):  # Skip the first waypoint (current position)
        print(f"Moving to waypoint {i+1}/{len(path)-1}: {waypoint}")
        
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
        print(f"Reached waypoint {i+1}: Current position: {robot_pos}")
    
    print("Path following complete!")


def main():
    """
    Main function for turtlebot A* navigation
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
    
    # Add some obstacles to the scene for testing
    try:
        # Add a box obstacle
        box_cfg = {
            "object": {
                "type": "PrimitiveObject",
                "name": "obstacle_box",
                "primitive_type": "Box",
                "size": [0.5, 0.5, 0.5],
                "rgba": [1.0, 0.0, 0.0, 0.7],  # Red color
                "position": [0.8, 0.0, 0.25],
                "orientation": [0, 0, 0, 1],
            }
        }
        env.scene.import_object(box_cfg)
        print("Added box obstacle")
        
        # Add a cylinder obstacle
        cylinder_cfg = {
            "object": {
                "type": "PrimitiveObject",
                "name": "obstacle_cylinder",
                "primitive_type": "Cylinder",
                "radius": 0.2,
                "height": 0.5,
                "rgba": [0.0, 1.0, 0.0, 0.7],  # Green color
                "position": [0.0, 0.8, 0.25],
                "orientation": [0, 0, 0, 1],
            }
        }
        env.scene.import_object(cylinder_cfg)
        print("Added cylinder obstacle")
    except Exception as e:
        print(f"Error adding obstacles: {e}")
        print("Continuing without additional obstacles...")
    
    # Initialize robot starting position and goal position at sensible locations
    start_pos = robot.get_position()[:2]
    print(f"Robot starting position: {start_pos}")
    
    # Define goal position away from the robot
    # Try to place the goal in a clear area
    goal_pos = np.array([1.5, 1.5])
    print(f"Goal position: {goal_pos}")
    
    # Create a marker for the goal
    try:
        marker_cfg = {
            "object": {
                "type": "PrimitiveObject",
                "name": "goal_marker",
                "primitive_type": "Sphere",
                "radius": 0.1,
                "rgba": [0.0, 0.0, 1.0, 0.7],  # Blue color
                "position": [goal_pos[0], goal_pos[1], 0.1],
                "orientation": [0, 0, 0, 1],
            }
        }
        env.scene.import_object(marker_cfg)
        print("Goal marker created successfully")
    except Exception as e:
        print(f"Could not create goal marker: {e}")
        print("Continuing without goal marker...")
    
    # Create occupancy grid
    print("Creating occupancy grid...")
    resolution = 0.05  # Grid resolution (5cm)
    grid_map, bounds = create_occupancy_grid(env.scene, resolution)
    
    # Calculate robot radius in grid cells
    robot_radius = 0.2 / resolution  # Turtlebot radius is around 20cm
    
    # Plan path using A*
    print("Planning path...")
    path = astar(start_pos, goal_pos, grid_map, resolution, robot_radius)
    
    if path:
        print(f"Path found with {len(path)} waypoints!")
        
        # Visualize the path
        visualize_path(grid_map, path, start_pos, goal_pos, bounds, resolution)
        
        # Move the robot along the path
        move_robot_along_path(robot, path, env)
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