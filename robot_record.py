"""
Robot Action Recorder for OmniGibson
Records robot actions at 60Hz frequency during keyboard teleoperation.
The recorded actions can be replayed later in robotview_generate.py for data collection.

Controls:
- R: Start/Stop recording
- ESC: Exit
- Standard robot teleop keys (see printed info)
"""

import torch as th
import json
import time
import os
import numpy as np

import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.macros import gm
from omnigibson.utils.ui_utils import KeyboardRobotController

# Don't use GPU dynamics and use flatcache for performance boost
gm.USE_GPU_DYNAMICS = False
gm.ENABLE_FLATCACHE = True

# Recording parameters
RECORDING_FREQUENCY = 60.0  # Hz
RECORDING_INTERVAL = 1.0 / RECORDING_FREQUENCY  # seconds

class ActionRecorder:
    """
    Records robot actions at a specified frequency with timestamps.
    """
    
    def __init__(self, output_dir="action_recordings"):
        self.output_dir = output_dir
        self.recording = False
        self.actions = []
        self.start_time = None
        self.last_record_time = 0
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
    def start_recording(self):
        """Start recording actions."""
        if not self.recording:
            self.recording = True
            self.actions = []
            self.start_time = time.time()
            self.last_record_time = self.start_time
            print(f"\n🔴 RECORDING STARTED at {time.strftime('%H:%M:%S')}")
            print("Move the robot with keyboard controls...")
        
    def stop_recording(self):
        """Stop recording and save actions to file."""
        if self.recording:
            self.recording = False
            duration = time.time() - self.start_time
            
            print(f"\n⏹️  RECORDING STOPPED")
            print(f"Duration: {duration:.2f} seconds")
            print(f"Total actions: {len(self.actions)}")
            print(f"Average frequency: {len(self.actions)/duration:.1f} Hz")
            
            # Save to file
            self.save_recording()
            
    def should_record_now(self):
        """Check if enough time has passed to record the next action."""
        current_time = time.time()
        return (current_time - self.last_record_time) >= RECORDING_INTERVAL
    
    def record_action(self, action):
        """Record an action with timestamp if recording is active."""
        if self.recording and self.should_record_now():
            current_time = time.time()
            relative_time = current_time - self.start_time
            
            # Convert action to list if it's a numpy array or tensor
            if hasattr(action, 'tolist'):
                action_list = action.tolist()
            elif hasattr(action, 'cpu'):
                action_list = action.cpu().numpy().tolist()
            else:
                action_list = list(action)
            
            # Record action with timestamp
            action_record = {
                "timestamp": relative_time,
                "action": action_list
            }
            
            self.actions.append(action_record)
            self.last_record_time = current_time
            
            # Visual feedback every second
            if len(self.actions) % RECORDING_FREQUENCY == 0:
                print(f"🔴 Recording: {len(self.actions)} actions ({relative_time:.1f}s)", end="\r")
    
    def save_recording(self):
        """Save recorded actions to JSON file."""
        if not self.actions:
            print("No actions to save!")
            return
            
        # Create filename with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"robot_actions_{timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        # Prepare data for saving
        recording_data = {
            "recording_info": {
                "frequency_hz": RECORDING_FREQUENCY,
                "duration_seconds": self.actions[-1]["timestamp"] if self.actions else 0,
                "total_actions": len(self.actions),
                "robot_type": "Fetch",
                "action_space_size": len(self.actions[0]["action"]) if self.actions else 0,
                "action_description": {
                    "0": "forward/backward movement",
                    "1": "left/right turn", 
                    "2": "head camera left/right rotation",
                    "3": "head camera up/down rotation",
                    "4-12": "arm and gripper controls"
                }
            },
            "actions": self.actions
        }
        
        # Save to JSON
        with open(filepath, 'w') as f:
            json.dump(recording_data, f, indent=2)
        
        print(f"💾 Actions saved to: {filepath}")
        
    def toggle_recording(self):
        """Toggle recording state."""
        if self.recording:
            self.stop_recording()
        else:
            self.start_recording()


def main():
    """
    Robot action recording demo with keyboard teleoperation.
    """
    print("="*60)
    print("ROBOT ACTION RECORDER")
    print("="*60)
    print("This script records robot actions at 60Hz for later replay.")
    print("Controls:")
    print("  R: Start/Stop Recording")
    print("  ESC: Exit")
    print("="*60)
    
    # Scene configuration
    scene_cfg = {
        "type": "InteractiveTraversableScene",
        "scene_model": "Rs_int"
    }
    
    # Robot configuration - same as robot_control_example.py
    robot_cfg = {
        "type": "Fetch",
        "obs_modalities": ["rgb"],
        "action_type": "continuous", 
        "action_normalize": True
    }
    
    # Environment configuration
    cfg = {
        "scene": scene_cfg,
        "robots": [robot_cfg]
    }
    
    # Create environment
    env = og.Environment(configs=cfg)
    robot = env.robots[0]
    
    # Set up robot controllers (same as robot_control_example.py)
    controller_config = {
        "base": {"name": "DifferentialDriveController"},
        "arm_0": {"name": "InverseKinematicsController"}, 
        "gripper_0": {"name": "MultiFingerGripperController"},
        "camera": {"name": "JointController"}
    }
    
    robot.reload_controllers(controller_config=controller_config)
    env.scene.update_initial_state()
    
    # Set camera position
    og.sim.viewer_camera.set_position_orientation(
        position=th.tensor([1.46949, -3.97358, 2.21529]),
        orientation=th.tensor([0.56829048, 0.09569975, 0.13571846, 0.80589577]),
    )
    
    # Reset environment
    env.reset()
    robot.reset()
    
    # Create action recorder
    recorder = ActionRecorder()
    
    # Create teleop controller
    action_generator = KeyboardRobotController(robot=robot)
    
    # Register recording toggle callback
    action_generator.register_custom_keymapping(
        key=lazy.carb.input.KeyboardInput.R,
        description="Start/Stop action recording",
        callback_fn=recorder.toggle_recording,
    )
    
    # Print keyboard teleop info
    action_generator.print_keyboard_teleop_info()
    
    print(f"\n🎮 Ready for teleoperation!")
    print(f"📊 Recording frequency: {RECORDING_FREQUENCY} Hz")
    print(f"💾 Output directory: {recorder.output_dir}")
    print(f"\nPress 'R' to start recording, then control the robot!")
    
    # Main control loop
    step = 0
    try:
        while True:
            # Get teleop action
            action = action_generator.get_teleop_action()
            
            # Record action if recording is active
            recorder.record_action(action)
            
            # Step environment
            env.step(action=action)
            step += 1
            
            # Print action for debugging (same format as mentioned)
            if step % 60 == 0:  # Print every second
                action_list = action.tolist() if hasattr(action, 'tolist') else list(action)
                print(f"\nAction: {action_list}")
                
    except KeyboardInterrupt:
        print("\n\n⏹️  Interrupted by user")
        
    finally:
        # Save any ongoing recording
        if recorder.recording:
            recorder.stop_recording()
        
        # Cleanup
        og.clear()
        print("👋 Recording session ended. Files saved in:", recorder.output_dir)


if __name__ == "__main__":
    main()