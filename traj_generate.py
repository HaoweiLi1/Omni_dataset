import os
import json
import time
import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.macros import gm
from omnigibson.utils.asset_utils import get_available_g_scenes, get_available_og_scenes
from omnigibson.utils.ui_utils import KeyboardEventHandler, choose_from_options


def main(random_selection=False, headless=False, short_exec=False):
    """
    Loads an interactive scene and allows camera trajectory recording.
    
    Controls:
    - R: Start/stop recording
    - ESC: Exit
    """
    og.log.info(f"Demo {__file__}\n    " + "*" * 80 + "\n    Description:\n" + main.__doc__ + "*" * 80)

    # Make sure the example is not being run headless
    if gm.HEADLESS:
        print("This demo should only be run not headless! Exiting early.")
        og.shutdown()

    # Choose scene type and model
    scene_options = {
        "InteractiveTraversableScene": "Procedurally generated scene with fully interactive objects",
    }
    scene_type = choose_from_options(options=scene_options, name="scene type", random_selection=random_selection)
    scenes = get_available_og_scenes() if scene_type == "InteractiveTraversableScene" else get_available_g_scenes()
    scene_model = choose_from_options(options=scenes, name="scene model", random_selection=random_selection)
    print(f"scene model: {scene_model}")

    # Configure environment
    cfg = {
        "scene": {
            "type": scene_type,
            "scene_model": scene_model,
        },
    }

    # Quick/Full load option
    if scene_type == "InteractiveTraversableScene":
        load_options = {
            "Quick": "Only load the building assets (i.e.: the floors, walls, ceilings)",
            "Full": "Load all interactive objects in the scene",
        }
        load_mode = choose_from_options(options=load_options, name="load mode", random_selection=random_selection)
        if load_mode == "Quick":
            cfg["scene"]["load_object_categories"] = ["floors", "walls", "ceilings"]

    # Load environment
    env = og.Environment(configs=cfg)

    # Enable camera teleoperation with default parameters
    cam_mover = og.sim.enable_viewer_camera_teleoperation()
    
    # Create output directory
    output_dir = "custom_trajectories"
    os.makedirs(output_dir, exist_ok=True)

    # Recording state variables
    recording = False
    trajectory = []
    frame_rate = 30.0  # Frames per second to capture
    last_capture_time = 0
    
    def toggle_recording():
        nonlocal recording, trajectory, last_capture_time
        
        if not recording:
            # Start recording
            recording = True
            trajectory = []
            last_capture_time = time.time()
            print("Recording started! Move camera SLOWLY for best results.")
        else:
            # Stop recording
            recording = False
            print(f"\nRecording stopped. Captured {len(trajectory)} poses.")
            
            if len(trajectory) > 0:
                # Save trajectory
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                filename = f"trajectory_{timestamp}.json"
                filepath = os.path.join(output_dir, filename)
                
                # Convert trajectory to serializable format
                serializable_traj = []
                for pose in trajectory:
                    pos, ori = pose
                    serializable_traj.append({
                        "position": pos.tolist(),
                        "orientation": ori.tolist()
                    })
                
                # Save to file
                with open(filepath, 'w') as f:
                    json.dump(serializable_traj, f, indent=2)
                
                print(f"Trajectory saved to {filepath}")
    
    # Register keyboard callbacks
    KeyboardEventHandler.initialize()
    KeyboardEventHandler.add_keyboard_callback(
        key=lazy.carb.input.KeyboardInput.R,
        callback_fn=toggle_recording,
    )
    KeyboardEventHandler.add_keyboard_callback(
        key=lazy.carb.input.KeyboardInput.ESCAPE,
        callback_fn=lambda: og.clear(),
    )

    # Print instructions
    print("\n" + "=" * 50)
    print("CAMERA TRAJECTORY RECORDER")
    print("=" * 50)
    print("  R: Start/Stop Recording")
    print("  ESC: Exit")
    print("\nTIP: Move the camera VERY SLOWLY to get a smooth trajectory.")
    print("     This is critical for good 3D reconstruction results.")
    print("=" * 50 + "\n")

    # Main loop
    steps = 0
    max_steps = -1 if not short_exec else 100
    while steps != max_steps:
        # Step the environment
        env.step([])
        
        # If recording, capture camera pose at the target frame rate
        if recording:
            current_time = time.time()
            if current_time - last_capture_time >= 1.0 / frame_rate:
                # Get camera pose
                pos, ori = cam_mover.cam.get_position_orientation()
                trajectory.append((pos, ori))
                last_capture_time = current_time
                
                # Visual feedback
                if len(trajectory) % 30 == 0:  # Once per second
                    print(f"● Recording: {len(trajectory)} frames ({len(trajectory)/frame_rate:.1f}s)", end="\r")
                else:
                    print(f"○ Recording: {len(trajectory)} frames ({len(trajectory)/frame_rate:.1f}s)", end="\r")
        
        steps += 1


if __name__ == "__main__":
    main()