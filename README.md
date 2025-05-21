# OmniGibson Dataset Generator

This project leverages the [OmniGibson](https://github.com/StanfordVL/OmniGibson) framework to generate multi-modal datasets for robotics and computer vision applications. It captures RGB images, depth maps, semantic segmentation, instance segmentation, and camera trajectory data from simulated 3D environments.

## Overview

OmniGibson is a robotics simulation platform developed by Stanford Vision and Learning Lab for realistic embodied AI research. This project extends OmniGibson's capabilities to create custom datasets with various sensor modalities.

## Features

- **RGB Image Generation**: Captures high-resolution (1024x1024) RGB frames of the simulated environment
- **Depth Map Generation**: Creates depth maps with both visual representations and raw numerical data
- **Semantic Segmentation**: Generates semantic class labels for each pixel with mapping to object categories
- **Instance Segmentation**: Provides unique instance IDs for each object in the scene
- **Camera Trajectory Recording**: Captures camera extrinsic matrices for each frame in a sequence
- **Scene Modification**: Adds and positions objects in scenes (like placing an apple on a table)
- **Robot View Dataset**: Captures data from a robot's perspective as it moves through the environment

## Main Scripts

- `scene_modify.py`: Modifies a scene by adding objects and captures multi-modal data
- `dataset_ramdom.py`: Generates a complete dataset with randomized camera positions
- `dataset_traj.py`: Generates a dataset following a pre-recorded camera trajectory
- `robotview_generate.py`: Captures data from a robot's perspective
- `traj_generate.py`: Interactive tool for recording custom camera trajectories
- `utils.py`: Contains shared utility functions for all scripts

## Script Details

### Dataset Generation
- **dataset_ramdom.py**: Generates multi-modal dataset with randomized camera positions throughout the scene. Creates organized output with RGB images, depth maps, segmentation data, and camera parameters.
- **dataset_traj.py**: Generates a complete dataset by following a pre-recorded camera trajectory path from a JSON file.

### Scene Modification
- **scene_modify.py**: Adds and positions objects in scenes (like placing objects on surfaces) and captures multi-modal data including RGB, depth, semantic and instance segmentation.

### Robot View Simulation
- **robotview_generate.py**: Simulates a robot's view by capturing data from a robot's perspective as it moves through the environment.

### Camera Trajectory Tools
- **traj_generate.py**: Interactive tool that allows users to record custom camera trajectories by navigating through scenes. Press 'R' to start/stop recording and ESC to exit.

### Utilities
- **utils.py**: Contains shared functions for directory management, data processing, visualization, and other common operations used across all scripts.

## Usage

1. Install OmniGibson following the [official installation guide](https://github.com/StanfordVL/OmniGibson)
2. Run the scripts based on your needs:

```bash
# Generate a dataset with random camera positions
python dataset_ramdom.py

# Generate a dataset following a pre-recorded camera trajectory
python dataset_traj.py

# Modify a scene by adding objects and capture multi-modal data
python scene_modify.py

# Capture data from a robot's perspective
python robotview_generate.py

# Record custom camera trajectories interactively
python traj_generate.py
```

## Output Structure

The scripts create organized output directories for each data type:

- `dataset_output/`: Complete dataset with all modalities
- `multimodal_output/`: Multi-modal scene data
- `robot_dataset_output/`: Robot perspective data
- `scene_output/`: Scene modification outputs
- `trajectory_data/`: Saved camera trajectory files

## Dependencies

- OmniGibson
- NumPy
- PyTorch
- PIL (Python Imaging Library)
- OpenCV (for video creation)
- Matplotlib (for visualization)

## Acknowledgments

This project builds upon the OmniGibson framework developed by the Stanford Vision and Learning Lab.