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

- `scene_modify.py`: Modifies a scene by adding objects (like an apple) and captures multi-modal data
- `dataset_generate.py`: Generates a complete dataset with random camera trajectories
- `robotview_generate.py`: Captures data from a robot's perspective

## Separate Module Scripts

The `separate` directory contains modular scripts that focus on specific data types:

- `rgb_generate.py`: Generates single RGB frames from the simulated environment
- `depth_generate.py`: Creates depth maps with visualization and raw depth values
- `seg_generate.py`: Produces semantic and instance segmentation data with mappings
- `traj_generate.py`: Records camera trajectories while capturing a sequence of frames
- `scene_modify_rgb.py`: Simplified version that only captures RGB data after scene modification

## Usage

1. Install OmniGibson following the [official installation guide](https://github.com/StanfordVL/OmniGibson)
2. Run the scripts based on your needs:

```bash
# Generate a complete multi-modal dataset
python dataset_generate.py

# Modify a scene and capture multi-modal data
python scene_modify.py

# Capture data from a robot's perspective
python robotview_generate.py

# Or use the separate scripts for specific modalities
python separate/rgb_generate.py
python separate/depth_generate.py
python separate/seg_generate.py
python separate/traj_generate.py
```

## Output Structure

The scripts create organized output directories for each data type:

- `dataset_output/`: Complete dataset with all modalities
- `multimodal_output/`: Multi-modal scene data
- `robot_dataset_output/`: Robot perspective data
- `scene_output/`: Scene modification outputs

## Dependencies

- OmniGibson
- NumPy
- PyTorch
- PIL (Python Imaging Library)
- OpenCV (for video creation)
- Matplotlib (for visualization)

## Acknowledgments

This project builds upon the OmniGibson framework developed by the Stanford Vision and Learning Lab.