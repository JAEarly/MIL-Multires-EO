
# Extending Scene-to-Patch Models: Multi-Resolution Multiple Instance Learning for Earth Observation

This repo contains the code for the paper "Extending Scene-to-Patch Models: Multi-Resolution
Multiple Instance Learning for Earth Observation".  
We use [Bonfire](https://github.com/JAEarly/Bonfire) for the backend MIL functionality.

![main_img](./out/interpretability/FloodNet/multi_res/311_interpretability.png)


Below we break down each of the directories in this repo:

### Config

Configuration for model parameters.

### Models

Contains the trained model files. Five repeats per configuration.

### Out

Interpretability outputs and other figures that are used in the paper

### Results

Raw results for our experiments: scene-level RSME and MAE, patch-level mIoU, and pixel-level mIoU.

### Scripts

Contains our executable scripts. These are the entry points to our experiments, and should be run from the root
of the repo.

### Src

Contains our project-specific code. This includes the dataset, high-level model implementations
(low-level model code is implemented in Bonfire), and interpretability studies.
