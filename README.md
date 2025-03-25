# RotNet-FA-Image

## Overview

RotNet-FA-Image is a Python-based deep learning project for predicting rotation angles of fractional anisotropy (FA) maps derived from NIfTI (.nii.gz) files. It leverages 3D convolutional neural networks (CNNs) implemented in PyTorch to regress rotation angles. The augmented data helps us extract meaningful features from the input data.

Original RotNet repo: https://github.com/d4nst/RotNet

## Environment
- Ubuntu version: 22.04.4 LTS 
- CUDA version: 12.2
- GPU: NVIDIA RTX A4000

## Building the Docker

The Docker container is set up using the provided `Dockerfile`. To build the Docker container, clone the repository and run the following command in the root directory:

```bash
DOCKER_BUILDKIT=1 sudo docker build -t beyondfa_vine:v1.1.3 .
```

The Docker runs the code from `scripts/entrypoint.sh`.

## Running the Docker

Your Docker container reads input data from `/input` and writes output data to `/output`. The input should be a .mha image and a .json file containing bvals/bvecs information.

To run this Docker:

```bash
input_dir=".../input_data"
output_dir=".../output_data"

mkdir -p $output_dir
sudo chmod 777 $output_dir

sudo docker run --rm \
    -v $input_dir:/input \
    -v $output_dir:/output \
    beyondfa_vine:v1.1.3
```

## Local Development Setup
```
# Create and activate a new conda environment
conda create -n isbi_challenge python=3.10 -y
conda activate isbi_challenge

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Install additional dependencies
pip3 install -r requirements.txt
```

## Preprocessing Script (preproc_fa.sh)
- Assumes [PreQual](https://github.com/MASILab/PreQual/) has already been run on input data -- we recommend that all challenge participants do this, as the testing data will be preprocessed this way
- Converts the .mha/.json data available from grand-challenge.org to .nii.gz/.bval/.bvec format that the preprocessing steps need
- Skull stripping
- Tensor fitting
- FA map generation

These steps may or may not be necessary based on your approach.
Though PreQual outputs include FA maps, your Docker needs to include any steps beyond preprocessing so that the training/testing data can be generated in the same way.

## Project Structure
- `NiiDataset`: Custom PyTorch Dataset class for loading and preprocessing NIfTI files.
- `RotationRegressionModel`: 3D CNN for predicting rotation angles.
- `Training Script`: Handles the training process, model evaluation, and visualization.

## Running the Model
- Prepare your dataset of .nii.gz files and place them in the data directory. (default: `./data`)
- Train the model using the provided training script:
```python
python3 rotnet3D_regression.py
```
- The model's training progress will be logged, and visualization outputs will be saved in the result_regression directory.

## Outputs
- Useful Feature: Saved as .npy array
- Training History: Plots of the mean squared error (MSE) loss over epochs.
- Inference converts vectors to .json file so that we can upload them to our MLP testing code
