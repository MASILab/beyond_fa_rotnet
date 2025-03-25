#!/bin/bash
# Read dwi from inputs/ and write rotnet inference to outputs/

# Fucking pytorch
# echo "Checking Python environment..."
# echo "Python version:"
# python3 --version
# echo "Python path:"
# python3 -c "import sys; print('\n'.join(sys.path))"
# echo "Current working directory:"
# pwd
# echo "Contents of current directory:"
# ls -la
# echo "Contents of /usr/local/lib/python3.10/site-packages:"
# ls -la /usr/local/lib/python3.10/site-packages || echo "Directory not found"
# echo "Contents of $HOME/.local/lib/python3.10/site-packages:"
# ls -la $HOME/.local/lib/python3.10/site-packages || echo "Directory not found"

# # Try importing torch
# echo "Attempting to import torch..."
# python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')"

# Create tmp directory for intermediate files
tmp_dir="/tmp/beyond_fa"
mkdir -p $tmp_dir

echo "Running BeyondFA baseline..."
echo "Listing /input..."
ls /input
echo "Listing /input/*..."
ls /input/*
echo "Listing /output..."
ls /output/

# Find the MHA file
dwi_mha_file=$(find "/input/images/dwi-4d-brain-mri" -name "*.mha" -type f | head -n 1)
if [ -z "$dwi_mha_file" ]; then
    echo "Error: No MHA file found in /input/images/dwi-4d-brain-mri"
    exit 1
fi

echo "Found MHA file: ${dwi_mha_file}"
json_file="/input/dwi-4d-acquisition-metadata.json"

# Run FA preprocessing in tmp directory
echo "Running FA preprocessing..."
bash preproc_fa.sh "$tmp_dir" "$dwi_mha_file" "$json_file"

# The FA map will be saved as fa_map_masked.nii.gz in the tmp directory
echo "FA preprocessing complete. Output saved to ${tmp_dir}/fa_map_masked.nii.gz"

# Run inference
echo "Running inference..."
python3 inference.py \
    --model_weights "final_model.pth" \
    --fa_map "${tmp_dir}/fa_map_masked.nii.gz" \
    --output_json "/output/features-128.json"

echo "Inference complete. Output saved to /output/features-128.json"

# Clean up tmp directory
rm -rf $tmp_dir

