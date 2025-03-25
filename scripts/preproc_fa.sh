#!/bin/bash
## By Jongyeon Yoon and Elyssa McMaster
out_dir=$1
dmri_file=$2
json_file=$3  # Accept the JSON file as input

# Convert json file to bval/bvec files using the Python script
echo "Converting JSON to bval/bvec..."
python3 convert_json_to_bvalbvec.py "$json_file" "$out_dir/dmri.bval" "$out_dir/dmri.bvec"

# Extract bvec and bval file paths for later use
bvec_file="$out_dir/dmri.bvec"
bval_file="$out_dir/dmri.bval"

echo "bvec file: $bvec_file"
echo "bval file: $bval_file"

echo "<<<<<Obtaining FA map>>>>>"

echo "<<<<Preprocessing dMRI data>>>>"
echo "**It is recommended to use isotropic resolution**"
echo "*If your data is anisotropic, you can use the mrgrid command from MRTrix3 to get isotropic voxel spacing*"

# Convert .mha to .nii using SimpleITK
echo "Converting MHA to NIfTI..."
python3 convert_mha_to_nifti.py "$dmri_file" "$out_dir/dmri.nii.gz"

dmri_file="$out_dir/dmri.nii.gz"

# Skull Stripping with a stricter mask
echo "<<Skull Stripping>>"
skull_stripped_file="$out_dir/skull_stripp_dmri.nii.gz"
skull_mask_file="$out_dir/skull_stripp_dmri_mask.nii.gz"
skull_stripped_file_4d="$out_dir/skull_stripp_dmri_4d.nii.gz"

cmd="bet $dmri_file $skull_stripped_file -m -f 0.2"  # Lowering f makes the mask stricter
[ ! -f $skull_stripped_file ] && (echo $cmd && eval $cmd) || echo "Output exists, skipping!"

cmd="mrcalc $dmri_file $skull_mask_file -mult $skull_stripped_file_4d"
[ ! -f $skull_stripped_file_4d ] && (echo $cmd && eval $cmd) || echo "Output exists, skipping!"

# Ensure the correct masked dMRI file is used for downstream steps
dmri_file="$skull_stripped_file_4d"

echo "<<<<Extracting FA map from preprocessed dMRI data>>>>"

echo "<<Tensor Fitting>>"
tensor_file="$out_dir/tensor.mif"
cmd="dwi2tensor $dmri_file $tensor_file -fslgrad $bvec_file $bval_file"
[ ! -f $tensor_file ] && (echo $cmd && eval $cmd) || echo "Output exists, skipping!"

echo "<<FA Calculation>>"
fa_map_mif="$out_dir/fa_map.mif"
fa_map_nii="$out_dir/fa_map.nii.gz"
cmd="tensor2metric $tensor_file -fa $fa_map_mif"
[ ! -f $fa_map_mif ] && (echo $cmd && eval $cmd) || echo "Output exists, skipping!"
cmd="mrconvert $fa_map_mif $fa_map_nii"
[ ! -f $fa_map_nii ] && (echo $cmd && eval $cmd) || echo "Output exists, skipping!"

echo "<<Generating Brain Mask>>"
brain_mask="$skull_mask_file"  # Use the mask from BET instead of dwi2mask
if [ ! -f "$brain_mask" ]; then
    echo "Brain mask not found! Creating a new one..."
    cmd="dwi2mask $dmri_file $brain_mask"
    (echo $cmd && eval $cmd)
else
    echo "Brain mask exists, skipping!"
fi

echo "<<Applying Brain Mask to FA Map>>"
masked_fa_map="$out_dir/fa_map_masked.nii.gz"
cmd="mrcalc $fa_map_nii $brain_mask -mult $masked_fa_map"

if [ ! -f "$masked_fa_map" ]; then
    echo $cmd && eval $cmd
    if [ $? -ne 0 ]; then
        echo "mrcalc failed! Trying fslmaths instead..."
        fslmaths $fa_map_nii -mas $brain_mask $masked_fa_map
    fi
else
    echo "Masked FA map exists, skipping!"
fi

echo "<<<Cleaning up intermediate files>>>"

rm -f $tensor_file
rm -f $fa_map_mif
rm -f $out_dir/corrected_dmri.nii.gz.par
echo "<<<<<Obtaining FA map finished>>>>>"





