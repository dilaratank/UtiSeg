"""
nnunet-preprocessing.py

Steps to take to preprocess the data before using it with the nnU-Net framework
"""

# (1)
# go to dataset folder and open terminal,
# run in command line, for both images and labels: 

### for file in *.png; do convert "$file" -colorspace Gray "$file"; done ###

# (2)
# then run thresholding on the labels (in helper.py)

# (3)
# then, for both images and labels, run in command line terminal
### mogrify -resize 512x512! *.png ###

# (4)
# Set the following variables
# export nnUNet_raw="/home/sandbox/dtank/my-scratch/data/nnunet/nnUNet_raw"
# export nnUNet_preprocessed="/home/sandbox/dtank/my-scratch/data/nnunet/nnUNet_preprocessed"
# export nnUNet_results="/home/sandbox/dtank/my-scratch/data/nnunet/nnUNet_results"

# (5)
# Run dataset planning and processing
### nnUNetv2_plan_and_preprocess -d 1 2 3 --verify_dataset_integrity -c 2d ###

# (6)
# Run code to reverse threshold

# reverse threshold
# image[image > 0] = 255
# io.imsave('/home/sandbox/dtank/my-scratch/data/nnunet/nnUNet_raw/Dataset001_STILL/labelsTr/STILL_122-check.png', image)

