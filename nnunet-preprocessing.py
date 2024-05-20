# run in command line: 

# for both images and labels, run
# for file in *.png; do convert "$file" -colorspace Gray "$file"; done

# then run thresholding on the labels (in helper.py)

# then, for both images and labels, run
# mogrify -resize 512x512! *.png

# export nnUNet_raw="/home/sandbox/dtank/my-scratch/data/nnunet/nnUNet_raw"
# export nnUNet_preprocessed="/home/sandbox/dtank/my-scratch/data/nnunet/nnUNet_preprocessed"
# export nnUNet_results="/home/sandbox/dtank/my-scratch/data/nnunet/nnUNet_results"

# nnUNetv2_plan_and_preprocess -d 1 2 3 --verify_dataset_integrity -c 2d
