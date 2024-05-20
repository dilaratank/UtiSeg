# MScUtiSeg
MSc Thesis project: Uterus Segmentation on TVUS Dataset

## Introduction
Transvaginal ultrasound (TVUS) is pivotal for diagnosing pathologies related to the reproductive system in individuals assigned female at birth, often serving as the primary imaging modality for gynecologic symptoms evaluation. Despite recent advancements in AI-driven segmentation, such as nnU-Net for MRI scans, its application to gynaecological ultrasound images remains unexplored, highlighting a research gap in this domain. Our feasibility study aims to bridge this research gap by (1) creating a small-scale TVUS dataset of at least 100 patients for the purpose of automated uterus segmentation *, and (2) training and evaluating the performance of two state-of-the-art deep-learning segmentation models for the purpose of analyzing the feasibility of deep-learning-based segmentation on TVUS.

\* The dataset created for this study has not been made public, but consits of ±1000 TVUS images where the uterus of the patient is visible 

## Project Overview
- luna_scripts: Contains slurm scripts for the luna cluster to train and evaluate the U-Net and nnU-Net models
- dataloaders.py & datasets.py: Contains the classes and functions for loading and preparing the data for the U-Net training
- eval.py: Contains the python script for evaluating the trained U-Net models
- helper.py: A collection of helper functions for U-Net and nnU-Net data storing
- nnunet-preprocessing.py: A summary of steps to take to preprocess the data before using it with the nnU-Net framework
- train.py: Contains the python script for training the U-Net models
- visualization.ipynb: A Jupyter Notebook that contains visualizations of the various implemented pre-processing and augmentation methods used in the study 

## Environment & Packages
A list of the Python packages and their versions in the environment used to develop and test the code

Package                       | Version
----------------------------- | -----------
acvl-utils                    | 0.2
aiohttp                       | 3.9.3
aiosignal                     | 1.3.1
albumentations                | 1.4.1
appdirs                       | 1.4.4
asttokens                     | 2.4.1
async-timeout                 | 4.0.3
attrs                         | 23.2.0
batchgenerators               | 0.25
certifi                       | 2024.2.2
charset-normalizer            | 3.3.2
click                         | 8.1.7
comm                          | 0.2.2
connected-components-3d       | 3.14.1
contourpy                     | 1.2.0
cycler                        | 0.12.1
debugpy                       | 1.8.1
decorator                     | 5.1.1
dicom2nifti                   | 2.4.10
docker-pycreds                | 0.4.0
dynamic-network-architectures | 0.3.1
efficientnet-pytorch          | 0.7.1
exceptiongroup                | 1.2.0
executing                     | 2.0.1
filelock                      | 3.13.1
fonttools                     | 4.49.0
frozenlist                    | 1.4.1
fsspec                        | 2024.2.0
future                        | 1.0.0
gitdb                         | 4.0.11
GitPython                     | 3.1.42
graphviz                      | 0.20.3
huggingface-hub               | 0.21.3
idna                          | 3.6
imagecodecs                   | 2024.1.1
imageio                       | 2.34.0
importlib-metadata            | 7.1.0
importlib-resources           | 6.1.3
ipykernel                     | 6.29.3
ipython                       | 8.18.1
jedi                          | 0.19.1
Jinja2                        | 3.1.3
joblib                        | 1.3.2
jupyter-client                | 8.6.1
jupyter-core                  | 5.7.2
kiwisolver                    | 1.4.5
lazy-loader                   | 0.3
lightning-utilities           | 0.10.1
linecache2                    | 1.0.0
MarkupSafe                    | 2.1.5
matplotlib                    | 3.8.3
matplotlib-inline             | 0.1.6
MedPy                         | 0.4.0
mpmath                        | 1.3.0
multidict                     | 6.0.5
munch                         | 4.0.0
nest-asyncio                  | 1.6.0
networkx                      | 3.2.1
nibabel                       | 5.2.1
nnunetv2                      | 2.4.1
numpy                         | 1.26.4
nvidia-cublas-cu12            | 12.1.3.1
nvidia-cuda-cupti-cu12        | 12.1.105
nvidia-cuda-nvrtc-cu12        | 12.1.105
nvidia-cuda-runtime-cu12      | 12.1.105
nvidia-cudnn-cu12             | 8.9.2.26
nvidia-cufft-cu12             | 11.0.2.54
nvidia-curand-cu12            | 10.3.2.106
nvidia-cusolver-cu12          | 11.4.5.107
nvidia-cusparse-cu12          | 12.1.0.106
nvidia-nccl-cu12              | 2.19.3
nvidia-nvjitlink-cu12         | 12.3.101
nvidia-nvtx-cu12              | 12.1.105
opencv-python                 | 4.9.0.80
opencv-python-headless        | 4.9.0.80
packaging                     | 23.2
pandas                        | 2.2.1
parso                         | 0.8.3
pexpect                       | 4.9.0
pillow                        | 10.2.0
pip                           | 20.2.4
platformdirs                  | 4.2.0
pretrainedmodels              | 0.7.4
prompt-toolkit                | 3.0.43
protobuf                      | 4.25.3
psutil                        | 5.9.8
ptyprocess                    | 0.7.0
pure-eval                     | 0.2.2
pydicom                       | 2.4.4
pygments                      | 2.17.2
pyparsing                     | 3.1.2
python-dateutil               | 2.8.2
python-gdcm                   | 3.0.23.1
pytorch-lightning             | 2.2.0.post0
pytz                          | 2024.1
PyYAML                        | 6.0.1
pyzmq                         | 25.1.2
requests                      | 2.31.0
safetensors                   | 0.4.2
scikit-image                  | 0.22.0
scikit-learn                  | 1.4.1.post1
scikit-multiflow              | 0.5.3
scipy                         | 1.12.0
seaborn                       | 0.13.2
segmentation-models-pytorch   | 0.3.3
sentry-sdk                    | 1.40.6
setproctitle                  | 1.3.3
setuptools                    | 50.3.2
SimpleITK                     | 2.3.1
six                           | 1.16.0
smmap                         | 5.0.1
sortedcontainers              | 2.4.0
stack-data                    | 0.6.3
sympy                         | 1.12
threadpoolctl                 | 3.3.0
tifffile                      | 2024.2.12
timm                          | 0.9.2
torch                         | 2.2.1
torchmetrics                  | 1.3.1
torchvision                   | 0.17.1
tornado                       | 6.4
tqdm                          | 4.66.2
traceback2                    | 1.4.0
traitlets                     | 5.14.2
triton                        | 2.2.0
typing-extensions             | 4.10.0
tzdata                        | 2024.1
unet                          | 0.7.7
unittest2                     | 1.1.0
urllib3                       | 2.2.1
wandb                         | 0.16.3
wcwidth                       | 0.2.13
yacs                          | 0.1.8
yarl                          | 1.9.4
zipp                          | 3.17.0
