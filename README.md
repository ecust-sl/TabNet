# TabNet

## Requirements
Our code works with the following environment.
- `torch==2.1.2`
- `torchvision==0.6.2`
- `opencv-python==4.4.0.42`

## Datasets
We use datasets in our paper.
First you need to follow the steps in step.txt in the nnUnet file directory to preprocess the image.
You need to put the dataset in the Task504_All folder and meet the following directory structure
- Task504_All
  - imagesTr
  - imagesTs
  - labelsTr
  - labelsTs

## Running

Traditional model training：python traditional_classify_binary.py
Out model training：python tabUnet_KF_map.py
