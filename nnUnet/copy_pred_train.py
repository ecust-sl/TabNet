import os
import shutil
base = '/home/xiao/nnUNet/nnUNetFrame/DATASET_train/nnUNet_trained_models/Dataset504_All/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_'
to_path = '/home/xiao/nnUNet/nnUNetFrame/DATASET_train/nnUNet_raw/nnUNet_raw_data/Task504_All/predTr'
for i in range(5):
    path = os.path.join(base + str(i), 'validation')
    for filename in os.listdir(path):
        if filename.endswith('nii.gz'):
            src = os.path.join(path,filename)
            to = os.path.join(to_path,filename)
            shutil.copy(src,to)