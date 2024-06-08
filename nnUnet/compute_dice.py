import nibabel as nib
import numpy as np

pred = nib.load('/home/xiao/nnUNet/nnUNetFrame/DATASET/nnUNet_trained_models/Dataset500_Tongji/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/validation/5379874.nii.gz')
pred_np = np.array(pred)
aspect = nib.load(r'E:\WH\3DUnetCNN\output_results\MR_Template_label.nii')
aspect_np = np.array(aspect)
result = aspect_np*pred_np
count = []
for i in range(42):
    count[i] = np.sum(result==i+1)
print(count)
