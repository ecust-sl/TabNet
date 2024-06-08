import os
import SimpleITK as sitk
import numpy as np

directory = '/home/xiao/nnUNet/nnUNetFrame/DATASET/nnUNet_trained_models/Dataset500_Tongji/nnUNetTrainer__nnUNetPlans__3d_fullres'
seg_np = np.zeros((96,229,193))

for i in range(5):
    seg_path = 'fold_'+str(i)+'/validation'
    final_path = os.path.join(directory, seg_path)
    for filename in os.listdir(final_path):
        if filename.endswith('nii.gz'):
            nps = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(final_path,filename)))
            seg_np = seg_np + nps
            # if seg_np.shape != (96,229,193):
            #     print(filename)
seg_np[seg_np<=10] = 0
map = sitk.GetImageFromArray(seg_np)
writer = sitk.ImageFileWriter()
writer.SetFileName('heat_map.nii.gz')
writer.Execute(map)

