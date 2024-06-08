import os

import numpy as np
import pandas as pd
import shutil
from batchgenerators.utilities.file_and_folder_operations import *
#src = '/home/xiao/spacing1/preprocessed'
# src = '/media/xiao/DATA/Hema/Hematoma-Expansion/preprocessed_puzhongxin'
src = '/media/xiao/DATA/registration_Orin/registration_old/preprocessed3/puzhongxin'
# src = '/media/xiao/DATA/registration_Orin/registration_old/preprocessed3/val'
#src = '/home/xiao/project/3DUnetCNN/Task500_Tongji'
# src_icu = '/home/xiao/spacing1/preprocessed_icu
# to_img = '/home/xiao/nnUNet/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task502_Puzhongxin/imagesTr'
# to_seg = '/home/xiao/nnUNet/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task502_Puzhongxin/labelsTr'
# to_img = '/home/xiao/nnUNet/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task503_Xinhua/imagesTr'
# to_seg = '/home/xiao/nnUNet/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task503_Xinhua/labelsTr'
to_img = '/home/xiao/nnUNet/nnUNetFrame/DATASET/nnUNet_raw/Task505_All/imagesOutTs'
to_seg = '/home/xiao/nnUNet/nnUNetFrame/DATASET/nnUNet_raw/Task505_All/labelsOutTs'
to_pre = '/home/xiao/nnUNet/nnUNetFrame/DATASET/nnUNet_raw/Task506_All/predTs'
# src = '/home/xiao/nnUNet/nnUNetFrame/DATASET/nnUNet_raw/Dataset505_All'
# to_img1 = '/home/xiao/nnUNet/nnUNetFrame/DATASET/nnUNet_raw/Dataset505_All/case study/imagesTs'
# to_seg1 = '/home/xiao/nnUNet/nnUNetFrame/DATASET/nnUNet_raw/Dataset505_All/case study/labelsTs'
# to_pred1= '/home/xiao/nnUNet/nnUNetFrame/DATASET/nnUNet_raw/Dataset505_All/case study/predTs'
# xls_path = '/home/xiao/Desktop/results/model_score.xls'
# df = pd.read_excel(xls_path)
# id_list = [str(i) for i in df['id'].values.tolist()]
if __name__ == '__main__':
    # if not os.path.exists(to_img):
    #     os.makedirs(to_img)
    # if not os.path.exists(to_seg):
    #     os.makedirs(to_seg)
    # patient_list = np.loadtxt('/home/xiao/nnUNet/file.txt')
    for patient in os.listdir(src):
    #for patient in id_list:
        if "ASPECTS" in patient or "MRI" in patient or "record" in patient:
            continue
        sub_path = os.path.join(src, patient)
        img_path = os.path.join(sub_path, 'img')
        seg_path = os.path.join(sub_path, 'seg')
        # seg_path = os.path.join(sub_path, 'seg')

        # img_path = os.path.join(src, 'imagesTr')
        # seg_path = os.path.join(src, 'labelsTr')

        # img_src = os.path.join(img_path, patient + '_extract_regist.nii.gz')
        # seg_src = os.path.join(seg_path, patient + '_regist.nii.gz')
        # img_src = os.path.join(img_path, patient + '_0000.nii.gz')
        # seg_src = os.path.join(seg_path, patient + '.nii.gz')


        seg_src = os.path.join(seg_path, patient) + '_regist.nii.gz'
        img_src = os.path.join(img_path, patient) + '_extract_regist.nii.gz'
        # if not os.path.isdir(img_src):
        #     img_src = os.path.join(img_path, patient) + '.nii.gz'
        # if not os.path.exists(img_src):
        #     img_src = os.path.join(img_path,patient) + '_icu.nii.gz'
        #     seg_src = os.path.join(seg_path,patient) + '_icu.nii.gz'

        img_to = os.path.join(to_img, patient+'.nii.gz')
        seg_to = os.path.join(to_seg, patient+'.nii.gz')
        shutil.copy(img_src, img_to)
        shutil.copy(seg_src, seg_to)

    # for patient_icu in os.listdir(src_icu):
    #     sub_path = os.path.join(src_icu, patient_icu)
    #     img_path = os.path.join(sub_path, 'img')
    #     seg_path = os.path.join(sub_path, 'seg')
    #     img_src = os.path.join(img_path, patient_icu + '_extract_regist.nii.gz')
    #     seg_src = os.path.join(seg_path, patient_icu + '.nii.gz')
    #     img_to = os.path.join(to_img, patient_icu + '_icu.nii.gz')
    #     seg_to = os.path.join(to_seg, patient_icu + '_icu.nii.gz')
    #     shutil.copy(img_src, img_to)
    #     shutil.copy(seg_src, seg_to)
