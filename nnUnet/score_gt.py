import SimpleITK as sitk
import os
import ants
import scipy.ndimage as ndimage
import nibabel as nib
import numpy as np
import csv
from scipy.ndimage import morphology

from sklearn.cluster import KMeans



def img_resample(image,size):

    # print image
    # print sitk.GetArrayFromImage(image)
    # print image.GetSize()

    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    original_origin = image.GetOrigin()

    resample = sitk.ResampleImageFilter()
    resample.SetInterpolator(sitk.sitkLinear)
    resample.SetInterpolator(sitk.sitkNearestNeighbor)
    resample.SetDefaultPixelValue(0)

    new_spacing = [1, 1, 2]
    resample.SetOutputSpacing(new_spacing)
    resample.SetOutputOrigin(original_origin)
    resample.SetOutputDirection(image.GetDirection())

    # size = [
    #     int(np.round(original_size[0] * (original_spacing[0] / new_spacing[0]))),
    #     int(np.round(original_size[1] * (original_spacing[1] / new_spacing[1]))),
    #     int(np.round(original_size[2] * (original_spacing[2] / new_spacing[2])))
    # ]
    resample.SetSize(size)
    # print resample

    newimage = resample.Execute(image)
    # print newimage.GetSize()
    newimage.SetOrigin(original_origin)
    return newimage


if __name__ == '__main__':

    aspect = sitk.ReadImage('/home/xiao/nnUNet/aspects/updateMR_Template_label.nii.gz')
    base = '/home/xiao/nnUNet/nnUNetFrame/DATASET_test_9.14/nnUNet_trained_models/Dataset505_All/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_'
    #path_gt = '/home/xiao/nnUNet/nnUNetFrame/DATASET/nnUNet_preprocessed/Dataset500_Tongji/gt_segmentations'
    path_gt = '/home/xiao/nnUNet/nnUNetFrame/DATASET_test_9.14/nnUNet_preprocessed/Dataset505_All/gt_segmentations'
    #path_gt = '/home/xiao/project/3DUnetCNN/Task504_All/labelsTr'
    not_MCA = [4,5,6,7]
    with open('new_score/score_9.17/score_gt_all.csv', mode='w', newline='') as file:
        size = [193,229,96]
        #aspect = img_resample(aspect, size)
        aspect_np = sitk.GetArrayFromImage(aspect)
        area = [2*(np.sum(aspect_np == i + 1)+np.sum(aspect_np == 10+(i + 1)))/1000 for i in range(10)]
        writer = csv.writer(file, delimiter=',')
        writer.writerow(area)
        for j in range(5):
            path = os.path.join(base+str(j),'validation')
            for filename in os.listdir(path):
                if filename.endswith('nii.gz'):
                    id = filename.split('.')[0]
                    if id.endswith('icu'):
                        id = id.split('_')[0]
                    #print(str(id))
                    sub_area = [0]*10
                    pred = sitk.ReadImage(os.path.join(path,filename))
                    pred_np = sitk.GetArrayFromImage(pred)
                    gt = sitk.ReadImage(os.path.join(path_gt, filename))
                    gt_np = sitk.GetArrayFromImage(gt)
                    ###################
                    result = aspect_np * gt_np
                    ###################
                    count = [2*(np.sum(result == x + 1)+np.sum(result == (x + 1+10)))/1000 for x in range(10)]
                    total = np.sum(pred_np==1)
                    total_gt = np.sum(gt_np==1)
                    count.append(id)
                    count.append(2*total/1000)

                    count.append(2*total_gt/1000)
                    score = 10
                    for y in range (10):
                        if y+1 in not_MCA:
                            if count[y]>0:
                                score = score-1
                                sub_area[y] += 1
                        else:
                            if count[y] >= area[y]*1/3:
                                score = score-1
                                sub_area[y] += 1
                    for a in range(10):
                        count.append(sub_area[a])
                    count.append(score)
                    writer.writerow(count)

        file.close()



    # data_dir = r'fixed/5379874/img/5379874.nii'
    # extract_dir = r'fixed/5379874/img/extract/5379874_extract_2.nii'
    # image = sitk.ReadImage(data_dir)
    # image = np.array(image)
    # mask = get_brain(image)
    # save2nii(mask, extract_dir)
    # print("Finish Brain Extraction")

    # data_dir = '136401001457843.nii'
    # data_to = '136401001457843_resize.nii'
    # #fixed_dir = 'fixed/DWI-ASPECTS Template/MR_Template.nii'
    # image = sitk.ReadImage(data_dir)
    #
    # #target = sitk.ReadImage(fixed_dir)
    #
    # image = img_resample(image)
    # #target = img_resample(target)
    # #img = resize_image_itk(image,target)
    # save2nii(image,data_to)