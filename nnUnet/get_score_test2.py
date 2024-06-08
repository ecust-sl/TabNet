import SimpleITK as sitk
import os
import numpy as np
import csv
def img_resample(image):

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

    size = [
        int(np.round(original_size[0] * (original_spacing[0] / new_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / new_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / new_spacing[2])))
    ]
    resample.SetSize(size)
    # print resample

    newimage = resample.Execute(image)
    # print newimage.GetSize()
    newimage.SetOrigin(original_origin)
    return newimage
if __name__ == '__main__':
    aspect = sitk.ReadImage('template/T1_aspect_label_resampled2.0.nii.gz')
    #aspect = sitk.ReadImage('/home/xiao/nnUNet/aspects/updateMR_Template_label.nii.gz')
    #aspect = sitk.ReadImage('/home/xiao/nnUNet/nnUNetFrame/ASPECTS_label .nii')
    #base = '/home/xiao/nnUNet/nnUNetFrame/DATASET_train/nnUNet_trained_models/Dataset504_All/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_'
    #path_gt = '/home/xiao/nnUNet/nnUNetFrame/DATASET_train/nnUNet_preprocessed/Dataset504_All/gt_segmentations'
    # path = '/home/xiao/nnUNet/nnUNetFrame/DATASET_test_9.14/nnUNet_raw/Dataset505_All/predTr'
    # path_gt = '/home/xiao/nnUNet/nnUNetFrame/DATASET_test_9.14/nnUNet_raw/Dataset505_All/labelsTr'
    path = '/media/xiao/DATA/dataset/nnUNet_raw/Dataset507_All/predOutTs'
    path_gt = '/media/xiao/DATA/dataset/nnUNet_raw/Dataset507_All/labelsOutTs'
    not_MCA = [7,8,9,10,14,18,19,20]
    left = [11,12,13,14,15,16,17,18,19,20]
    right = [1, 3 ,4 ,8 ,2 ,5 ,6 ,7 ,9 , 10]
    #not_MCA = [4,5,6,7,14,15,16,17]
    with open('new_score/score_03.01_xh/score_test_out_xh_new.csv', mode='w', newline='') as file:
        #size = [193,229,96]
        #aspect = img_resample(aspect)
        aspect_np = sitk.GetArrayFromImage(aspect)
        aspect_np[aspect_np > 20] = 0
        # total area of 10 zone
        #area = [5*0.5*(np.sum(aspect_np == i + 1)+np.sum(aspect_np == (10+i + 1)))/1000 for i in range(10)]
        area = [2*(np.sum(aspect_np == i + 1))/1000 for i in range(20)]
        writer = csv.writer(file, delimiter=',')
        writer.writerow(area)
        for j in range(1):
            #thresh = 0.09*j
            #####
            #path = os.path.join(base+str(j),'validation')
            ######
            for filename in os.listdir(path):
                if filename.endswith('nii.gz'):
                    id = filename.split('.')[0]
                    if id.endswith('icu'):
                        id = id.split('_')[0]
                    print(str(id))
                    sub_area = [0]*20
                    pred = sitk.ReadImage(os.path.join(path,filename))
                    #pred = img_resample(pred)
                    pred_np = sitk.GetArrayFromImage(pred)
                    gt = sitk.ReadImage(os.path.join(path_gt, filename))
                    gt_np = sitk.GetArrayFromImage(gt)
                    result = aspect_np * pred_np
                    result_gt = gt_np
                    # infraction area of 10 zone
                    #count = [5*(np.sum(result == i + 1)+np.sum(result == (10+i + 1)))/1000 for i in range(10)]
                    count = [2*(np.sum(result == i + 1))/1000 for i in range(20)]
                    total = 2*np.sum(pred_np==1)
                    total_gt = 2*np.sum(gt_np==1)
                    count.append(id)
                    count.append(total/1000)
                    count.append(total_gt/1000)
                    score = 10
                    for k in range (20):
                        if k+1 not in not_MCA:
                            if count[k] >= 1.0 / 3*area[k]:
                            #if count[k] > 0:
                                score = score-1
                                sub_area[k] += 1
                        else:
                            # M1-M6
                            score_check = 0.00
                            if k + 1 == 8 or k + 1 == 14:
                                score_check = 0.02    #0.01#0.02#0.04#0.1  # avg = 0.176
                            elif k+1 == 9 or k+1 ==19:
                                score_check = 0.02    #0#0.02#0.04#0.11  #avg = 0.181
                            elif k + 1 == 7 or k + 1 == 18:
                                score_check = 0.08    #0.08#0.08#0.06  #avg = 0.066
                            elif k + 1 == 10 or k + 1 == 20:
                                score_check = 0.06   #0.05#0.06#0.08#0.1 # avg = 0.126
                            else:
                                score_check = 0.00
                            if count[k] > score_check*area[k]:
                                score -= 1
                                sub_area[k] += 1

                    count.append(score)
                    for i in range(20):
                        count.append(sub_area[i])
                    double = 0
                    for i in range(len(left)):
                        if count[left[i]-1] > 0 and count[right[i]-1] > 0:
                            double = 1
                    count.append(double)
                    writer.writerow(count)

        file.close()
