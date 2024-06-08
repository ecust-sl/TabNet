import SimpleITK as sitk
import os
import numpy as np
import csv


def img_resample(image, size):
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
    # aspect = sitk.ReadImage('/home/xiao/nnUNet/aspects/updateMR_Template_label.nii.gz')
    aspect = sitk.ReadImage('template/T1_aspect_label_resampled2.0.nii.gz')
    #base = '/home/xiao/nnUNet/nnUNetFrame/DATASET/nnUNet_raw/Dataset504_All/predTs'
    # path_gt = '/home/xiao/nnUNet/nnUNetFrame/DATASET/nnUNet_raw/Dataset505_All/labelsTr'
    path_gt = '/home/xiao/nnUNet/nnUNetFrame/DATASET/nnUNet_raw/Dataset505_All/labelsOutTs'
    path = '/home/xiao/nnUNet/nnUNetFrame/DATASET/nnUNet_raw/Dataset505_All/predOutTs'
    # path_gt = '/home/xiao/nnUNet/nnUNetFrame/DATASET_test/nnUNet_raw/Dataset_AllTest/labelsTr'
    not_MCA =[7, 8, 9, 10]
    # not_MCA = []
    with open('/home/xiao/nnUNet/new_score/score_02.29_pzx/score_nn_all_out_test.csv', mode='w', newline='') as file:
        # size = [193,229,96]
        #aspect = img_resample(aspect, size)
        aspect_np = sitk.GetArrayFromImage(aspect)
        aspect_np[aspect_np > 20] = 0
        # total area of 10 zone
        area = [5 * (np.sum(aspect_np == i + 1) + np.sum(aspect_np == (10 + i + 1))) / 1000 for i in range(10)]
        # area = [2*(np.sum(aspect_np == i + 1))/1000 for i in range(20)]
        writer = csv.writer(file, delimiter=',')
        writer.writerow(area)
        for j in range(1):
            for filename in os.listdir(path):
                if filename.endswith('nii.gz'):
                    id = filename.split('.')[0]
                    if id.endswith('icu'):
                        id = id.split('_')[0]
                    print(str(id))
                    sub_area = [0] * 10
                    pred = sitk.ReadImage(os.path.join(path, filename))
                    pred_np = sitk.GetArrayFromImage(pred)
                    gt = sitk.ReadImage(os.path.join(path_gt, filename))
                    gt_np = sitk.GetArrayFromImage(gt)
                    result = aspect_np * pred_np
                    # infraction area of 10 zone
                    count = [5*(np.sum(result == i + 1)+np.sum(result == (10+i + 1)))/1000 for i in range(10)]
                    #count = [5 * (np.sum(result == i + 1)) / 1000 for i in range(20)]
                    total = 5 * np.sum(pred_np == 1)
                    total_gt = 5 * np.sum(gt_np == 1)
                    count.append(id)
                    count.append(5 * total / 1000)
                    count.append(5 * total_gt / 1000)
                    score = 10
                    for k in range(10):
                        if k + 1 in not_MCA:
                            if count[k] > area[k] * 0.02:
                            #if count[k] > 0:
                                score = score - 1
                                sub_area[k] += 1
                        else:
                            # M1-M6
                            if count[k] >= area[k] * 1 / 3:
                                score = score - 1
                                sub_area[k] += 1
                    count.append(score)
                    for i in range(10):
                        count.append(sub_area[i])
                    writer.writerow(count)
        file.close()


