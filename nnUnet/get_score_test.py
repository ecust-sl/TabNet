import SimpleITK as sitk
import os
import numpy as np
import csv

import pandas as pd


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
    path_gt = '/home/xiao/nnUNet/nnUNetFrame/DATASET/nnUNet_raw/Dataset505_All/labelsOutTs'
    path = '/home/xiao/nnUNet/nnUNetFrame/DATASET/nnUNet_raw/Dataset505_All/predOutTs'
    path_manual = '/home/xiao/nnUNet/四中心数据汇总.xls'
    data_manual = pd.read_excel(path_manual)
    ids = data_manual['id'].values.tolist()
    manual = data_manual['manual'].values.tolist()
    not_MCA = [5,6,7,8,9,10,14,18,19,20]
    left = [11,12,13,14,15,16,17,18,19,20]
    right = [1, 3 ,4 ,8 ,2 ,5 ,6 ,7 ,9 , 10]
    #not_MCA = [4,5,6,7,14,15,16,17]
    roww = ['id','1区体积','2区体积','3区体积','4区体积','5区体积','6区体积','7区体积','8区体积','9区体积','10区体积','总出血','实际出血','abs-d','score','manual','abs-score','1区扣分','2区扣分'	,'3区扣分'	,'4区扣分',	'5区扣分','6区扣分','7区扣分',	'8区扣分',	'9区扣分','10区扣分'
          ]
    thr = 0
    sec = 0
    with open('new_score/score_02.29_pzx/score_test_out_tune.csv', mode='w', newline='') as file:
        #size = [193,229,96]
        #aspect = img_resample(aspect)
        aspect_np = sitk.GetArrayFromImage(aspect)
        aspect_np[aspect_np > 20] = 0
        area = [0] * 10
        # total area of 10 zone
        #area = [5*0.5*(np.sum(aspect_np == i + 1)+np.sum(aspect_np == (10+i + 1)))/1000 for i in range(10)]
        # area = [2*(np.sum(aspect_np == i + 1))/1000 for i in range(20)]
        area[0] = (np.sum(aspect_np == 1) + np.sum(aspect_np == 11)) * 1.0 / 1000
        area[1] = (np.sum(aspect_np == 2) + np.sum(aspect_np == 15)) * 1.0 / 1000
        area[2] = (np.sum(aspect_np == 3) + np.sum(aspect_np == 12)) * 1.0 / 1000
        area[3] = (np.sum(aspect_np == 4) + np.sum(aspect_np == 13)) * 1.0 / 1000
        area[4] = (np.sum(aspect_np == 5) + np.sum(aspect_np == 16)) * 1.0 / 1000
        area[5] = (np.sum(aspect_np == 6) + np.sum(aspect_np == 17)) * 1.0 / 1000
        area[6] = (np.sum(aspect_np == 7) + np.sum(aspect_np == 18)) * 1.0 / 1000
        area[7] = (np.sum(aspect_np == 8) + np.sum(aspect_np == 14)) * 1.0 / 1000
        area[8] = (np.sum(aspect_np == 9) + np.sum(aspect_np == 19)) * 1.0 / 1000
        area[9] = (np.sum(aspect_np == 10) + np.sum(aspect_np == 20)) * 1.0 / 1000
        writer = csv.writer(file, delimiter=',')
        writer.writerow(roww)

        for j in range(1):
            #thresh = 0.09*j
            #####
            #path = os.path.join(base+str(j),'validation')
            ######
            for filename in os.listdir(path_gt):
                if filename.endswith('nii.gz'):
                    id = filename.split('.')[0]
                    if id.endswith('icu'):
                        id = id.split('_')[0]
                    # print(str(id))
                    sub_area = [0]*10
                    count = [0] * 10
                    pred = sitk.ReadImage(os.path.join(path,filename))
                    #pred = img_resample(pred)
                    pred_np = sitk.GetArrayFromImage(pred)
                    gt = sitk.ReadImage(os.path.join(path_gt, filename))
                    gt_np = sitk.GetArrayFromImage(gt)
                    result = aspect_np * pred_np
                    count.append(id)
                    # infraction area of 10 zone
                    #count = [5*(np.sum(result == i + 1)+np.sum(result == (10+i + 1)))/1000 for i in range(10)]
                    count[0] = (np.sum(result == 1) + np.sum(result == 11)) * 1.0 / 1000
                    count[1] = (np.sum(result == 2) + np.sum(result== 15)) * 1.0 / 1000
                    count[2] = (np.sum(result == 3) + np.sum(result == 12))* 1.0 / 1000
                    count[3] = (np.sum(result == 4) + np.sum(result == 13))* 1.0 / 1000
                    count[4] = (np.sum(result == 5) + np.sum(result == 16))* 1.0 / 1000
                    count[5] = (np.sum(result == 6) + np.sum(result == 17))* 1.0 / 1000
                    count[6] = (np.sum(result == 7) + np.sum(result == 18))* 1.0 / 1000
                    count[7] = (np.sum(result == 8) + np.sum(result == 14))* 1.0 / 1000
                    count[8] = (np.sum(result == 9) + np.sum(result == 19))* 1.0 / 1000
                    count[9] = (np.sum(result == 10) + np.sum(result == 20))* 1.0 / 1000
                    total = 2*np.sum(pred_np==1)
                    total_gt = 2*np.sum(gt_np==1)

                    count.append(total/1000)
                    count.append(total_gt/1000)
                    vol_d = (total - total_gt) / 1000
                    if vol_d < 0:vol_d = -vol_d
                    count.append(vol_d)
                    score = 10
                    score_mask = 10
                    for k in range (10):
                        if k+1 not in not_MCA:
                            if count[k] >= 1.0 / 3*area[k]:
                            #if count[k] > 0:
                                score = score-1
                                sub_area[k] += 1
                        else:
                            # M1-M6
                            score_check = 0.00
                            if k + 1 == 8 or k + 1 == 14:
                                score_check = 0.01    #0.01#0.02#0.04#0.1  # avg = 0.176
                            elif k+1 == 9 or k+1 ==19:
                                score_check = 0.02    #0#0.02#0.04#0.11  #avg = 0.181
                            elif k + 1 == 7 or k + 1 == 18:
                                score_check = 0.08    #0.08#0.08#0.06  #avg = 0.066
                            elif k + 1 == 10 or k + 1 == 20:
                                score_check = 0.06   #0.05#0.06#0.08#0.1 # avg = 0.126
                            elif k + 1 == 5:
                                score_check = 0.10
                            elif k + 1 == 6:
                                score_check = 0.25
                            else:
                                score_check = 0.00
                            if count[k] > score_check*area[k]:
                                score -= 1
                                sub_area[k] += 1

                    count.append(score)
                    f = 0
                    score_per = -1
                    for i , val in enumerate(ids):
                        if val == id:
                            count.append(manual[i])
                            score_per = manual[i]
                            f = 1
                            break
                    if f == 0:count.append(-1)

                    for i in range(10):
                        count.append(sub_area[i])


                    def abss(x):
                        return -x if x < 0 else x
                    if str(id) == '310107011038100':
                        print('id ==', id, 'score ==', score, 'socre_pre ==', score_per)
                    if abss(score - score_per) == 3:
                        thr += 1
                        print('id ==' ,id ,'score ==', score, 'socre_pre ==', score_per)
                    if abss(score - score_per) == 2:
                        sec += 1
                        # print('id ==' , id , 'score ==', score, 'socre_pre ==', score_per)

                    # double = 0
                    # for i in range(len(left)):
                    #     if count[left[i]-1] > 0 and count[right[i]-1] > 0:
                    #         double = 1
                    # count.append(double)
                    # writer.writerow(count)
            # writer.writerow(area)
            print('abs==3:',thr , 'abs==2:', sec)


        file.close()
