import monai
import cv2
import numpy as np
import os
import SimpleITK as sitk
import nibabel as nib
def compute_tp_fp_fn_tn(mask_ref: np.ndarray, mask_pred: np.ndarray, ignore_mask: np.ndarray = None):
    if ignore_mask is None:
        use_mask = np.ones_like(mask_ref, dtype=bool)
    else:
        use_mask = ~ignore_mask
    tp = np.sum((mask_ref & mask_pred) & use_mask)
    fp = np.sum(((~mask_ref) & mask_pred) & use_mask)
    fn = np.sum((mask_ref & (~mask_pred)) & use_mask)
    tn = np.sum(((~mask_ref) & (~mask_pred)) & use_mask)
    return tp, fp, fn, tn
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
def saveNii(aspects,fu_seg, pred, now_seg, save_path, ta):

    #aspects[aspects>20]=0
    # fu_seg = (fu_seg > 0)
    # pred = (pred > 0)
    fu_seg = 255*(fu_seg > 0)
    pred = 255* (pred > 0)
    aspects_i = aspects
    for x in range(20):

        area_mask = np.zeros_like(pred,dtype=np.uint8)
        area_mask[aspect_np == (x + 1)] = 1
        mask_ref_i = aspects_i * area_mask
        mask_pred_i = pred * area_mask
        tp_i, fp_i, fn_i, tn_i = compute_tp_fp_fn_tn(mask_ref_i, mask_pred_i, ignore_mask=None)
        dice_i = 2 * tp_i / (2 * tp_i + fp_i + fn_i + 1e-6)

        if dice_i == 0:
           aspects_i[aspects_i==(x+1)]=0

    TP = 2. * ((fu_seg==pred) * (fu_seg==255) * (pred==255))
    FP = 1. * (pred==255) * (fu_seg!=255)
    FN = 3. *(fu_seg==255) * (pred!=255)

    edge_nii = []
    contours_list = []
    edge_nii_pred = []
    contours_list_pred = []
    for i in range(fu_seg.shape[0]):
        now_ = cv2.cvtColor(np.uint8(now_seg[i]), cv2.COLOR_GRAY2RGB)
        edges = np.zeros_like(now_)
        edges_pred = np.zeros_like(now_)
        contours, _ = cv2.findContours(np.uint8(now_seg[i]), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:] # cv2.RETR_TREE
        contours_pred, _ = cv2.findContours(np.uint8(pred[i]), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:] # cv2.RETR_TREE
        edges = cv2.drawContours(edges, contours, -1, (7,7,7), 1)[:, :, 1] #3 will get the edges
        edges_pred = cv2.drawContours(edges_pred, contours_pred, -1, (7,7,7), 1)[:, :, 1] #3 will get the edges
        edge_nii.append(edges)
        edge_nii_pred.append(edges_pred)
        contours_list.append(contours)
        contours_list_pred.append(contours_pred)

    edge_nii=np.asarray(edge_nii)
    edge_nii_pred=np.asarray(edge_nii_pred)
    mask = TP + FP + FN
    mask_all = mask + edge_nii+edge_nii_pred
    mask_edge = np.logical_and(mask,edge_nii)
    mask_edge_pred = np.logical_and(mask,edge_nii_pred)

    #mask = mask_edge * 7. + np.logical_not(mask_edge) * mask_all #+ mask_edge_pred * 6.

    #mask = pred/255*21+np.logical_not(pred)*aspects
    mask = aspects_i
    monai.data.write_nifti(mask.transpose([2,1,0]), save_path,affine=ta)

folder_ref = '/home/xiao/nnUNet/nnUNetFrame/DATASET/nnUNet_raw/Dataset505_All/labelsTr'
folder_pred = '/home/xiao/nnUNet/nnUNetFrame/DATASET/nnUNet_raw/Dataset505_All/predTr'

# folder_ref = '/home/xiao/nnUNet/nnUNetFrame/DATASET_train/nnUNet_raw/Dataset504_All/labelsTr'
#
# folder_pred = '/home/xiao/nnUNet/nnUNetFrame/DATASET_train/nnUNet_raw/Dataset504_All/predTr'
# folder_ref = '/media/xiao/DATA/registration_Orin/registration_old/preprocessed_train/train_label/labelsTr'
#
# folder_pred = '/media/xiao/DATA/registration_Orin/registration_old/preprocessed_train/train/predTr'
save = '/home/xiao/nnUNet/visual/visual_test_1012/'
save_aspects = '/home/xiao/nnUNet/visual/aspect_train_1012/'
aspect = sitk.ReadImage('template/T1_aspect_label_resampled2.0.nii.gz')
not_MCA = [7,8,9,10,14,18,19,20]
#size = [193,229,96]
#aspect = img_resample(aspect, size)
aspect_np = sitk.GetArrayFromImage(aspect)
for filename in os.listdir(folder_ref):
    if filename.endswith('nii.gz'):
        label_path = os.path.join(folder_ref,filename)
        pred_path = os.path.join(folder_pred,filename)
        ta = nib.load(pred_path).affine.copy()
        label_np = sitk.GetArrayFromImage(sitk.ReadImage(label_path))
        pred_np = sitk.GetArrayFromImage(sitk.ReadImage(pred_path))
        saveNii(aspects=aspect_np,fu_seg=label_np,pred=pred_np,now_seg=label_np,save_path=os.path.join(save_aspects,filename),ta = ta)

