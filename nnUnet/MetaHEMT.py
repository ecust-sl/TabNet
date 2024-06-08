# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 20:01:53 2021

@author: edwadzheng
"""

import sys, os
sys.path.append("./")
sys.path.append("../util")
import numpy as np
import torch
import monai
import argparse
from util.dataloader import Meta3Dloader
from util import model_zoo
import yaml
import cv2
import SimpleITK as sitk

torch.multiprocessing.set_sharing_strategy('file_system')
monai.config.print_config()

def process_data(data, device='cuda'):
    now = data['now'].to(device)
    fu = data['fu'].to(device)
    now_mask = data['now_seg'].to(device)
    now_mask = 1.0 * (now_mask > 0.3)
    fu_mask = data['fu_seg'].to(device)
    fu_mask = 1.0 * (fu_mask > 0.3)
    return now, fu, now_mask, fu_mask

def to_DHW(tensor):
    """
    convert 5D tensor from BCHWD to DHW as BHW with numpy
    """
    tensor = tensor.squeeze().permute(2,0,1)
    return tensor

def rescale_meta(data, config):
    
    data[data>config['max']] = config['max']
    data[torch.logical_and(data<config['min'], data>0.)] = config['max']
    results = config['scale'] * (data - config['min'] + config['eps']) / (config['max'] - config['min'] + config['eps'])
    return results

def processed_meta(data, config, device='cuda'):
    
    meta_data = data['meta'].copy()
    meta_batch = []
    rescale_keys = config['rescale_meta']
    for k in rescale_keys:
        meta_batch.append(rescale_meta(meta_data[k], config[k]).unsqueeze(1).float())
        del meta_data[k]
    for k in meta_data:
        if k not in config['selected_meta']:
            continue
        else:
            meta_batch.append(meta_data[k].unsqueeze(1).float())
    meta_batch = torch.cat(meta_batch, 1)
    return meta_batch.to(device)

def saveNii(fu_seg, pred, now_seg, save_path, ta):
    fu_seg = 255*(fu_seg>0)
    pred = 255*(pred>0)
    TP = 2. * ((fu_seg==pred) * (fu_seg==255) * (pred==255))
    FP = 1. * (pred==255) * (fu_seg!=255)
    FN = 3. *(fu_seg==255) * (pred!=255)
    edge_nii = []
    contours_list = []
    for i in range(fu_seg.shape[0]):
        now_ = cv2.cvtColor(np.uint8(now_seg[i]), cv2.COLOR_GRAY2RGB)
        edges = np.zeros_like(now_)
        contours, _ = cv2.findContours(np.uint8(now_seg[i]), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:] # cv2.RETR_TREE
        edges = cv2.drawContours(edges, contours, -1, (7,7,7), 1)[:, :, 1] #3 will get the edges
        edge_nii.append(edges)
        contours_list.append(contours)

    edge_nii=np.asarray(edge_nii)
    mask = TP + FP + FN
    mask_all = mask + edge_nii
    mask_edge = np.logical_and(mask,edge_nii)    
    mask = mask_edge * 7. + np.logical_not(mask_edge) * mask_all    
    monai.data.write_nifti(mask.transpose([1,2,0]), save_path, affine=ta)


def inference(test_loader, device, model, config, save_path):
    """
    Args: 
        loader: data loader
        device: designated running device
        model: testing pytorch model
        val_config: other hyper-parameters

    Returns:
        dice score
    """
    meta_config = config['MetaData']
    
    # define evaluation metrics 
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            now, fu, now_seg, fu_seg = process_data(data, device)
            meta_data = processed_meta(data, meta_config, device)
            obj_name = data['now_meta_dict']['filename_or_obj'][0]
            obj_name = obj_name.split(os.sep)[-1].split('-')[0]         
            pred_seg, flow = model(now, now_seg, meta_data)  
            pred_seg = (pred_seg > 0.) * 1.0
            pred_seg = ((pred_seg + now_seg) > 0. ) * 1.0
            pred_image = model.stn(now, flow)

            # save the prediction into disk
            ta = data['now_meta_dict']['affine'][0].numpy()
            sp = os.path.join(save_path, obj_name)
            if not os.path.exists(sp): os.makedirs(sp)
            monai.data.write_nifti(now.squeeze().cpu().numpy(), os.path.join(sp, 'now.nii.gz'), affine=ta)
            monai.data.write_nifti(now_seg.squeeze().cpu().numpy(), os.path.join(sp, 'now_seg.nii.gz'), affine=ta)
            monai.data.write_nifti(fu.squeeze().cpu().numpy(), os.path.join(sp, 'fu.nii.gz'), affine=ta)
            monai.data.write_nifti(fu_seg.squeeze().cpu().numpy(), os.path.join(sp, 'fu_seg.nii.gz'), affine=ta)
            monai.data.write_nifti(pred_seg.squeeze().cpu().numpy(), os.path.join(sp, 'pred_seg.nii.gz'), affine=ta)
            monai.data.write_nifti(pred_image.squeeze().cpu().numpy(), os.path.join(sp, 'pred_img.nii.gz'), affine=ta)
            saveNii(fu_seg.squeeze().cpu().permute(2,0,1).numpy(),
                    pred_seg.squeeze().cpu().permute(2,0,1).numpy(),
                    now_seg.squeeze().cpu().permute(2,0,1).numpy(),
                    os.path.join(sp, 'visual_mask.nii.gz'), ta)
            
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./config/config.yaml', type=str, help='path to the data folder.')
    parser.add_argument('--split', type=str, required=True, help='split for pairs')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint path for eval.')
    parser.add_argument('--save_path', type=str, default=None, help='save_results')

    FLAGS = parser.parse_args()
    print(FLAGS)
    
    # load valid config
    f = open(FLAGS.config)
    config = yaml.load(f, Loader=yaml.FullLoader)
    val_config = config['Val']
    model_config = config['Model']
    common_config = config['Common']
    preprocess_config = config['Preprocess']
    # create dataloader
    test_loader = Meta3Dloader(val_config['path'], split=FLAGS.split, 
                                      batch_size=val_config['batch_size'],
                                      scope_type = val_config['scope_type'],
                                      preprecess_hype = preprocess_config)
    
    if torch.cuda.is_available()==True:
        device = torch.device('cuda:'+str(val_config['gpu_id']))
    else:
        raise ValueError("Error, requires GPU.")

    # define model
    model = model_zoo.MetaTFPNet(model_config).to(device)
    save_path = FLAGS.save_path
    model.load_state_dict(torch.load(FLAGS.ckpt, map_location='cuda:'+str(val_config['gpu_id'])))
    inference(test_loader, device, model, config, save_path)