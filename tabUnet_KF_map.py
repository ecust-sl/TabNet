import time

from tabnet_lib.tab_model import TabNetClassifier
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
np.random.seed(0)

import scipy
import json
import os
from pathlib import Path

from matplotlib import pyplot as plt
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import torch.nn as nn
from unet3d.scripts.train import main as get_data
from unet3d.utils.pytorch import functions
from sklearn.model_selection import KFold
def load_criterion(criterion_name, n_gpus=0):
    try:
        criterion = getattr(functions, criterion_name)
    except AttributeError:
        criterion = getattr(torch.nn, criterion_name)()
        if n_gpus > 0:
            criterion.cuda()
    return criterion
project_dir = '/home/shilei/project/code'
#project_dir = '/media/xiao/新加卷/3DUnetCNN_local'
tab_dir = os.path.join(project_dir,'table_data/tabunet_uni_map-2.xlsx')
train = pd.read_excel(tab_dir)
train = train.astype('str')
# tab_dir = os.path.join(project_dir,'table_data/clinic_data_cls_2.xlsx')
# train = pd.read_excel(tab_dir)

target = 'label'
unused_feat = ['id' , 'NIHSS' , 'manual' , 'double']

    # unused_feat = ['id_new','TSS', 'DIFF_ABS', 'NIHSS','SCORE','dice','total',
    #                'location','shifts','smokes','drinks','diabetes','myocardials',
    #                'coronarys','atrias','hypertensions','strokes','Set','uni_vol','sum_map']
    # unused_feat = ['id_new','dice','SCORE','M1','M2','M3','M4','M5','M6',
    #                         'Caudate','Inular_ribbon','Lentiform_nucleus','Internal_capsule',
    #                         'total', 'TSS', 'DIFF_ABS', 'NIHSS',
    #                         'location', 'shifts', 'smokes', 'drinks', 'diabetes',
    #                        'myocardials', 'coronarys', 'atrias', 'hypertensions', 'strokes','Set'
    #     ]

features = [col for col in train.columns if col not in unused_feat + [target]]
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
X = train[features].values
y = train['label'].values
# label0 = train[train["label"]==0]
# label1 = train[train["label"]==1]
# X = label1 + label0
ans = 0
for train_indices, valid_indices in skf.split(X , y):
    print("这是第", ans ,"轮次开始交叉验证--------")
    ans += 1
    train = pd.read_excel(tab_dir)
    train = train.astype('str')
    # print(len(train_indices) , ' ' , len(valid_indices))
    train_list = train['id'][train_indices].tolist()
    val_list = train['id'][valid_indices].tolist()
    # print(train_list)
    # print(train)
    # test_list = train['id_new'][test_indices].tolist()
    data_list = {"training": train_list, "validation": val_list}
    path = '/home/shilei/project/code/result_model12_pre/0327/model1/data_split/fold_' + str(ans) + '.json'
    with open(path, "w") as f:
        json.dump(data_list, f)
    with open("examples/all504/data_list.json", "w") as f:
        json.dump(data_list, f)
    nunique = train.nunique()
    types = train.dtypes
    categorical_columns = ['']
    categorical_dims = {}
    for col in train.columns:
        if types[col] == 'object':
            # print(col, train[col].nuniqu e( ))
            l_enc = LabelEncoder()
            train[col] = train[col].fillna("VV_likely")
            train[col] = l_enc.fit_transform(train[col].values)
            categorical_columns.append(col)
            categorical_dims[col] = len(l_enc.classes_)
        else:
            train.fillna(train.loc[train_indices, col].mean(), inplace=True)

    t1 = time.time()
    train_loader, val_loader, seg_model = get_data(1)
    t2 = time.time()
    print('data loaded: (s)', (t2 - t1))
    # if "Set" not in train.columns:
    #     train["Set"] = np.random.choice(["train", "valid", "test"], p =[.8, .1, .1], size=(train.shape[0],))
    #
    # train_indices = train[train.Set=="train"].index
    # valid_indices = train[train.Set=="valid"].index
    # test_indices = train[train.Set=="test"].index
    # unused_feat = ['TSS','DIFF_ABS','NIHSS','location','shifts','smokes','drinks','diabetes','myocardials','coronarys','atrias','hypertensions','strokes']
    unused_feat = ['id','NIHSS' , 'manual', 'double','score']

    # unused_feat = ['id_new','TSS', 'DIFF_ABS', 'NIHSS','SCORE','dice','total',
    #                'location','shifts','smokes','drinks','diabetes','myocardials',
    #                'coronarys','atrias','hypertensions','strokes','Set','uni_vol','sum_map']
    # unused_feat = ['id_new','dice','SCORE','M1','M2','M3','M4','M5','M6',
    #                         'Caudate','Inular_ribbon','Lentiform_nucleus','Internal_capsule',
    #                         'total', 'TSS', 'DIFF_ABS', 'NIHSS',
    #                         'location', 'shifts', 'smokes', 'drinks', 'diabetes',
    #                        'myocardials', 'coronarys', 'atrias', 'hypertensions', 'strokes','Set'
    #     ]

    features = [col for col in train.columns if col not in unused_feat + [target]]
    print((features))
    cat_idxs = [i for i, f in enumerate(features) if f in categorical_columns]

    cat_dims = [categorical_dims[f] for i, f in enumerate(features) if f in categorical_columns]

    # grouped_features = [[0, 1, 2], [8, 9, 10]]

    tabnet_params = {
        "n_steps": 3,
        "cat_idxs": cat_idxs,
        "cat_dims": cat_dims,
        "cat_emb_dim": 2,
        "optimizer_fn": torch.optim.Adam,
        "optimizer_params": dict(lr=1e-2),
        "scheduler_params": {"step_size": 20,  # how to use learning rate scheduler
                             "gamma": 0.5},
        "scheduler_fn": torch.optim.lr_scheduler.StepLR,
        "mask_type": 'entmax'  # "sparsemax"
        # "grouped_features" : grouped_features

    }
    print(tabnet_params)
    # seg_model = None
    # train_loader = None
    # val_loader = None
    clf = TabNetClassifier(**tabnet_params
                          )

    X_train = train[features].values[train_indices]
    y_train = train[target].values[train_indices]

    X_valid = train[features].values[valid_indices]
    y_valid = train[target].values[valid_indices]
    # clf = nn.DataParallel(clf)
    # X_test = train[features].values[test_indices]
    # y_test = train[target].values[test_indices]
    max_epochs = 100
    from pytorch_tabnet.augmentations import ClassificationSMOTE
    aug = ClassificationSMOTE()
    #aug = None
    # This illustrates the behaviour of the model's fit method using Compressed Sparse Row matrices
    sparse_X_train = scipy.sparse.csr_matrix(X_train)  # Create a CSR matrix from X_train
    sparse_X_valid = scipy.sparse.csr_matrix(X_valid)  # Create a CSR matrix from X_valid
    # Fitting the model
    #clf = torch.nn.DataParallel(clf)
    clf.fit(
        model_name='model1',
        X_train=X_train, y_train=y_train,
        eval_set=[(X_train, y_train),(X_valid, y_valid)],
        tr_loader=train_loader,
        val_loader=val_loader,
        eval_name=['train','val'],
        eval_metric=['auc','accuracy','precision','recall','f1_score','specificity','model1_pre', 'model2_pre'],
        # eval_metric=['custom'],
        # eval_metric=['balanced_accuracy'],
        model_seg = seg_model,
        fold = ans,
        #model_seg =None,
        loss_seg = load_criterion("per_channel_dice_loss",n_gpus=1),
        #loss_fn = load_criterion("CrossEntropyLoss",n_gpus=1),
        max_epochs=max_epochs , patience=200,
        batch_size=32, virtual_batch_size=128,
        num_workers=4,
        weights=1,
        drop_last=False,
        augmentations=aug, #aug, None
    )

    # This illustrates the warm_start=False behaviour
    save_history = []

    #plot losses
    plt.plot(clf.history['loss_cls'])
    plt.savefig('/home/shilei/project/code/result_model12_pre/0327/model1/loss_png/loss_cls_plot.png')
    plt.close()

    # 绘制并保存训练准确率曲线图
    plt.plot(clf.history['train_accuracy'])
    plt.savefig('/home/shilei/project/code/result_model12_pre/0327/model1/loss_png/losstrain_accuracy_plot.png')
    plt.close()

    # 绘制并保存验证准确率曲线图
    plt.plot(clf.history['val_accuracy'])
    plt.savefig('/home/shilei/project/code/result_model12_pre/0327/model1/loss_png/val_accuracy_plot.png')
    plt.close()

    # 绘制并保存学习率曲线图
    plt.plot(clf.history['lr'])
    plt.savefig('/home/shilei/project/code/result_model12_pre/0327/model1/loss_png/learning_rate_plot.png')
    plt.close()


