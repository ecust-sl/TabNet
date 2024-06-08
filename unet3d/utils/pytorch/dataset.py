import pandas as pd
from tqdm import tqdm
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
import json
from ..sequences import (WholeVolumeToSurfaceSequence, HCPRegressionSequence, get_metric_data,
                         WholeVolumeAutoEncoderSequence, WholeVolumeSegmentationSequence, WindowedAutoEncoderSequence,
                         SubjectPredictionSequence, fetch_data_for_point, WholeVolumeCiftiSupervisedRegressionSequence,
                         WholeVolumeSupervisedRegressionSequence)
from ..utils import nib_load_files
from torch.utils.data import Dataset
import torch

project_dir = '/home/shilei/project/code'
#project_dir = '/media/xiao/新加卷/3DUnetCNN_local'


class WholeBrainCIFTI2DenseScalarDataset(WholeVolumeToSurfaceSequence, Dataset):
    def __init__(self, *args, batch_size=1, shuffle=False, **kwargs):
        super().__init__(*args, batch_size=batch_size, shuffle=shuffle, **kwargs)

    def __len__(self):
        return len(self.epoch_filenames)

    def __getitem__(self, idx):
        feature_filename, surface_filenames, metric_filenames, subject_id = self.epoch_filenames[idx]
        metrics = nib_load_files(metric_filenames)
        x = self.resample_input(feature_filename)
        y = self.get_metric_data(metrics, subject_id)
        return torch.from_numpy(x).float().permute(3, 0, 1, 2), torch.from_numpy(y).float()

    def get_metric_data(self, metrics, subject_id):
        return get_metric_data(metrics, self.metric_names, self.surface_names, subject_id).T.ravel()

    def get_data_info(self):
        pass


class HCPRegressionDataset(HCPRegressionSequence, Dataset):
    def __init__(self, *args, points_per_subject=1, **kwargs):
        super().__init__(*args, batch_size=points_per_subject, **kwargs)

    def __len__(self):
        return len(self.epoch_filenames)

    def __getitem__(self, idx):
        x, y = self.fetch_hcp_subject_batch(*self.epoch_filenames[idx])
        return torch.from_numpy(np.moveaxis(np.asarray(x), -1, 1)).float(), torch.from_numpy(np.asarray(y)).float()


class HCPSubjectDataset(SubjectPredictionSequence):
    def __init__(self, *args, batch_size=None, **kwargs):
        if batch_size is not None:
            print("Ignoring the set batch_size")
        super().__init__(*args, batch_size=1, **kwargs)

    def __getitem__(self, idx):
        x = self.fetch_data_for_index(idx)
        return torch.from_numpy(np.moveaxis(np.asarray(x), -1, 0)).float()

    def __len__(self):
        return len(self.vertices)

    def fetch_data_for_index(self, idx):
        return fetch_data_for_point(self.vertices[idx], self.feature_image, window=self.window, flip=self.flip,
                                    spacing=self.spacing)


class AEDataset(WholeVolumeAutoEncoderSequence, Dataset):
    def __init__(self, *args, batch_size=1, shuffle=False, metric_names=None, **kwargs):
        super().__init__(*args, batch_size=batch_size, shuffle=shuffle, metric_names=metric_names, **kwargs)

    def __len__(self):
        return len(self.epoch_filenames)

    def __getitem__(self, idx):
        item = self.epoch_filenames[idx]
        x, y = self.resample_input(item)
        return (torch.from_numpy(np.moveaxis(np.asarray(x), -1, 0)).float(),
                torch.from_numpy(np.moveaxis(np.asarray(y), -1, 0)).float())


class WholeVolumeSegmentationDataset(WholeVolumeSegmentationSequence, Dataset):
    def __init__(self, *args, batch_size=1, shuffle=False, metric_names=None, **kwargs):
        super().__init__(*args, batch_size=batch_size, shuffle=shuffle, metric_names=metric_names, **kwargs)
        with open('result_model12_pre/0327/model1/data_split/fold_5.json', 'r') as file:
            data = json.load(file)
            training_ids = data['training']
        tab_dir = os.path.join(project_dir,'table_data/tabunet_uni_map-2.xlsx')
        train = pd.read_excel(tab_dir)
        train = train[train['id'].isin(training_ids)]
        # print('train ===', train)
        # tab_dir = os.path.join(project_dir, 'table_data/clinic_data_cls_2.xlsx')
        # train = pd.read_excel(tab_dir)
        target = 'label'

        train = train.astype('str')
        types = train.dtypes
        # categorical_columns = ['','double','M1','M2','M3','M4','M5','M6','Caudate','Inular_ribbon',
        #                        'Lentiform_nucleus','Internal_capsule','sex','loc','shift','smoke',
        #                        'drink','diabete','myocardial','coronary','atria','hypertension','stroke']
        categorical_columns = ['']
        categorical_dims = {}
        for col in train.columns :
            if col in ['id']:
                continue
            if types[col] == 'object' :
                l_enc = LabelEncoder()
                train[col] = train[col].fillna("VV_likely")
                train[col] = l_enc.fit_transform(train[col].values)
                categorical_columns.append(col)
                # if col in categorical_columns:
                categorical_dims[col] = len(l_enc.classes_)
            else:
                train.fillna(train.loc[:, col].mean(), inplace=True)

        unused_feat = ['NIHSS','manual','double','score']


        features = [col for col in train.columns if col not in unused_feat + [target]]

        cat_idxs = [i-1 for i, f in enumerate(features) if f in categorical_columns]

        cat_dims = [categorical_dims[f] for i, f in enumerate(features) if f in categorical_columns]
        self.cat_id = cat_idxs
        self.cat_dim = cat_dims
        # grouped_features = [[0, 1, 2], [8, 9, 10]]
        data = features+[target]
        self.tab_info = train[data].values
        # print('info------------------------' , len(data))
        self.tab_data = {}
        ans = 0
        for row in (self.tab_info):
            # print('row' , ans , '=' , row)
            ans += 1
            key = row[0]
            # print('key == ', key)
            values = row[1:]
            self.tab_data[key] = values
        self.img_data = {}
        for i in tqdm(range((len(self.epoch_filenames)))):
            item = self.epoch_filenames[i]
            try:
                x, y = self.resample_input(item)
            except:
                print('error',item[-1])
            self.img_data[item[4]] = x,y

    def get_tab(self,index):
        # print('index==' , index)
        # print('table_data == ' , self.tab_data)

        tab_info = self.tab_data.get(index)
        if tab_info is None:
            print('erro_data',index)
        return tab_info[:-1],tab_info[-1]
    def get_data_info(self):
        return self.cat_id,self.cat_dim
    def __len__(self):
        return len(self.epoch_filenames)


    def __getitem__(self, idx):
        item = self.epoch_filenames[idx]
        # print('item == ', item)
        #x,y = self.resample_input(item)
        x, y = self.img_data[item[4]]
        tabx,taby = self.get_tab(item[4])
        return (torch.from_numpy(np.moveaxis(np.copy(x), -1, 0)).float(),
                torch.from_numpy(np.moveaxis(np.copy(y), -1, 0)).byte(),
                np.asarray(tabx,dtype=np.float32),
                np.asarray(taby,dtype=np.float32)
                )

class WholeVolumeSegmentationDataset_2(WholeVolumeSegmentationSequence, Dataset):
    def __init__(self, *args, batch_size=1, shuffle=False, metric_names=None, **kwargs):
        super().__init__(*args, batch_size=batch_size, shuffle=shuffle, metric_names=metric_names, **kwargs)
        with open('result_model12_pre/0327/model1/data_split/fold_3.json', 'r') as file:
            data = json.load(file)
            training_ids = data['validation']
        tab_dir = os.path.join(project_dir, 'table_data/tabunet_uni_map-2.xlsx')
        train = pd.read_excel(tab_dir)
        train = train[train['id'].isin(training_ids)]
        # tab_dir = os.path.join(project_dir, 'table_data/clinic_data_cls_2.xlsx')
        # train = pd.read_excel(tab_dir)
        target = 'label'

        train = train.astype('str')
        types = train.dtypes
        # categorical_columns = ['','double','M1','M2','M3','M4','M5','M6','Caudate','Inular_ribbon',
        #                        'Lentiform_nucleus','Internal_capsule','sex','loc','shift','smoke',
        #                        'drink','diabete','myocardial','coronary','atria','hypertension','stroke']
        categorical_columns = ['']
        categorical_dims = {}
        for col in train.columns :
            if col in ['id']:
                continue
            if types[col] == 'object' :
                l_enc = LabelEncoder()
                train[col] = train[col].fillna("VV_likely")
                train[col] = l_enc.fit_transform(train[col].values)
                categorical_columns.append(col)
                # if col in categorical_columns:
                categorical_dims[col] = len(l_enc.classes_)
            else:
                train.fillna(train.loc[:, col].mean(), inplace=True)

        unused_feat = ['NIHSS', 'manual', 'double','uni_vol' , 'sum_map','score']

        features = [col for col in train.columns if col not in unused_feat + [target]]

        cat_idxs = [i-1 for i, f in enumerate(features) if f in categorical_columns]

        cat_dims = [categorical_dims[f] for i, f in enumerate(features) if f in categorical_columns]
        self.cat_id = cat_idxs
        self.cat_dim = cat_dims
        # grouped_features = [[0, 1, 2], [8, 9, 10]]
        data = features+[target]
        self.tab_info = train[data].values
        # print('info------------------------' , len(data))
        self.tab_data = {}
        ans = 0
        for row in (self.tab_info):
            # print('row' , ans , '=' , row)
            ans += 1
            key = row[0]
            values = row[1:]
            self.tab_data[key] = values
        self.img_data = {}
        for i in tqdm(range((len(self.epoch_filenames)))):
            item = self.epoch_filenames[i]
            try:
                x, y = self.resample_input(item)
            except:
                print('error',item[-1])
            self.img_data[item[4]] = x,y

    def get_tab(self,index):
        # print('index==' , index)
        # print('table_data == ' , self.tab_data)
        tab_info = self.tab_data.get(index)
        if tab_info is None:
            print('erro_data',index)
        return tab_info[:-1],tab_info[-1]
    def get_data_info(self):
        return self.cat_id,self.cat_dim
    def __len__(self):
        return len(self.epoch_filenames)


    def __getitem__(self, idx):
        item = self.epoch_filenames[idx]
        #x,y = self.resample_input(item)
        x, y = self.img_data[item[4]]
        # print('item ===', item[4])
        tabx,taby = self.get_tab(item[4])
        return (torch.from_numpy(np.moveaxis(np.copy(x), -1, 0)).float(),
                torch.from_numpy(np.moveaxis(np.copy(y), -1, 0)).byte(),
                np.asarray(tabx,dtype=np.float32),
                np.asarray(taby,dtype=np.float32)
                )
class WholeVolumeSupervisedRegressionDataset(WholeVolumeSupervisedRegressionSequence, Dataset):
    def __init__(self, *args, batch_size=1, shuffle=False, **kwargs):
        super().__init__(*args, batch_size=batch_size, shuffle=shuffle, **kwargs)

    def __len__(self):
        return len(self.epoch_filenames)

    def __getitem__(self, idx):
        item = self.epoch_filenames[idx]
        x, y = self.resample_input(item)
        return (torch.from_numpy(np.moveaxis(np.asarray(x), -1, 0)).float(),
                torch.from_numpy(np.moveaxis(np.asarray(y), -1, 0)).float())


class WholeVolumeCiftiSupervisedRegressionDataset(WholeVolumeCiftiSupervisedRegressionSequence,
                                                  WholeVolumeSupervisedRegressionDataset):
    pass


class WindowedAEDataset(WindowedAutoEncoderSequence, Dataset):
    def __init__(self, *args, points_per_subject=1, **kwargs):
        super().__init__(*args, batch_size=points_per_subject, **kwargs)

    def __len__(self):
        return len(self.epoch_filenames)

    def __getitem__(self, idx):
        x, y = self.fetch_hcp_subject_batch(*self.epoch_filenames[idx])
        return (torch.from_numpy(np.moveaxis(np.asarray(x), -1, 1)).float(),
                torch.from_numpy(np.moveaxis(np.asarray(y), -1, 1)).float())
