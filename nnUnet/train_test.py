import pandas as pd
import torch
import torchvision
import xlwt as xlwt
import os
import shutil

data_test = pd.read_excel('/home/xiao/Desktop/record/score_all_gt_test_20.xlsx')
data_train = pd.read_excel('/home/xiao/Desktop/record/score_all_gt_train_20.xlsx')
data_valid = pd.read_csv('/home/xiao/nnUNet/result.csv')
#data_test = pd.read_excel('/home/xiao/Desktop/record/result-high.xls')
#data_train = pd.read_excel('/home/xiao/Desktop/record/result-high.xls')

ids_test = data_test['id']
ids_train = data_train['id']
ids_valid = data_valid['ct_name']
# print(ids_test.values.size)
path_test = '/media/xiao/DATA/registration_Orin/registration_old/preprocessed3/test'#保存test路径
path_train = '/media/xiao/DATA/registration_Orin/registration_old/preprocessed3/train'#b保存train路径
path_1 = '/media/xiao/DATA/registration_Orin/registration_old/preprocessed3/xinhua'
path_2 = '/media/xiao/DATA/registration_Orin/registration_old/preprocessed3/dongfang'
path_3 = '/media/xiao/DATA/registration_Orin/registration_old/preprocessed3/tongji'
# path = '/media/xiao/DATA/registration_Orin/registration_old/preprocessed3/tongji'
# path = '/media/xiao/DATA/registration_Orin/registration_old/preprocessed3/puzhongxin'
# path = '/media/xiao/DATA/registration_Orin/registration_old/preprocessed3/xinhua'
def add_prefix_files(path):             #定义函数名称               #准备添加的前缀内容
    old_names = os.listdir(path)  #取路径下的文件名，生成列表
    for old_name in old_names:      #遍历列表下的文件名
            file_name = old_name
            #print(file_name)
            for id in ids_train:
                #id = id[2:-2]
                if str(file_name) == str(id):
                    for id_valid in ids_valid:
                        id_valid2 = id_valid[2:-2]
                        # print(id_valid2)
                        if id_valid2 == str(file_name):
                            # print(file_name)
                            new_name = os.path.join(path_train, old_name)
                            #new_name_test = os.path.join(path_test, old_name)
                            ori_name = os.path.join(path, old_name)
                            if os.path.isdir(new_name):
                                continue
                            else:
                                shutil.copytree(ori_name, new_name)
                                #shutil.copytree(ori_name, new_name_test)



def del_back(path):
    names = os.listdir(path)
    for old_name in names:
        new_name = old_name.replace(' ','')
        print(new_name)
        os.rename(os.path.join(path,old_name),os.path.join(path,new_name))

def count_dir(path):
    names = os.listdir(path)
    print(len(names))


import os
import shutil
import random
# Example usage:
# 原始文件夹路径
source_folders = [path_1, path_2, path_3]

# 目标文件夹路径
target_folder = '/media/xiao/DATA/registration_Orin/registration_old/preprocessed3'
train_folder = os.path.join(target_folder, 'train')
val_folder = os.path.join(target_folder, 'val')
test_folder = os.path.join(target_folder, 'test')

# 创建目标文件夹和子文件夹
os.makedirs(train_folder, exist_ok=True)
os.makedirs(val_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

# 初始化计数器
total_folders = 0
for folder in source_folders:
    total_folders += len([name for name in os.listdir(folder) if os.path.isdir(os.path.join(folder, name))])

# 计算每个集合应包含的文件夹数量
train_folders = int(0.7 * total_folders)
val_folders = int(0.2 * total_folders)
test_folders = total_folders - train_folders - val_folders

# 存储每个文件夹的路径
all_folders = []
for folder in source_folders:
    all_folders.extend(
        [os.path.join(folder, name) for name in os.listdir(folder) if os.path.isdir(os.path.join(folder, name))])

# 随机打乱文件夹的顺序
random.shuffle(all_folders)

# 分配文件夹到训练、验证和测试集
train_folders_selected = all_folders[:train_folders]
val_folders_selected = all_folders[train_folders:train_folders + val_folders]
test_folders_selected = all_folders[train_folders + val_folders:]


# 复制文件夹到相应的目标文件夹
def copy_folder(src, dst):
    shutil.copytree(src, dst)


for folder in train_folders_selected:
    dst = os.path.join(train_folder, os.path.basename(folder))
    copy_folder(folder, dst)

for folder in val_folders_selected:
    dst = os.path.join(val_folder, os.path.basename(folder))
    copy_folder(folder, dst)

for folder in test_folders_selected:
    dst = os.path.join(test_folder, os.path.basename(folder))
    copy_folder(folder, dst)

print(
    f"文件夹已成功划分为训练集({train_folders}个文件夹)、验证集({val_folders}个文件夹)和测试集({test_folders}个文件夹)。")

