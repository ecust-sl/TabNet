import os
import random
import shutil

# train_root = '/home/xiao/nnUNet/nnUNetFrame/DATESET/nnUNet_raw/nnUNet_raw_data/Task504_All'
# test_root = '/home/xiao/nnUNet/nnUNetFrame/DATESET/nnUNet_raw/nnUNet_raw_data/Task504_All'
#
#
# def moveFile(trainDir, testDir):
#     pathDir = os.listdir(os.path.join(trainDir,'imagesTr'))  # 取图片的原始路径
#     filenumber = len(pathDir)
#     # rate1 = 0.8  # 自定义抽取csv文件的比例，比方说100张抽80个，那就是0.8
#     rate1 = 0.2
#     picknumber1 = int(filenumber * rate1)  # 按照rate比例从文件夹中取一定数量的文件
#     sample1 = random.sample(pathDir, picknumber1)  # 随机选取picknumber数量的样本
#     for name in sample1:
#         mask_name = name
#         shutil.move(os.path.join(trainDir,'imagesTr',name), os.path.join(testDir,'imagesTs',name))
#         shutil.move(os.path.join(train_root,'labelsTr',mask_name), os.path.join(testDir, 'labelsTs', mask_name))
#
#
# if __name__ == '__main__':
#     moveFile(train_root, test_root)
# path = '/media/xiao/DATA/registration_Orin/registration_old/preprocessed3/all_data'
# path_tongji = '/media/xiao/DATA/registration_Orin/registration_old/preprocessed3/tongji'
# path_xinhua = '/media/xiao/DATA/registration_Orin/registration_old/preprocessed3/valid_xinhua'
# path_puzhongxin = '/media/xiao/DATA/registration_Orin/registration_old/preprocessed3/valid_puzhongxin'
# path_dongfang = '/media/xiao/DATA/registration_Orin/registration_old/preprocessed3/valid_dongfang'
# dir1 = os.listdir(path_dongfang)
# dir2 = os.listdir(path_puzhongxin)
# dir3 = os.listdir(path_xinhua)
# dirs = dir1 + dir2 + dir3
# # print(dirs)
# for file in os.listdir(path):
#     if str(file) not in dirs:
#         print('111')
#         src_path = os.path.join(path, file)
#         new_path = os.path.join(path_tongji, file)
#         if os.path.isdir(new_path):continue
#         else:shutil.copytree(src_path, new_path)
# path_img = '/home/xiao/nnUNet/nnUNetFrame/DATASET/nnUNet_raw/Dataset505_All/imagesTr'
# path_label = '/home/xiao/nnUNet/nnUNetFrame/DATASET/nnUNet_raw/Dataset505_All/labelsTr'
# dir_img = os.listdir(path_img)
# dir_label = os.listdir(path_label)
# dir_label = [label[:-7] for label in dir_label]
# for img in dir_img:
#     img = img[:-11]
#     if img not in dir_label:
#         print(img)
import os
import json

# 定义训练和验证文件夹的路径
train_folder = '/media/xiao/DATA/registration_Orin/registration_old/preprocessed3/train'
val_folder = '/media/xiao/DATA/registration_Orin/registration_old/preprocessed3/val'


# 提取文件夹中的文件名
def extract_filenames(folder_path):
    return [f for f in os.listdir(folder_path)]


# 提取训练和验证文件夹的文件名
train_filenames = extract_filenames(train_folder)
val_filenames = extract_filenames(val_folder)

# 创建JSON数据
data = {
    'train': train_filenames,
    'val': val_filenames
}

# 将JSON数据写入文件
with open('splits_final.json', 'w') as file:
    json.dump(data, file, indent=4)

print("文件名已成功提取并保存到filenames.json文件中。")

