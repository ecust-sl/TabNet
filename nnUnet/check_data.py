import numpy as np
import pandas as pd
import xlwt as xlwt
import csv

def check_NIHSS(a):
    return a>=0.8

def check_NIHSS2(a):
    return a<0.8
data = pd.read_csv('/home/xiao/nnUNet/new_score/score_9.26/score_test_9.26_0.00.csv')
w = open('/home/xiao/nnUNet/new_score/score_9.27/res2.csv','w')
r = open('/home/xiao/nnUNet/new_score/score_9.27/area_equal.csv','r')
data3 = pd.read_csv('/home/xiao/nnUNet/new_score/score_9.27/area_equal.csv')
data4 = pd.read_csv('/home/xiao/nnUNet/new_score/score_9.27/area_equal.csv')
A_value = data3['score']
B_value = data3['人工评分改']
A_value = list(map(float,A_value))
B_value = list(map(float,B_value))
npB = np.array(B_value)
npA = np.array(A_value)
sub_valye = np.maximum(npA-npB,npB-npA)
data4['diff'] = sub_valye
data4.to_csv('/home/xiao/nnUNet/new_score/score_9.27/area_equal.csv')

def abss(x):
    if x < 0:
        return -x
    else:
        return x







#
# print(data_id)
# print(data_score)






