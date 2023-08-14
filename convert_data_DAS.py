import pandas as pd
import numpy as np
import os
import json
# data = np.loadtxt(fname=os.path.join('D:/Desktops/CNN-BiLSTM/wajue.xlsx'))
# #data = pd.read_excel('D:/Desktops/CNN-BiLSTM/wajue.xlsx')
# #print(data)
# result=data.T
# #print(result)
# df=pd.DataFrame(result)
# print(df)
#
# df.to_excel('Result_no_label.xlsx',index = False,header=False)
# -*- coding: utf-8 -*-
# ----------------------------------------------------#
#   转换原始数据
# ----------------------------------------------------#
# 窗口大小
window_height = 20000
window_step = 20000
location=4942
# with open(r'C:\Users\DELL\Desktop\1-大赛数据集\标注.json', 'r', encoding='utf-8') as f:
#     label_explain = json.load(f)
# pass
# dir = r'C:\Users\DELL\Desktop\1-大赛数据集'
# save_dir = r'data'
#
# abnormal_arr = list()
# abnormal_label = list()
# for dt in label_explain['data']:
#     #print(dt)
#     data = np.loadtxt(fname=os.path.join(dir, dt['file_name']))
#     #abnormal_data = data[:, location: location+100]
#     #print(abnormal_data.shape)
#     # while y + window_height < len(abnormal_data):
#     x1, x2 = dt["boxes"][0]
#     x1, x2 = map(int, [x1, x2])
#     #print(x1,x2)
#     label = dt["labels"][0]
#     abnormal_data = data[:, x1: x2]
#     #print(label)
#     #print(abnormal_data[:,1])
#     columns = abnormal_data[0]
#     #print(len(abnormal_data))
#     #print(abnormal_data)
#     for i in range(len(columns)):
#         y = 0
#         #print(abnormal_data[:, i])
#         while y + window_height < len(abnormal_data):
#             abnormal_arr.append(abnormal_data[:,i][y:y + window_height])
#             #print(abnormal_data[:,i+location][y:y + window_height])
#             while i+x1 >= x1 and i+x2 <= x2+x2-x1:
#                 #print(label)
#                 abnormal_label.append(label)
#                 #print(abnormal_label)
#                 break
#             else:
#                 abnormal_label.append(6)
#             y = y + window_step
#
# # abnormal_arr = np.array(abnormal_arr)
# # abnormal_label = np.array(abnormal_label)

#----------------------------------------------------------------------------------------------------------------------------#
with open(r'D:\Desktops\CDIL-CBAM-BiLSTM\Perimeter Security Dataset\标签.json', 'r', encoding='utf-8') as f:
    label_explain = json.load(f)
pass
dir = r'D:\Desktops\CDIL-CBAM-BiLSTM\Perimeter Security Dataset'
save_dir = r'data'

abnormal_arr = list()
abnormal_label = list()
for dt in label_explain['data']:
    #print(dt)
    data = np.loadtxt(fname=os.path.join(dir, dt['file_name']))
    #abnormal_data = data[:, location: location+100]
    #print(abnormal_data.shape)
    # while y + window_height < len(abnormal_data):
    x1, x2 = dt["boxes"][0]
    x1, x2 = map(int, [x1, x2])
    #print(x1,x2)
    label = dt["labels"][0]
    abnormal_data = data[:, x1: x2]
    #print(label)
    #print(abnormal_data[:,1])
    columns = abnormal_data[0]
    #print(len(abnormal_data))
    #print(abnormal_data)
    for i in range(len(columns)):
        y = 0
        #print(abnormal_data[:, i])
        while y + window_height < len(abnormal_data):
            abnormal_arr.append(abnormal_data[:,i][y:y + window_height])
            #print(abnormal_data[:,i+location][y:y + window_height])
            while i+x1 >= x1 and i+x2 <= x2+x2-x1:
                #print(label)
                abnormal_label.append(label)
                #print(abnormal_label)
                break
            else:
                abnormal_label.append(6)
            y = y + window_step

abnormal_arr = np.array(abnormal_arr)
abnormal_label = np.array(abnormal_label)
# # save_dir = r'E:\2-瀑布图数据\训练数据'
save_dir = r'data'
np.save(os.path.join(save_dir, 'DAS_20k_arr.npy'), abnormal_arr)
np.save(os.path.join(save_dir, 'DAS_20k_label.npy'), abnormal_label)



