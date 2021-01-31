from RD.rumor_model.liumeiqi.rumor_detection import MyGRU
from RD.rumor_model.liumeiqi import rumor_detection
import os
import json
import torch
from abc import ABC
import jieba.analyse
from torch import nn
import torch.nn.functional as func
from gensim.models import Word2Vec
from torch.nn.utils.rnn import pad_sequence
# path1 = path_bace = os.path.abspath('.')+'\\val\\'
path1 = 'D:\\RumorDetection\\RD_website\\RD\\static\\data\\'
dirs = os.listdir(path1)  # 所有的文件名
path_bace = os.path.abspath('.')
path4 = path_bace+'\\RD_GRU_model02.pt'
model = torch.load(path4)  # 加载模型
if torch.cuda.is_available():
    model = model.cuda()
def read_list(file):
    filepath = path1+file
    with open(filepath, 'rb') as load_f:
        load_dict = json.load(load_f)
    return load_dict


labels = []
labels_pre = []


for filename in dirs:
    posts = read_list(filename)  # 返回列表
    for post in posts:
        labels.append(post[0])
        label = rumor_detection.run_val(post,model)
        labels_pre.append(label)
num_c = 0
num = len(labels)
for i in range(num):
    if labels[i] == labels_pre[i]:
        num_c += 1
print(labels)
print(labels_pre)
print(num_c / num)