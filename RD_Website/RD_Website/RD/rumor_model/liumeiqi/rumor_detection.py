#  完整的预处理文件
import os
import json
import torch
from abc import ABC
import jieba.analyse
from torch import nn
import torch.nn.functional as func
from gensim.models import Word2Vec
from torch.nn.utils.rnn import pad_sequence
# 验证集原文件
# path_bace = os.path.abspath('.') 直接运行路径和django运行路径不同
path_bace = os.path.abspath('')
path_bace = os.path.join(path_bace, 'RD\\rumor_model\\liumeiqi')
# print("路径:",path_bace )
# path1 = path_bace+'\\val\\'
# 停用词文件位置
path2 = path_bace+'\\stopwords.txt'
# using word 文件
path3 = path_bace+'\\word_counter_5.json'
# 谣言检测模型保存位置
path4 = path_bace+'\\RD_GRU_model02.pt'
# Word2vec模型保存位置
path5 = path_bace+'\\MyW2Model_20'
#  文件夹下所有文件的名字，验证集名字
length_seq = 50  # 一个句子的长度
length_word = 20


# 停用词
def fun_stopwords():
    f_stopwords = [line1.strip() for line1 in
                   open(path2, encoding='UTF-8').readlines()]
    return f_stopwords


# 保留词
def fun_using_word():
    with open(path3, 'rb') as use:
        file_using = json.load(use)
    return file_using


#  进行分词并去停用词
def removestops(text_sd):
    out_list = []
    text_sd = text_sd.replace(' ', '')  # 去掉空格
    sent_list = jieba.cut(text_sd, cut_all=False, HMM=True)
    for word in sent_list:
        if word not in stopwords:
            out_list.append(word)
    return out_list


#  保留词频大于五的词
def reverse(text_sd):
    out_list = []
    for word in text_sd:
        if word in using_word:
            out_list.append(word)
    return out_list


class MyGRU(nn.Module, ABC):
    def __init__(self, input_dim, n_hidden):  # , n_tag 可以做多分类
        super(MyGRU, self).__init__()
        self.gru = nn.GRU(input_dim, n_hidden, batch_first=True)
        self.linear1 = nn.Linear(n_hidden, 2)

    def forward(self, x):  # 一个batch的所有数据
        x, _ = self.gru(x)
        _ = _[0]
        # _ = _.squeeze(0)
        _ = self.linear1(_)  # linear层之后，x的size为(len(x),n_tag)
        y = func.log_softmax(_, dim=1)  # 对第1维先进行softmax计算，然后log一下。y的size为(len(x),n_tag)。
        return y


model_w2 = Word2Vec.load(path5)  # 加载词向量模型


def word_2vec(seq_pref):
    vector = []
    for word in seq_pref:
        vector_20 = list(model_w2[word])
        vector = vector + vector_20
    if len(vector) < length_word*length_seq:
        for i in range(0, length_word*length_seq-len(vector)):
            vector.append(0)
    return vector  # 一个句子的向量(列表)


def fixed_length(list_seq):
    fixed_l = []
    with torch.no_grad():
        for seq in list_seq:  # seq是一列表[南海，事件，西沙，群岛]
            if len(seq) < length_seq:
                seq_pref = seq  # seq_pre是长度小于length_seq的列表[南海，事件，西沙，群岛]
            else:
                seq_pref = seq[:length_seq]
            seq_tensor = torch.FloatTensor(word_2vec(seq_pref))  # 一个句子的张量
            fixed_l.append(seq_tensor)
        fixed_l_v = pad_sequence(fixed_l, batch_first=True)
        return fixed_l_v  # 一个帖子（json文件）所有句子张量(定长经过填充)


def run_val(text_list, model):
    text = []
    for post in text_list[1:]:
        seq = post['text']
        seq = removestops(seq)  # 进行分词，去停用词
        seq = reverse(seq)  # 保留词频大于五的词
        if seq:
            text.append(seq)
    tensor_text = fixed_length(text)  # 一个帖子的定长张量[[句子]，[句子]]
    tensor_text = tensor_text.unsqueeze(0)
    # print(tensor_text.size())
    if torch.cuda.is_available():
        tensor_text = tensor_text.cuda()
    label_pre = model(tensor_text)
    _, predict_label = torch.max(label_pre, 1)
    predict_label = list(predict_label.cpu().numpy())
    return str(predict_label[0])


stopwords = fun_stopwords()  # 创建停用词表
stopwords.append('转发')
stopwords.append('微博')
using_word = fun_using_word()  # 创建保留词表


if __name__ == '__main__':

    model = torch.load(path4)  # 加载模型
    if torch.cuda.is_available():
        model = model.cuda()