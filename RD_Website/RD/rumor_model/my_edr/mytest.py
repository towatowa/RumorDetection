# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 22:04:31 2021

@author: Ps
"""

import os
import json
import numpy as np
import re
import pkuseg
import pickle
from RD.rumor_model.my_edr.ERD_CN import rdm_model, sent_pooler, rdm_classifier
import torch
import gensim
import sys
import codecs

seg = pkuseg.pkuseg()
# data_dir = "./data_all/val"

model_file_name = './RD/rumor_model/my_edr/倚天屠龙记_词向量模型.txt'
word2vec = gensim.models.word2vec.Word2Vec.load(model_file_name)
print("load word2vec finished")

def transIrregularWord(line):
    '''
    替换微博中不规范的文字
    '''

    if not line:
        return ''
    line.lower()
    line = re.sub("@[^ ]*", "{ 提到某人 }", line)
    line = re.sub("#[^ ]*", "{ 某个话题 }", line)
    line = re.sub("http(.?)://[^ ]*", "{ 网页链接 }", line)
    return seg.cut(line)

def load_val_data(data_dir):
    '''
    加载验证集
    '''
    global data, files, data_ID, data_len, eval_flag, data_y
    data = {}
    files = []
    data_ID = []
    data_len = []
    
    num_label0 = 0
    num_label1 = 0
    
    data_list = os.listdir(data_dir)#枚举所有文件名字

    count = 0
    for file in data_list:
        count += 1
        if count > 20:
            break
        eid = int(file.split('.')[0])
        data_ID.append(eid)#将文件的名字加入列表中
        data_file = os.path.join(data_dir, '%s.json'%eid)#生成每一个文本的绝对地址
        data[eid] = {}
        with open(data_file, encoding='utf-8') as fr:
            event_data = json.load(fr)[1:]
            texts = [tweet['original_text'] for tweet in event_data]
            created_at = [tweet['t'] for tweet in event_data]
            idxs = np.array(created_at).argsort().tolist()
            data[eid]['text'] = [transIrregularWord(texts[idx]) for idx in idxs]#保存文本
            data[eid]['created_at'] = [created_at[idx] for idx in idxs]#保存id
            data_len.append(len(texts))#保存每一段文本长度
    # with open('val_dict.txt', 'wb') as handle:
    #     pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    # np.save("val_ID.npy", data_ID)
    # np.save("val_len.npy", data_len)


def load_val_data_new(data_file):
    '''
    加载验证集
    '''
    global data, files, data_ID, data_len, eval_flag, data_y
    data = {}
    files = []
    data_ID = []
    data_len = []

    num_label0 = 0
    num_label1 = 0

    with open(data_file, 'r',encoding='utf-8') as f:
        all_data = json.load(f)
        file_num = len(all_data)  # liumeiqi

    count = 0
    for i,d in enumerate(all_data):
        count += 1
        if count > 20:
            break
        eid = i
        data_ID.append(eid)  # 将文件的名字加入列表中
        # data_file = os.path.join(data_dir, '%s.json' % eid)  # 生成每一个文本的绝对地址
        data[eid] = {}

        event_data = d[1:]
        texts = [tweet['original_text'] for tweet in event_data]
        created_at = [tweet['t'] for tweet in event_data]
        idxs = np.array(created_at).argsort().tolist()
        data[eid]['text'] = [transIrregularWord(texts[idx]) for idx in idxs]  # 保存文本
        data[eid]['created_at'] = [created_at[idx] for idx in idxs]  # 保存id
        data_len.append(len(texts))  # 保存每一段文本长度
        data[eid]['real'] = d[0]
    return file_num


def get_df_batch(start, batch_size, new_data_len=[], cuda=False):
    '''
    得到一个batch的内容，
    返回,data_x:一个batch中所有事件的词向量列表[batch* data_len* 300]
    m_data_y :一个batch所有事件对应的标签
    m_data_len: 一个batch所有事件的posts个数,源和回复
    '''
    data_x = []     # 存放text
    data_id = []
    data_y = []
    m_data_len = np.zeros([batch_size], dtype=np.int32) # 存放事件的推文个数 长度
    miss_vec = 0
    hit_vec = 0
    if len(new_data_len) > 0:
        t_data_len = new_data_len

    else:
        t_data_len = data_len  # t_data_len = data_len[693,44..],记录微博评论的次数。
    mts = start * batch_size
    if mts >= len(data_ID):
        mts = mts % len(data_ID)
    for i in range(batch_size):
        #遍历一个batch的内容
        data_y.append(data[data_ID[mts]]['real'])
        m_data_len[i] = t_data_len[mts]  # 将一个batch的长度给m_data_len[] :长度为评论的次数
        data_id.append(data_ID[mts])
        seq = []
        for j in range(t_data_len[mts]):  # 将一个事件的原微博和评论都转成词向量
            # 遍历一篇微博的所有text
            sent = []       # 存放某个单词的词向量
            t_words = data[data_ID[mts]]['text'][j]   # 一个post的text # t_words :一篇微博的第j个text,
            if len(t_words) == 0:
                # print("ID:%s   j:%3d    empty sentence:"%(valid_data_ID[mts], j), t_words) # 11.6改
                print("ID:%s   j:%3d    empty sentence:"%(data_ID[mts], j), t_words)
                continue
            for k in range(len(t_words)):
                m_word = t_words[k]     # m_word:某个单词
                try:
                    sent.append( torch.tensor([word2vec[m_word]], dtype=torch.float32) )
                except KeyError:
                    miss_vec += 1  # 记录有多少次不在word2vec中，
                    # sent.append(torch.tensor([word2vec['{'] + word2vec['未知'] + word2vec['词'] + word2vec['}']],
                    #                          dtype=torch.float32))
                    sent.append(torch.tensor(word2vec['啊'] +[word2vec['张三丰'] +  word2vec['武当'] ], dtype=torch.float32) )
                except IndexError:
                    raise
                else:
                    hit_vec += 1
            sent_tensor = torch.cat(sent)  # 将所有单词的词向量tensor 合并成一个列表
            seq.append(sent_tensor)     #   将一个事件的所有post词存储在list中,[[t_word,300],[t_word,300]....]
        data_x.append(seq)      # 将一个事件的所有posts词向量加载进来.
        mts += 1
        if mts >= len(data_ID): # read data looply
            mts = mts % len(data_ID)
    return data_x, m_data_len, data_id, data_y

#加载已有模型参数
# log_dir = os.path.join(sys.path[0], "ERD_CN")
# pretrained_file = "%s/ERD_best.pkl" % log_dir
# checkpoint = torch.load(pretrained_file,map_location='cpu')
# sent_pooler.load_state_dict(checkpoint['sent_pooler'])
# rdm_model.load_state_dict(checkpoint["rmdModel"])
# rdm_classifier.load_state_dict(checkpoint["rdm_classifier"])

# #将模型设置成验证状态
# rdm_model.eval()
# #获得验证集
# load_val_data(data_dir)
# #设置每一回合训练文本数量
# batch_size = 16
# #计算一共有几个steo
# steps = np.ceil(len(data) / batch_size)

# out = codecs.open('check.txt', 'w', encoding='utf-8')
# for i in range(int(steps)):
#     #获得每一个batch的数据集
#     x, x_len,x_ID = get_df_batch(i, batch_size)
#     #padding数据集
#     seq = sent_pooler(x)
#     #得到模型输出
#     rdm_hiddens = rdm_model(seq)
#     batchsize, _, _ = rdm_hiddens.shape # 求batch,或许最后一个batch_size,不是满的
#     rdm_outs = torch.cat(
#          [rdm_hiddens[i][x_len[i]-1].unsqueeze(0) if (x_len[i]-1) < 1565 else
#          rdm_hiddens[i][1564].unsqueeze(0) for i in range(batchsize)]
#         # a list of tensor, where the ndim of tensor is 1 and the shape of tensor is [hidden_size]
#     )    # 对rdm_hiddens,选取向量, rdm_hiddens=[batch_size, max_twitter_len，256] -> rdm_outs[batchsize, 256]
#     rdm_scores = rdm_classifier(
#         rdm_outs
#     )  # rdm_scores:[batchsize, 2],是否是谣言的概率
#     rdm_preds = rdm_scores.argmax(axis=1)  # argmax(axis=0/1) argmax返回的是最大数的索引.[batchsize], 深度学习输出结果
#     rdm_preds = rdm_preds.numpy()
#
#     for j in range(len(rdm_preds)):
#         out.write(str(x_ID[j])+'\t'+str(rdm_preds[j])+'\n')
#
# out.close()

def run_eval(data_file,checkpoint):
    sent_pooler.load_state_dict(checkpoint['sent_pooler'])
    rdm_model.load_state_dict(checkpoint["rmdModel"])
    rdm_classifier.load_state_dict(checkpoint["rdm_classifier"])

    # 将模型设置成验证状态
    rdm_model.eval()
    # 获得验证集
    file_num = load_val_data_new(data_file)
    # 设置每一回合训练文本数量
    batch_size = file_num
    # 计算一共有几个steo
    steps = np.ceil(len(data) / batch_size)

    labels = []
    labels_pre = []
    for i in range(int(steps)):
        # 获得每一个batch的数据集
        x, x_len, x_ID, y = get_df_batch(i, batch_size)
        # padding数据集
        seq = sent_pooler(x)
        # 得到模型输出
        rdm_hiddens = rdm_model(seq)
        batchsize, _, _ = rdm_hiddens.shape  # 求batch,或许最后一个batch_size,不是满的
        rdm_outs = torch.cat(
            [rdm_hiddens[i][x_len[i] - 1].unsqueeze(0) if (x_len[i] - 1) < 1565 else
             rdm_hiddens[i][1564].unsqueeze(0) for i in range(batchsize)]
            # a list of tensor, where the ndim of tensor is 1 and the shape of tensor is [hidden_size]
        )  # 对rdm_hiddens,选取向量, rdm_hiddens=[batch_size, max_twitter_len，256] -> rdm_outs[batchsize, 256]
        rdm_scores = rdm_classifier(
            rdm_outs
        )  # rdm_scores:[batchsize, 2],是否是谣言的概率
        rdm_preds = rdm_scores.argmax(axis=1)  # argmax(axis=0/1) argmax返回的是最大数的索引.[batchsize], 深度学习输出结果
        rdm_preds = rdm_preds.numpy()

        # for j in range(len(rdm_preds)):
        #     out.write(str(x_ID[j]) + '\t' + str(rdm_preds[j]) + '\n')

        for j in range(len(rdm_preds)):
            labels_pre.append(str(rdm_preds[j]))

        labels.extend(y)

    return labels, labels_pre


if __name__ == '__main__':
    log_dir = os.path.join(sys.path[0], "ERD_CN")
    pretrained_file = "%s/ERD_best.pkl" % log_dir
    checkpoint = torch.load(pretrained_file, map_location='cpu')
    labels, labels_pre = run_eval('./file/weibo.json',checkpoint)
    print(labels,labels_pre)
