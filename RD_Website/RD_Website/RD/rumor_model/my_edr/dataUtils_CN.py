#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import os
import time
import datetime
import numpy as np
import gensim
import random
import math
import re
import pickle
import torch
import torch.nn as nn
# In[26]:

import pkuseg
# In[45]:

from sklearn.metrics import accuracy_score

# In[2]:

files = []
data = {}
data_ID = []
data_len = []
data_y = []

valid_data_ID = []
valid_data_y = []
valid_data_len = []

# In[27]:


# with open("./word2vec_CN.pkl", "rb") as handle:
#     word2vec = pickle.load(handle)

# word2vec = gensim.models.KeyedVectors.load_word2vec_format(r'D:\D\python\pycharm\ERD_pytorch\hadoop\word2vec.bin', binary=True, encoding="utf-8")
#加载训练好的word2vec模型
# model_file_name = '倚天屠龙记_词向量模型.txt'
# word2vec = gensim.models.word2vec.Word2Vec.load(model_file_name)
# print("load word2vec finished")

# In[35]:

reward_counter = 0
eval_flag = 0

seg = pkuseg.pkuseg()

# In[37]:

def get_data_ID():
    global data_ID
    return data_ID

def get_data_len():
    global data_len
    return data_len

def get_curtime():
    return time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))

def list_files(data_path):
    global data, files
    fs = os.listdir(data_path)
    for f1 in fs:
        tmp_path = os.path.join(data_path, f1)
        if not os.path.isdir(tmp_path):
            if tmp_path.split('.')[-1] == 'json':
                files.append(tmp_path)
        else:
            list_files(tmp_path)

def str2timestamp(str_time):
    month = {'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04',
             'May': '05', 'Jun': '06', 'Jul': '07', 'Aug': '08',
             'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'}
    ss = str_time.split(' ')
    m_time = ss[5] + "-" + month[ss[1]] + '-' + ss[2] + ' ' + ss[3]
    d = datetime.datetime.strptime(m_time, "%Y-%m-%d %H:%M:%S")
    t = d.timetuple()
    timeStamp = int(time.mktime(t))
    return timeStamp

def data_process(file_path):
    ret = {}
    ss = file_path.split("/")
    data = json.load(open(file_path, mode="r", encoding="utf-8"))
    # 'Wed Jan 07 11:14:08 +0000 2015'
    # print("SS:", ss)
    ret[ss[6]] = {'label': ss[5], 'text': [data['text'].lower()], 'created_at': [str2timestamp(data['created_at'])]}
    return ret

def transIrregularWord(line):
    if not line:
        return ''
    line.lower()
    line = re.sub("@[^ ]*", "{ 提到某人 }", line)
    # .sub(),正则表达式，执行替换 将前面的"@[^ ]*",替换为"{ 提到某人 }"，替换目标line
    line = re.sub("#[^ ]*", "{ 某个话题 }", line)
    line = re.sub("http(.?)://[^ ]*", "{ 网页链接 }", line)
    return seg.cut(line)

    # def load_data_fast():


def load_data_fast():
    global data, data_ID, data_len, data_y, valid_data_ID, valid_data_y, valid_data_len
    with open("data/weibo_dict.txt", "rb") as handle:
        data = pickle.load(handle)
    data_ID = np.load("data/weibo_ID.npy", encoding="latin1").tolist()  # tolist()将数组转换成列表
    data_len = np.load("data/weibo_len.npy", encoding="latin1").tolist()
    data_y = np.load("data/weibo_y.npy", encoding="latin1").tolist()
    valid_data_ID = np.load("data/test_weibo_ID.npy", encoding="latin1").tolist()
    valid_data_len = np.load("data/test_weibo_len.npy", encoding="latin1").tolist()
    valid_data_y = np.load("data/test_weibo_y.npy", encoding="latin1").tolist()
    max_sent = max( map(lambda value: max(map(lambda txt_list: len(txt_list), value['text']) ), list(data.values()) ) )   # max_sent:最长一篇微博内容有多少词
    print("max_sent:", max_sent, ",  max_seq_len:", max(data_len))   # data_len:表示一篇微博的有多少条评论和转发次数 如；4016873519.json44个评论转发
    eval_flag = int(len(data_ID) / 4) * 3   # （60/4）*3=45
    print(eval_flag)
    print("{} data loaded".format(len(data))) # 加载了6条微博


def sortTempList(temp_list):
    time = np.array([item[0] for item in temp_list])
    posts = np.array([item[1] for item in temp_list])
    idxs = time.argsort().tolist()
    rst = [[t, p] for (t, p) in zip(time[idxs], posts[idxs])]
    del time, posts
    return rst


def load_data(weibo_file, weibo_dir):
    '''处理数据，已修到process_data'''
    global data, files, data_ID, data_len, eval_flag, data_y
    data = {}
    files = []
    data_ID = []
    data_len = []
    
    with open(weibo_file) as fr:
        for line in fr:
            s = line.split('\t')
            eid = s[0].strip("eid:")
            label = int(s[1].strip('label:'))
            data_ID.append(eid)
            if label == 1:
                data_y.append([0, 1])
            elif label == 0:
                data_y.append([1, 0])

    for eid in data_ID:
        data_file = os.path.join(weibo_dir, "%s.json"%eid)
        data[eid] = {}
        with open(data_file, encoding='utf-8') as fr:
            event_data = json.load(fr)
            texts = [tweet['original_text'] for tweet in event_data]
            created_at = [tweet['t'] for tweet in event_data]
            idxs = np.array(created_at).argsort().tolist()
            data[eid]['text'] = [transIrregularWord(texts[idx]) for idx in idxs]
            data[eid]['created_at'] = [created_at[idx] for idx in idxs]
            data_len.append(len(texts))



def get_df_batch(start, batch_size, new_data_len=[], cuda=False):
    '''
    得到一个batch的内容，
    返回,data_x:一个batch中所有事件的词向量列表[batch* data_len* 300]
    m_data_y :一个batch所有事件对应的标签
    m_data_len: 一个batch所有事件的posts个数,源和回复
    '''
    data_x = []     # 存放text
    m_data_y = np.zeros([batch_size, 2], dtype=np.int32) # 存放标签
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
        m_data_y[i] = data_y[mts]  # 将一个batch的标签给m_data_y[]
        m_data_len[i] = t_data_len[mts]  # 将一个batch的长度给m_data_len[] :长度为评论的次数
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
                    sent.append( torch.tensor(word2vec['啊'] +[word2vec['张三丰'] +  word2vec['武当'] ], dtype=torch.float32) )
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
    return data_x, m_data_len, m_data_y


from RD.rumor_model.my_edr.config import *
def get_rl_batch(ids, seq_states, stop_states, counter_id, start_id, total_data):
    # x, y, ids, seq_states, max_id = get_rl_batch(ids, seq_states, isStop, max_id, 0, 0)
    '''
    取batch个事件的某个推文词向量，标签，
    返回input_x[batch_size,100,300],一个batch事件中前100个text的词向量 ///或许是前20个句子.的词向量
    input_y:[batch_size, 2] :对应事件的表签
    ids:还是序列,没变
    seq_states:应该都变成1
    counter_id:=batch_size=20
    '''
    input_x = np.zeros([FLAGS.batch_size, FLAGS.max_sent_len, FLAGS.embedding_dim], dtype=np.float32)
    input_y = np.zeros([FLAGS.batch_size, FLAGS.class_num], dtype=np.float32)
    miss_vec = 0
    total_data = len(data_len)
    for i in range(FLAGS.batch_size):
        # seq_states:records the id of a sentence in a sequence
        # stop_states: records whether the sentence is judged by the program
        if stop_states[i] == 1 or seq_states[i] >= data_len[ids[i]]:
            ids[i] = counter_id + start_id
            seq_states[i] = 0
            try:
                t_words = data[ data_ID[ids[i]] ]['text'][seq_states[i]]
            except:
                print(ids[i], seq_states[i])
            for j in range(len(t_words)):
                m_word = t_words[j]
                try:
                    input_x[i][j] = word2vec[m_word]
                except:
                    miss_vec = 1
            input_y[i] = data_y[ids[i]]
            counter_id += 1
            counter_id = counter_id % total_data
        else:
            try:
                t_words = data[ data_ID[ids[i]] ]['text'][seq_states[i]] # t_words:事件的第i个text,
            except:
                print("ids and seq_states:", ids[i], seq_states[i])
                t_words = []
            for j in range(len(t_words)):
                m_word = t_words[j]
                try:
                    input_x[i][j] = word2vec[m_word]  # input_x:[batch_size, 100, 300],这里好像只存储一个句子的前100个词向量.
                except:
                    miss_vec += 1
            input_y[i] = data_y[ids[i]]
        # point to the next sequence
        seq_states[i] += 1  # 将seq_states加1，表明第i个post已经准备训练，

    return input_x, input_y, ids, seq_states, counter_id


# In[46]:
'''
valid_data_len
'''
# def accuracy_on_valid_data(rdm_model = None, sent_pooler = None, rdm_classifier=None, new_data_len=[], cuda=True):
def accuracy_on_valid_data(rdm_model = None, sent_pooler = None, rdm_classifier=None, valid_new_len=[], cuda=False):
    batch_size = 16 # 原本是20，数据较少，改成2.
    t_steps = int(len(data_ID)/batch_size)
    sum_acc = 0.0
    miss_vec = 0
    mts = 0
    hit_vec = 0
    if len(valid_new_len) > 0:
        t_data_len = valid_new_len
    else:
        t_data_len = valid_data_len
    
    for step in range(t_steps):
        data_x = []
        m_data_y = np.zeros([batch_size, 2], dtype=np.int32)
        m_data_len = np.zeros([batch_size], dtype=np.int32)
        for i in range(batch_size):

            # m_data_y[i] = valid_data_y[mts]
            m_data_y[i] = valid_data_y[mts]
            m_data_len[i] = t_data_len[mts]
            seq = []
            for j in range(t_data_len[mts]):
                sent = []
                t_words = data[valid_data_ID[mts]]['text'][j]
                if len(t_words) == 0:
                    print("ID:%s   j:%3d    empty sentence:"%(valid_data_ID[mts], j), t_words)
                    continue        

                for k in range(len(t_words)):
                    m_word = t_words[k]
                    try:
                        sent.append( torch.tensor([word2vec[m_word]], dtype=torch.float32))
                    except KeyError:
                        miss_vec += 1
                        sent.append( torch.tensor([word2vec['我'] + word2vec['这'] ], dtype=torch.float32) )
                    except IndexError:
                        raise
                    else:
                        hit_vec += 1
                sent_tensor = torch.cat(sent)                        
                seq.append(sent_tensor)

            data_x.append(seq)
            mts += 1
            '''
            取决于验证逻辑
            '''
            # if mts >= len(valid_data_ID): # read data looply
            #     mts = mts % len(valid_data_ID)
            if mts >= len(t_data_len):
                mts = mts % len(t_data_len)
        
        
        if rdm_model is not None and sent_pooler is not None and rdm_classifier is not None:
            with torch.no_grad():
                seq = sent_pooler(data_x)
                rdm_hiddens = rdm_model(seq)
                batchsize, _, _ = rdm_hiddens.shape
                # print("batch:",batchsize)
                # print("rdm_hiddens:", rdm_hiddens,len(rdm_hiddens))
                rdm_outs = torch.cat(
                    [rdm_hiddens[i][m_data_len[i]-1].unsqueeze(0) for i in range(batchsize)]
                    # a list of tensor, where the ndim of tensor is 1 and the shape of tensor is [hidden_size]
                )
                rdm_scores = rdm_classifier(
                    rdm_outs
                )
                rdm_preds = rdm_scores.argmax(axis=1)
                y_label = torch.tensor(m_data_y).argmax(axis=1) if cuda else torch.tensor(m_data_y).argmax(axis=1)
                acc = accuracy_score(y_label.cpu().numpy(), rdm_preds.cpu().numpy())

            # torch.empty_cache()

        sum_acc += acc
    mean_acc = sum_acc / (1.0*t_steps)
    print("验证平均准确率mean_acc:", mean_acc)
    return mean_acc

# In[47]:

def get_reward_0(isStop, ss, pys, ids, seq_ids):
    '''           (isStop, stopScore, ssq, ids, seq_states)
        isStop: 判断是否停止检测，1 停止，0 继续
        stopScore: 预测值
        ssq: rdm_classifier 结果，所有的对应每个事件中每个post的谣言预测值预测结果
        ids: 序列，第几个事件
        seq_states: 某个事件的第几个post .
    '''
    global reward_counter
    reward = torch.zeros([len(isStop)], dtype=torch.float32)  # 每一步的奖励
    Q_Val = torch.zeros([len(isStop)], dtype=torch.float32)   # Q值
    for i in range(len(isStop)):
        if isStop[i] == 1:
            # 给出停止的动作后，计算奖励，根据rdm分类结果与标签比较
            try:
                if pys[ids[i]][seq_ids[i]-1].argmax() == np.argmax(data_y[ids[i]]):
                    # 第ids[i]个事件中第[seq_ids[i]-1]个post预测结果与标签相同，即预测正确。
                    reward_counter += 1  # more number of correct prediction, more rewards
                    r = 1 + FLAGS.reward_rate * math.log(reward_counter)
                    reward[i] = r   
                else:
                    # 预测错误，惩罚，值较大
                    reward[i] = -100
            except:
                print("i:", i)
                print("ids_i:", ids[i])
                print("seq_ids:", seq_ids[i])
                print("pys:", pys[ids[i]])
                raise
            Q_Val[i] = reward[i]
        else:
            # 如果继续检测，惩罚，值较小
            reward[i] = -0.01
            Q_Val[i] = reward[i] + 0.99 * max(ss[i])
    return reward, Q_Val

# In[48]:

def get_reward(isStop, ss, pys, ids, seq_ids):
    '''没有用到'''
    global reward_counter
    reward = torch.zeros([len(isStop)], dtype=torch.float32)
    Q_Val = torch.zeros([len(isStop)], dtype= torch.float32)
    for i in range(len(isStop)):
        if isStop[i] == 1:
            if pys[ids[i]][seq_ids[i]-1].argmax() == np.argmax(data_y[ids[i]]):
                reward_counter += 1 # more number of correct prediction, more rewards
                r = 1 + min(FLAGS.reward_rate * math.log(reward_counter), 10)
                reward[i] = r   
            else:
                reward[i] = -100
            Q_Val[i] = reward[i]
        else:
            reward[i] = -0.01 
            Q_Val[i] = reward[i] + 0.99 * max(ss[i])
    return reward, Q_Val

# In[49]:

def get_reward_v1(isStop, mss, ssq, ids, seq_states, cm_model, rdm_hiddens_seq):
    '''没有用到'''
    global reward_counter
    reward = torch.zeros([len(isStop)], dtype=torch.float32)
    Q_Val = torch.zeros([len(isStop)], dtype= torch.float32)
    for i in range(len(isStop)):
        if isStop[i] == 1:
            if ssq[ids[i]][seq_states[i]-1].argmax() == np.argmax(data_y[ids[i]]):
                reward_counter += 1 # more number of correct prediction, more rewards
                r = 1 + min(FLAGS.reward_rate * math.log(reward_counter), 10)
                reward[i] = r   
                if data_len[ids[i]] > seq_states[i]:
                    with torch.no_grad():
                        subsequent_score = cm_model.Classifier(
                            nn.functional.relu(
                                cm_model.DenseLayer(
                                    rdm_hiddens_seq[ids[i]]
                                )
                            )
                        )               
                    # torch.empty_cache()
                    for j in range(seq_states[i], data_len[ids[i]]):
                        if subsequent_score[j][0] > subsequent_score[j][1]:
                            reward[i] += -20
                            break
                        else:
                            reward[i] +=  15.0/data_len[ids[i]]
            else:
                reward[i] = -100
            Q_Val[i] = reward[i]
        else:
            reward[i] = -0.01
            Q_Val[i] = reward[i] + 0.99 * max(mss[i])
    return reward, Q_Val

# In[51]:


def get_new_len(sent_pooler, rdm_model, cm_model, FLAGS, cuda=False):
    batch_size = FLAGS.batch_size
    new_len = []  # 训练集新长度
    if len(data_ID) % batch_size == 0: # the total number of events
        flags = int(len(data_ID) / FLAGS.batch_size)
    else:
        flags = int(len(data_ID) / FLAGS.batch_size) + 1
    for i in range(flags):
        with torch.no_grad():
            x, x_len, y = get_df_batch(i, batch_size)
            seq = sent_pooler(x)
            rdm_hiddens = rdm_model(seq)
            batchsize, _, _ = rdm_hiddens.shape
            rdm_outs = torch.cat(

                [ rdm_hiddens[i][x_len[i]-1] for i in range(batchsize)]

                # [rdm_hiddens[i][x_len[i]-1].unsqueeze(0) for i in range(batchsize)]
                # a list of tensor, where the ndim of tensor is 1 and the shape of tensor is [hidden_size]
            ).reshape(
                [-1, rdm_model.hidden_dim]
            )
            stopScores = cm_model.Classifier(
                    nn.functional.relu(
                        cm_model.DenseLayer(
                            rdm_hiddens.reshape([-1, rdm_model.hidden_dim])
                    )
                )
            ).reshape(
                [batchsize, -1, 2]
            )
            isStop = stopScores.argmax(axis=-1).cpu().numpy()   # cm 给出是否停止预测 [20, max_data_len]
            # iS_idx:行数【20】， iS: 每行的值
            # argmax()取出最大值所对应的索引。最大值为1，且小于对应的事件长度。
            # 得出最后面预测的停止的索引，作为新的推文长度
            tmp_len = [iS.argmax()+1 if (iS.max() ==1 and (iS.argmax()+1)<x_len[iS_idx]) else x_len[iS_idx] for iS_idx, iS in enumerate(isStop)]
            print("tmp_len",tmp_len)
                                                                                    #iS_idx:第几个 i，元素位置。. iS ：元素值

            for t_idx in range(len(tmp_len)):
                try:
                    assert tmp_len[t_idx] <= x_len[t_idx]
                except:
                    print("i:", t_idx)
                    print("new_len:", tmp_len)
                    print("data_len:", x_len)
                    raise

            new_len.extend(tmp_len)

    batchsize = 20
    mts = 0
    hit_vec = 0
    miss_vec = 0
    t_steps = int(len(valid_data_ID)/batchsize)
    valid_new_len = []      # 验证集新长度
    for step in range(t_steps):
        data_x = []
        m_data_y = np.zeros([batch_size, 2], dtype=np.int32)
        m_data_len = np.zeros([batch_size], dtype=np.int32)
        for i in range(batch_size):
            m_data_y[i] = valid_data_y[mts]
            m_data_len[i] = valid_data_len[mts]
            seq = []
            for j in range(valid_data_len[mts]):
                sent = []
                t_words = data[valid_data_ID[mts]]['text'][j]

                for k in range(len(t_words)):
                    m_word = t_words[k]
                    try:
                        sent.append( torch.tensor([word2vec[m_word]], dtype=torch.float32))
                    except KeyError:
                        miss_vec += 1
                        sent.append( torch.tensor(word2vec['啊'] +[word2vec['张三丰'] +  word2vec['武当']  ], dtype=torch.float32) )
                    except IndexError:
                        raise
                    else:
                        hit_vec += 1
                if len(sent) != 0 :
                    sent_tensor = torch.cat(sent)
                else:
                    print("empty sentence:", t_words)
                seq.append(sent_tensor)
            data_x.append(seq)
            mts += 1
            if mts >= len(data_ID): # read data looply
                mts = mts % len(data_ID)

        with torch.no_grad():
            seq = sent_pooler(data_x)
            rdm_hiddens = rdm_model(seq)
            batchsize, _, _ = rdm_hiddens.shape
            rdm_outs = torch.cat(
                [ rdm_hiddens[i][m_data_len[i]-1] for i in range(batchsize)] 
                # a list of tensor, where the ndim of tensor is 1 and the shape of tensor is [hidden_size]
            ).reshape(
                [-1, rdm_model.hidden_dim]
            )
            stopScores = cm_model.Classifier(
                    nn.functional.relu(
                        cm_model.DenseLayer(
                            rdm_hiddens.reshape([-1, rdm_model.hidden_dim])
                    )
                )
            ).reshape(
                [batchsize, -1, 2]
            )
            isStop = stopScores.argmax(axis=-1).cpu().numpy()

            tmp_len = [iS.argmax()+1 if (iS.max() ==1 and (iS.argmax()+1)<m_data_len[iS_idx]) else m_data_len[iS_idx] for iS_idx, iS in enumerate(isStop)]

            for t_idx in range(len(tmp_len)):
                try:
                    assert tmp_len[t_idx] <= m_data_len[t_idx]
                except:
                    print("i:", t_idx)
                    print("new_len:", tmp_len)
                    print("data_len:", x_len)
                    raise
        valid_new_len.extend(tmp_len)

    print("new_len", new_len)
    print("max_new_len:", max(new_len))
    print("valid_new_len", valid_new_len)
    print("mean_new_len:", sum(new_len)*1.0/len(new_len))
    return new_len[:len(data_len)], valid_new_len[:len(valid_data_len)]

# In[52]:

def get_RL_Train_batch_V1(D, FLAGS, batch_size, cuda=False):
    m_batch = random.sample(D, batch_size)  # list
    rdm_state = torch.zeros([batch_size, 256], dtype=torch.float32)
    s_rw = np.zeros([FLAGS.batch_size], dtype=np.float32)
    s_ids = []
    s_seqStates = []
    for i in range(batch_size):
        rdm_state[i] = m_batch[i][0]
        s_ids.append(m_batch[i][1])
        s_seqStates.append(m_batch[i][2])
        s_rw[i] = m_batch[i][3]
    if cuda:
        return rdm_state.cuda(), s_ids, s_seqStates, s_rw   # tensor ;list ; list
    else:
        return rdm_state, s_ids, s_seqStates, s_rw   # tensor ;list ; list

