import sys
import random
import torch
import importlib
from tensorboardX import SummaryWriter
import torch.nn.utils.rnn as rnn_utils
import pickle
import tqdm
import os
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import deque
import json
import pdb
from RD.rumor_model.my_edr.dataUtils_CN import *
import numpy as np

# from ERD_pytorch.torch.TWITTER.process_twitter_data import get_curtime
# from ERD_pytorch.torch.dataUtils import get_new_len, get_data_len, get_rl_batch, get_reward_0, get_RL_Train_batch_V1
# from ERD_pytorch.torch.dataUtils_CN import get_data_ID
# from ERD_pytorch.torch.test import get_df_batch


def TrainCMModel_V0(sent_pooler, rdm_model, rdm_classifier, cm_model, stage, t_rw, t_steps, log_dir, logger, FLAGS, cuda=False):
    #sent_pooler, rdm_model, rdm_classifier, cm_model,                  0,    0.5,  目标步数：2000,   log_dir, None, FLAGS, cuda=True
    batch_size = FLAGS.batch_size
    t_acc = 0.9
    ids = np.array(range(batch_size), dtype=np.int32)  # 序列0-batch
    seq_states = np.zeros([batch_size], dtype=np.int32)  # 序列状态
    isStop = torch.zeros([batch_size], dtype=torch.int32)  # 是否停止
    max_id = batch_size
    # df_init_states = torch.zeros([1, batch_size, rdm_model.hidden_dim], dtype=torch.float32).cuda()     #[1, batch, 256]
    df_init_states = torch.zeros([1, batch_size, rdm_model.hidden_dim], dtype=torch.float32)
    writer = SummaryWriter(log_dir, filename_suffix="_ERD_CM_stage_%3d"%stage)   # stage = 0 #保存
    D = deque()     # 经验池
    #deque()类似列表
    ssq = []    # 存放所有事件rdm_classifier()后的结果.
    print("in RL the begining")
    rl_optim = torch.optim.Adam([{'params': sent_pooler.parameters(), 'lr': 2e-5},
                                 {'params': rdm_model.parameters(), 'lr': 2e-5},
                                 {'params':cm_model.parameters(), 'lr':1e-3}])  # 优化函数
    data_ID = get_data_ID()
    valid_data_len = get_new_len(sent_pooler, rdm_model, cm_model, FLAGS, cuda=False)    # 优化之前的有效数据长度，将返回的两个值存到一个双列表中
    data_len = get_data_len()
    
    if len(data_ID) % batch_size == 0:  # the total number of events
        flags = int(len(data_ID) / FLAGS.batch_size)
    else:
        flags = int(len(data_ID) / FLAGS.batch_size) + 1
    # flags 一个数据集除以batch得出多少轮训练,对数据集进行切片训练
    for i in range(flags):
        # 这一步将训练集所有进行特征提取，得到rdm_hiddens ,rdm_classifier结果存到ssq[]中。即得到每个事件的每个post，经过RDM检测结果。以便用于比较CM后比较检测结果
        with torch.no_grad():
            x, x_len, y = get_df_batch(i, batch_size)
            seq = sent_pooler(x)        #
            rdm_hiddens = rdm_model(seq)
            batchsize, _, _ = rdm_hiddens.shape
            print("batch %d" %i)
            if len(ssq) > 0:
                ssq.extend([rdm_classifier(h) for h in rdm_hiddens])
            else:
                ssq = [rdm_classifier(h) for h in rdm_hiddens]  # 将rdm_classifier结果,追加到ssq[]列表  rdm_scores:[batchsize, 2],是否是谣言的概率
            # torch.cuda.empty_cache()
            # torch.empty_cache()

    print(get_curtime() + " Now Start RL training ...")
    counter = 0
    sum_rw = 0.0  # sum of rewards
    
    while True:
        if counter > FLAGS.OBSERVE:
            '''
            OBSERVE=1000
            真正的CM训练，通过get_RL_Train_batch_V1（）从经验池随机选20个样本进行训练。
            
            '''
            sum_rw += rw.mean()
            if counter % 200 == 0:
                # 每200次输出一次 奖励
                sum_rw = sum_rw / 2000
                print(get_curtime() + " Step: " + str(counter-FLAGS.OBSERVE) + " REWARD IS " + str(sum_rw))
                if counter > t_steps:
                    # 如果大于目标步数，退出
                    print("Retch The Target Steps")
                    break
                sum_rw = 0.0
            s_state, s_x, s_isStop, s_rw = get_RL_Train_batch_V1(D, FLAGS, batch_size)  # 从经验池中取数据，s_state，状态，作为h0，s_x:某个事件的某个post词向量，s_isStop:是否停止，s_rw:奖励
            word_tensors = torch.tensor(s_x)  #，转成tensor
            batchsize, max_sent_len, emb_dim = word_tensors.shape
            sent_tensor = sent_pooler.linear(word_tensors.reshape([-1, emb_dim])).reshape([batchsize, max_sent_len, emb_dim]).max(axis=1)[0].unsqueeze(1)    #池化层
            df_outs, df_last_state = rdm_model.gru_model(sent_tensor, s_state.unsqueeze(0))  # 将s_state 和 池化后结果x,输入GRU,得到hi,和向下传播的
            batchsize, _, hidden_dim = df_outs.shape
            stopScore, isStop = cm_model(df_outs.reshape([-1, hidden_dim]))   # CM对hi，处理。生成估计值。
            out_action = (stopScore).sum(axis=1)    # 估计动作
            rl_cost = torch.pow(torch.Tensor(s_rw) - out_action, 2).mean()   # s_rw：现实，表签， out_action :估计。  求平方差
            rl_optim.zero_grad()    # 梯度清零
            rl_cost.backward()      # 反向传播
            # torch.empty_cache()    # 释放GPU
            rl_optim.step()             # 优化
            # print("RL Cost:", rl_cost)    # 输出损失值
            writer.add_scalar('RL Cost', rl_cost, counter - FLAGS.OBSERVE)
            if (counter - FLAGS.OBSERVE)%100 == 0:
                # 每100，输入损失值
                print("*** %6d|%6d *** RL Cost:%8.6f"%(counter, t_steps, rl_cost))   # rl_cost:rl损失函数
                valid_new_len = get_new_len(sent_pooler, rdm_model, cm_model, FLAGS, cuda=False)     # 每100次训练，查看优化后的新长度
                # 返回两只列表，即将新的训练集长度，和验证集长度都返回，存到一个双列表中。
                print("valid_data_len:", np.array(valid_data_len))
                print("valid_new_len:", np.array(valid_new_len))
                for i in range(len(valid_new_len)):
                    print("diff {} len,{}:".format(i, np.array(valid_data_len[i]) - np.array(valid_new_len[i])))
        #######################
        # 下面是先进性1000次训练，存放到经验池中
        x, y, ids, seq_states, max_id = get_rl_batch(ids, seq_states, isStop, max_id, 0, 0)
        for j in range(FLAGS.batch_size):
            if seq_states[j] == 1:
                # 如果是新的时间，就给它赋值个新的H0
                df_init_states[0][j].fill_(0.0)
                
        with torch.no_grad():
            word_tensors = torch.tensor(x)  # 将x转成tensor.[batch_size, 100, 300]
            batchsize, max_sent_len, emb_dim = word_tensors.shape      # 获取维度

            sent_tensor = sent_pooler.linear(word_tensors.reshape([-1, emb_dim])).reshape([batchsize, max_sent_len, emb_dim]).max(axis=1)[0].unsqueeze(1)  # 池化层
            df_outs, df_last_state = rdm_model.gru_model(sent_tensor, df_init_states)  # 输入不同，有的输入的是input_x,所有数据维度，而这个输入的是某一层（深度）
            batchsize, _, hidden_dim = df_outs.shape
            stopScore, isStop = cm_model(df_outs.reshape([-1, hidden_dim]))
            
        for j in range(batch_size):
            # 一个随机率
            if random.random() < FLAGS.random_rate:
                isStop[j] = torch.randn(2).argmax()
            if seq_states[j] == data_len[ids[j]]:
                # 如果seq_states记录的推文到达最后一个post,必须截至
                isStop[j] = 1
        rw, Q_val = get_reward_0(isStop, stopScore, ssq, ids, seq_states)   # 计算奖励值和q值
        for j in range(FLAGS.batch_size):
            D.append((df_init_states[0][j], x[j], isStop[j], rw[j]))  # 经验池 ,状态，x[j];某个事件的某个推文词向量，是否终止，奖励值
            if len(D) > FLAGS.max_memory:
                D.popleft()
        df_init_states = df_last_state
        counter += 1
