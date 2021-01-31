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
from RD.rumor_model.my_edr.dataUtils_CN import *
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from RD.rumor_model.my_edr.config import *
import tqdm
# from ERD_pytorch.torch.ERD import FLAGS


def TrainRDMModel_V0(rdm_model, sent_pooler, rdm_classifier,
                    t_steps=100, stage=0, new_data_len=[], valid_new_len=[], best_valid_acc = 0.0, logger=None,
                        log_dir="RDMBertTrain", cuda=False):

    batch_size = FLAGS.batch_size
    sum_loss = 0.0
    sum_acc = 0.0
    t_acc = 0.9   # 没有用
    ret_acc = 0.0  # 没有用，但却返回。
    init_states = torch.zeros([1, batch_size, rdm_model.hidden_dim], dtype=torch.float32)
    weight = torch.tensor([2.0, 1.0], dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss(weight=weight)
    optim = torch.optim.Adagrad([
                                {'params': sent_pooler.parameters(), 'lr': 5e-3},
                                {'params': rdm_model.parameters(), 'lr': 5e-3},
                                {'params': rdm_classifier.parameters(), 'lr': 5e-3}
                             ]
    )
    
    writer = SummaryWriter(log_dir, filename_suffix="_ERD_CM_stage_%3d"%stage)
    #创建一个“SummaryWriter”，它将写出事件和摘要到事件文件。
    # for step in range(499, t_steps):
    for step in tqdm.trange(0, t_steps):
        optim.zero_grad()
        #将梯度置零
        try:
            #x, x_len, y = get_df_batch(step*batch_size, batch_size)
            x, x_len, y = get_df_batch(step * batch_size, batch_size,new_data_len)
            #
            # [batch_size, max_twitter_len, max_word_len]   文本
            # [batch_size, max_twitter_len, max_word_len， 300] :data_x=x
            # [batch_size, max_twitter_len，300] # seq,池化之后

            seq = sent_pooler(x)        # 将一个batch_size的全部数据进行训练
            #最大池化，返回池化后向量seq
            # [batch_size, max_twitter_len，300]
            # [seq_len, batch_size, input_dim]
            # input_dim是输入的维度，比如是
            # batch_size是一次往RNN输入句子的数目，比如是5。
            # seq_len是一个句子的最大长度，比如15
            # 总共20个时间步，也就是20个循环单元。RNN的输入维度是input_dim，总共输入seq_len个时间步，
            # 则每个时间步输入到这个整个RNN模块的维度是[batch_size,input_dim]

            rdm_hiddens = rdm_model(seq)
            # [batch_size, max_twitter_len，256]


#             rdm_hiddens, rdm_out, rdm_cell = rdm_model(seq, x_len.tolist())
            batchsize, _, _ = rdm_hiddens.shape # 求batch,或许最后一个batch_size,不是满的
            rdm_outs = torch.cat(
                [ rdm_hiddens[i][x_len[i]-1].unsqueeze(0) for i in range(batchsize)] 
                # a list of tensor, where the ndim of tensor is 1 and the shape of tensor is [hidden_size]
            )    # 对rdm_hiddens,选取向量, rdm_hiddens=[batch_size, max_twitter_len，256] -> rdm_outs[batchsize, 256]
            rdm_scores = rdm_classifier(
                rdm_outs
            )  # rdm_scores:[batchsize, 2],是否是谣言的概率
            rdm_preds = rdm_scores.argmax(axis=1)  # argmax(axis=0/1) argmax返回的是最大数的索引.[batchsize], 深度学习输出结果
            y_label = torch.tensor(y).argmax(axis=1).cuda() if cuda else torch.tensor(y).argmax(axis=1)  # 也是返回标签中最大的索引,即,标签为1的索引列好[batchsize],标签
            acc = accuracy_score(y_label.cpu().numpy(), rdm_preds.cpu().numpy())    # 经过rdm_model和rdm_classifier, 分类的准确率
            # 计算准确率的函数accuracy_score(y_true, ypred)norma lize：默认值为True，返回正确分类的比例；如果为False，返回正确分类的样本数
            loss = loss_fn(rdm_scores, y_label)  # 交叉熵损失函数
            loss.backward()     # 反向传播
            # torch.empty_cache()    # 释放显存

        except RuntimeError as exception:
            if "out of memory" in str(exception):
                print("WARNING: out of memory")
                # print("%d, %d | x_len:"%(step, j), x_len)
                print("%d| x_len:"%(step), x_len)
                # if hasattr(torch, 'empty_cache'):
                    # torch.empty_cache()
#                     time.sleep(5)
                raise exception
            else:   
                raise exception

        optim.step()        # 优化
        writer.add_scalar('Train Loss', loss, step)         # 将训练损失,步数写进日志
        writer.add_scalar('Train Accuracy', acc, step)      # 将准确率,步数写进日志

        sum_loss += loss
        sum_acc += acc
        
        # torch.empty_cache()
        
        if step % 1 == 0:
            '''
            每十次step,输出一次训练的平均损失和平均准确率,
            '''
            sum_loss = sum_loss / 10
            sum_acc = sum_acc / 10
            print('%3d | %d , train_loss/accuracy = %6.8f/%6.7f'             % (step, t_steps, 
                sum_loss, sum_acc,
                ))
            # if step%500 == 499:
            if step%1 ==0:
                '''
                如果步数五百step进行一次验证,  输出一次校验的准确率.应该没到这步.
                '''
                # valid_acc = accuracy_on_valid_data(rdm_model, sent_pooler, rdm_classifier)
                valid_acc = accuracy_on_valid_data(rdm_model, sent_pooler, rdm_classifier, valid_new_len)
                if valid_acc > best_valid_acc:
                    print("valid_acc:", valid_acc)
                    writer.add_scalar('Valid Accuracy', valid_acc, step)
                    best_valid_acc = valid_acc
                    if stage != 0:
                        '''保存模型'''
                        rdm_save_as = '%s/ERD_best_%d.pkl'% (log_dir, stage)
                    else:
                        rdm_save_as = '%s/ERD_best.pkl'% (log_dir)
                    torch.save(
                        {
                            "rmdModel":rdm_model.state_dict(),
                            "sent_pooler":sent_pooler.state_dict(),
                            "rdm_classifier": rdm_classifier.state_dict()
                        },
                        rdm_save_as
                    )
                    print("ERD模型最好的模型已保存")
            sum_acc = 0.0
            sum_loss = 0.0
    print(get_curtime() + " Train df Model End.")
    return ret_acc
