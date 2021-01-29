import collections
import os
import random
import time
import json
from tqdm import tqdm
import torch
import torchtext

# 读取数据
def read_pheme(folder='train', data_root=r"F:\pycharmProjects\peiXunProgram\Data\splitDataset"):
    data = []
    for label in ['rumours', 'non-rumours']:
        folder_name = os.path.join(data_root, folder, label)
        for file in tqdm(os.listdir(folder_name)):
            with open(os.path.join(folder_name, file), 'r', encoding='utf-8') as f:
                json_data = json.load(f)
                # print(json_data[0])
                data.append(json_data)
                # print(data[0])
    random.shuffle(data)
    return data

def get_tokenized_pheme(data):
    '''
    :param data: list of [string, label]
    '''
    return [review for review, _ in data]

# 建立数据的词汇表
def get_vocab_pheme(data):
    tokenized_data = get_tokenized_pheme(data)
    counter = collections.Counter([tk for st in tokenized_data for tk in st])
    return torchtext.vocab.Vocab(counter, min_freq=5)

# 求数据集内容的最大长度
def max_length(obj):
    if isinstance(obj, str):
        return 0
    elif all(isinstance(i, str) for i in obj):
        return len(obj)
    else:
        return max(max_length(i) for i in obj)

# 处理数据，
def preprocess_pheme(data, vocab):
    tokenized_data = get_tokenized_pheme(data)
    max_l = max_length(tokenized_data)     # 将每条内容设置成最大长度，不够的补0
    # max_l = 500  # 将每条评论通过截断或者补0，使得长度变成500

    def pad(x):
        return x + [0] * (max_l - len(x))
        # return x[:max_l] if len(x) > max_l else x + [0] * (max_l - len(x))

    features = torch.tensor([pad([vocab.stoi[word] for word in words]) for words in tokenized_data])
    labels = torch.tensor([score for _, score in data])
    return features, labels

# 从预训练好的vocab中提取出words对应的词向量
def load_pretrained_embedding(words, pretrained_vocab):
    '''从预训练好的vocab中提取出words对应的词向量'''
    embed = torch.zeros(len(words), pretrained_vocab.vectors[0].shape[0])   # 初始化为0
    oov_count = 0   # out of vocabulary
    for i, word in enumerate(words):
        try:
            idx = pretrained_vocab.stoi[word]
            embed[i, :] = pretrained_vocab.vectors[idx]
        except KeyError:
            oov_count += 1
    if oov_count > 0:
        print('There are %d oov words.' % oov_count)
    return embed

# 模型训练
def train(train_iter, test_iter, net, loss, optimizer, device, num_epochs):
    net = net.to(device)
    print("training on ", device)
    batch_count = 0
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))

# 模型评估
def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        # 如果没指定device就使用net的device
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                net.eval()  # 评估模式, 这会关闭dropout
                acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                net.train()     # 改回训练模式
            else:   # 自定义的模型, 3.13节之后不会用到, 不考虑GPU
                if('is_training' in net.__code__.co_varnames): # 如果有is_training这个参数
                    # 将is_training设置成False
                    acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item()
                else:
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
    return acc_sum / n

# 数据检测
def predict_rumor(net, vocab, sentence):
    """sentence是词语的列表"""
    device = list(net.parameters())[0].device
    sentence = torch.tensor([vocab.stoi[word] for word in sentence], device=device)
    label = torch.argmax(net(sentence.view((1, -1))), dim=1)
    return '1' if label.item() == 1 else '0'