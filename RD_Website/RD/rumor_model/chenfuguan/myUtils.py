import json
import os
import torch
import torchtext.vocab as Vocab
import collections
import pandas as pd
from string import punctuation as enpunctuation
import re
import nltk
from torch import nn


# 获取谣言原文和非谣言原文和谣言评论和非谣言评论文件的全部路径
def get_data_dir(target_path):
    rumor_class_dirs = os.listdir(target_path)

    rumor_file_dir_list = []  # 谣言数据根目录
    non_rumor_file_dir_list = []  # 非谣言数据根目录

    # 解析谣言和非谣言的数据目录
    for filename in rumor_class_dirs:
        rumor_file_dir_list.append(os.path.join(target_path, filename, "rumours"))
        non_rumor_file_dir_list.append(os.path.join(target_path, filename, "non-rumours"))

    all_non_rumor_content_file_dir_list = []  # 所有谣言评论数据目录（根的下下一级目录）
    all_rumor_content_file_dir_list = []
    all_rumor_srctweet_file_dir_list = []  # 所有谣言原文数据目录（根的下下一级目录）
    all_non_rumor_srctweet_file_dir_list = []

    for files in rumor_file_dir_list:
        files_dir = os.listdir(files)
        for file_dir_files in files_dir:
            all_rumor_content_file_dir_list.append(os.path.join(files, file_dir_files, "reactions"))
            all_rumor_srctweet_file_dir_list.append(os.path.join(files, file_dir_files, "source-tweet"))

    for files in non_rumor_file_dir_list:
        file_dir = os.listdir(files)
        for file_dir_files in file_dir:
            all_non_rumor_content_file_dir_list.append(os.path.join(files, file_dir_files, "reactions"))
            all_non_rumor_srctweet_file_dir_list.append(os.path.join(files, file_dir_files, "source-tweet"))

    return all_rumor_srctweet_file_dir_list, all_rumor_content_file_dir_list, all_non_rumor_srctweet_file_dir_list, all_non_rumor_content_file_dir_list


# 读取数据，通过文件的路径读取，label为谣言和非谣言的标签
def read_data(data_dir_list, label):
    data = []
    # data_src = []
    for dir in data_dir_list:
        for filename in os.listdir(dir):
            with open(os.path.join(dir, filename), 'r', encoding='utf8') as f:
                content = f.read()
                data.append([content, label])
            # data_dict = json.loads(content)
            # data.append([data_dict["text"]])
    return data  # data[[content, label]]


def read_content_data(data_path):
    a_twt_content = []
    # data_src = []
    for dir in data_path:
        data = []
        for filename in os.listdir(dir):
            with open(os.path.join(dir, filename), 'r', encoding='utf8') as f:
                content = f.read()
                data.append(content)
        a_twt_content.append(data)
        # data_dict = json.loads(content)
        # data.append([data_dict["text"]])
    return a_twt_content  # data[[content, label]]


# emoji、boxDrawing、Face：https://apps.timwhitlock.info/emoji/tables/unicode#block-6c-other-additional-symbols


# 过滤emoji更全的方法
# pip install emoji

def filterEmoji(desstr, restr=' '):
    # 过滤emoji
    try:
        co = re.compile(u'[\U00010000-\U0010ffff]')
    except re.error:
        co = re.compile(u'[\uD800-\uDBFF][\uDC00-\uDFFF]')
    return co.sub(restr, desstr)


def filterBoxDrawing(desstr, restr=''):
    # 过滤形如：╠、╤等boxdrawing字符
    co = re.compile(u'[\u2500-\u257f]')
    return co.sub(restr, desstr)


def filterFace(desstr, restr=' '):
    # 过滤：形如[衰]、[生气]、[开心]、[捂脸]等表情，用词典更好些
    p = re.compile('\[.{1,4}\]')
    t = p.findall(desstr)
    for i in t:
        desstr = desstr.replace(i, restr)
    return desstr


def filterSpecialSym(desstr, restr=' '):
    # print u'1\u20e3\ufe0f' #10个特殊的类似emoij的表情
    co = re.compile(u'[0-9]?\u20e3\ufe0f?')
    return co.sub(restr, desstr)


def bodyNorm(body):
    # body = re.compile(u'''\\\\\\\\\\\\\\\\n''').sub(' ', body) # 得用16个斜杠才行震惊
    body = re.compile(u'''\\\\+?n''').sub(' ', body)
    body = filterSpecialSym(body)
    body = filterEmoji(body)
    body = filterBoxDrawing(body)
    body = filterFace(body)
    return body


import numpy as np


def loadData(filename):
    data = []
    fr = open(filename, 'r', encoding='utf8')
    for line in fr.readlines():  # 逐行读取
        lineArr = line.strip().split('\t')  # 滤除行首行尾空格，以\t作为分隔符，对这行进行分解
        num = np.shape(lineArr)[0]
        data.append(["".join(lineArr[0:num - 1]), int(lineArr[num - 1])])  # 这一行的除最后一个被添加为数据
        # labelMat.append(int(lineArr[num-1]))#这一行的最后一个数据被添加为标签
    return data


def get_tokenized_imdb(data):
    """
    data: list of [string, label]
    """

    def tokenizer(text):
        return [tok.lower() for tok in text.split(' ')]

    return [tokenizer(review[0]) for review in data]


def preprocess_imdb(data, vocab):
    max_l = 300  # 将每条评论通过截断或者补0，使得长度变成300

    def pad(x):
        return x[:max_l] if len(x) > max_l else x + [0] * (max_l - len(x))

    tokenized_data = get_tokenized_imdb(data)
    features = torch.tensor([pad([vocab.stoi[word] for word in words]) for words in tokenized_data])
    labels = torch.tensor([score for _, score in data])
    return features, labels


def load_pretrained_embedding(words, pretrained_vocab, token_to_idx):
    """从预训练好的vocab中提取出words对应的词向量"""
    W = pretrained_vocab.weight.data
    embed = torch.zeros(len(words), W.shape[1])  # 初始化为0
    oov_count = 0  # out of vocabulary
    for i, word in enumerate(words):
        try:
            idx = token_to_idx[word]
            embed[i, :] = W[idx]
        except KeyError:
            oov_count += 1
    if oov_count > 0:
        print("There are %d oov words." % oov_count)
    return embed


def del_mark(word):
    punc = '~`!#$%^&*()_+-=|\';":/.,?><~·！@#￥%……&*（）——+-=“：’；、。，？》《{\]}\[✏'
    return re.sub(r"[%s]+" % punc, "", word)


def get_stopword():
    """获取停用词"""
    stopwords = pd.read_csv("RD/rumor_model/chenfuguan/data/stop_words-master/english.txt", index_col=False, sep="\t", quoting=3,
                            names=['stopword'], encoding='utf-8')
    stopwords = stopwords.stopword.values.tolist()
    return stopwords


# 停用
def calculate_result(test_data, net, vocab):
    """计算检测结果"""
    test_result = []
    for item in test_data:
        test_result.append([item, detect(net, vocab, item)])
    count = 0
    for i in range(0, len(test_result)):
        if int(test_result[i][1]) ^ test_data[i][1] == 0:
            count += 1
    return test_result, 1.0 * count / len(test_result) * 100


# 停用
def init_src_twt(src_twt_data):
    """处理从数据库读取的原贴"""
    src_data = []
    punc = '\]~`!#$%^&*()_+-=|\';"":/.,?><~·！@#￥%……&*（）——+-=“：’；、。，？》《{}1234567890→✔️\[✏'
    for item in src_twt_data:  # 将文本封装成字典列表，形式为{id 谣言文本 标签}
        data_dict = json.loads(item.src_twt)
        text = data_dict['text']
        text = re.sub('http.*', " URL", text)
        text = re.sub('@\w+|\n|[%s]+' % punc, "", text)
        text = bodyNorm(text)  # 去表情
        src_data.append([item.id, text])
    return src_data  # [id, text]


# -----------------------------------------------------------------------------------------------------------------
# 使用的网络函数，需要把这个类放在manange.py， 或者通过导包的形式导入使用的地方，没试过不知道可不可以
# -----------------------------------------------------------------------------------------------------------------
class BiRNN(nn.Module):
    def __init__(self, vocab, embed_size, num_hiddens, num_layers):
        super(BiRNN, self).__init__()
        self.embedding = nn.Embedding(len(vocab), embed_size)
        # bidirectional设为True即得到双向循环神经网络
        self.encoder = nn.LSTM(input_size=embed_size,
                               hidden_size=num_hiddens,
                               num_layers=num_layers,
                               bidirectional=True)
        # 初始时间步和最终时间步的隐藏状态作为全连接层输入
        self.decoder = nn.Linear(4 * num_hiddens, 2)

    def forward(self, inputs):
        # inputs的形状是(批量大小, 词数)，因为LSTM需要将序列长度(seq_len)作为第一维，所以将输入转置后
        # 再提取词特征，输出形状为(词数, 批量大小, 词向量维度)
        embeddings = self.embedding(inputs.permute(1, 0))
        # rnn.LSTM只传入输入embeddings，因此只返回最后一层的隐藏层在各时间步的隐藏状态。
        # outputs形状是(词数, 批量大小, 2 * 隐藏单元个数)
        outputs, _ = self.encoder(embeddings)  # output, (h, c)
        # 连结初始时间步和最终时间步的隐藏状态作为全连接层输入。它的形状为
        # (批量大小, 4 * 隐藏单元个数)。
        encoding = torch.cat((outputs[0], outputs[-1]), -1)
        outs = self.decoder(encoding)
        return outs


# -----------------------------------------------------------------------------------------------------------------
# 去停用词和去标点等处理
def drop_stopwords(srctweet, stopwords):
    """
    功能：去停用词和预处理
    srctweet: 原贴
    return:list[[word]]
    """
    punc = '\]~`!#$%^&*()_+-=|\';"":/.,?><~·！@#￥%……&*（）——+-=“：’；、。，？》《{}1234567890→✔️\[✏'
    srctweet = re.sub('http.*', " URL", srctweet)
    srctweet = re.sub('@\w+|\n|[%s]+' % punc, "", srctweet)
    srctweet = bodyNorm(srctweet)  # 去表情
    srctweet = srctweet.split()
    srctweet_clean = []
    for word in srctweet:
        if word in stopwords:
            continue
        word = del_mark(word)
        srctweet_clean.append(word)
    return srctweet_clean  # [words]


# 谣言检测
def detect(net, vocab, sentence, stopwords):
    """sentence是原贴"""
    sentence = drop_stopwords(sentence, stopwords)
    device = list(net.parameters())[0].device
    sentence = torch.tensor([vocab.stoi[word] for word in sentence], device=device)
    label = torch.argmax(net(sentence.view((1, -1))), dim=1)
    # return 'positive' if label.item() == 1 else 'negative'
    return int(1) if label.item() == 1 else int(0)


# 获取模型网络
def get_net():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = torch.load("RD/rumor_model/chenfuguan/data/LSTM_net_loss=0.0036.pt")  # 需要修改路径
    net = net.to(device)
    print("detection on ", device)
    net.embedding.weight.requires_grad = False  # 直接加载预训练好的, 所以不需要更新它
    return net


# 获取词典
def get_vocab_imdb():
    data = loadData("RD/rumor_model/chenfuguan/data/train_data.txt")  # 需要修改一下路径
    tokenized_data = get_tokenized_imdb(data)
    counter = collections.Counter([tk for st in tokenized_data for tk in st])
    return Vocab.Vocab(counter, min_freq=5)  # 去除出现次数小于5的


"""
# ------------------------------------------------------------------------------------------------------------------
# 使用例子
# 上面的get_net、get_vocab_imb函数涉及到读取本地文件，所以需要修改读取路径
# 获取词典
vocab = get_vocab_imdb()
# 获取停用词
stopwords = get_stopword()
# 获取网络
net = get_net()
# 使用
detecte(net, vocab, '检测语句', stopwords)
# -----------------------------------------------------------------------------------------------------------------
"""
