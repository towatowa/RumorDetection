'''
    该文件是加载模型
'''
import torch
import RD.CX_RD.utils.utils as utils
import os
from torch import nn
import torch.nn.functional as F
from RD.CX_RD import remove_stopWords
import glob
import json

# 模型的路径
path = os.path.join('RD', 'CX_RD', 'Data', 'save_net', 'net_params.pkl')
# 读取训练集
train_data = utils.read_pheme('deleteColumn_train', data_root=os.path.join('RD', 'CX_RD', 'Data', 'splitDataset'))
# 建立词汇表
vocab = utils.get_vocab_pheme(train_data)


# ----------------定义textCNN模型--------------------
class GlobalMaxPool1d(nn.Module):
    def __init__(self):
        super(GlobalMaxPool1d, self).__init__()

    def forward(self, x):
        # x shape: (batch_size, channel, seq_len)
        return F.max_pool1d(x, kernel_size=x.shape[2])  # shape: (batch_size, channel, 1)


class TextCNN(nn.Module):
    def __init__(self, vocab, embed_size, kernel_sizes, num_channels):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(len(vocab), embed_size)
        # 不参与训练的嵌入层
        self.constant_embedding = nn.Embedding(len(vocab), embed_size)
        self.dropout = nn.Dropout(0.5)
        self.decoder = nn.Linear(sum(num_channels), 2)
        # 时序最大池化层没有权重，所以可以共用一个实例
        self.pool = GlobalMaxPool1d()
        self.convs = nn.ModuleList()  # 创建多个一维卷积层
        for c, k in zip(num_channels, kernel_sizes):
            self.convs.append(nn.Conv1d(in_channels=2 * embed_size,
                                        out_channels=c,
                                        kernel_size=k))

    def forward(self, inputs):
        # 将两个形状是(批量大小, 词数, 词向量维度)的嵌入层的输出按词向量连结
        embeddings = torch.cat((
            self.embedding(inputs),
            self.constant_embedding(inputs)), dim=2)  # (batch, seq_len, 2*embed_size)
        # 根据Conv1D要求的输入格式，将词向量维，即一维卷积层的通道维(即词向量那一维)，变换到前一维
        embeddings = embeddings.permute(0, 2, 1)
        # 对于每个一维卷积层，在时序最大池化后会得到一个形状为(批量大小, 通道大小, 1)的
        # Tensor。使用flatten函数去掉最后一维，然后在通道维上连结
        encoding = torch.cat([self.pool(F.relu(conv(embeddings))).squeeze(-1) for conv in self.convs], dim=1)
        # 应用丢弃法后使用全连接层得到输出
        outputs = self.decoder(self.dropout(encoding))
        return outputs


# 加载模型
embed_size, kernel_sizes, nums_channels = 100, [3, 4, 5], [100, 100, 100]
cx_model_net = TextCNN(vocab, embed_size, kernel_sizes, nums_channels)
cx_model_net.load_state_dict(torch.load(path))


# -----------------谣言检测--------------------
# 谣言检测函数
def detection(data_handle):
    # 如果长度不够，补长度
    while len(data_handle[0]) <= 5:
        data_handle[0].append('0')

    # 进行检测
    judge = utils.predict_rumor(cx_model_net, vocab, data_handle[0])
    # print(judge)
    return judge
