# 调用模型测试
import os
import glob
import json
from RD.CX_RD import cx_model_net

# 读取数据，没去除停用词
data_path = r'F:\pycharmProjects\peiXunProgram\Data\content_id_data\content_id_splitData\test'
datas = []
for root, dirs, files in os.walk(data_path):
    for sDir in dirs:
        json_list = glob.glob(os.path.join(root, sDir, '*.json'))
        for json_path in json_list:
            with open(json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
                datas.append(json_data)
            f.close()


# 调用谣言检测函数
cx_model_net.detection(datas)