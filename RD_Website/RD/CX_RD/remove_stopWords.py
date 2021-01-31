'''
    该文件是去除停用词
'''
import os
import json
import re
import pandas as pd

stop_words = pd.read_csv(os.path.join('RD', 'CX_RD', 'Data', 'english.txt'), index_col=False, sep="\t", quoting=3, names=['stopword'], encoding='utf-8')

# 过滤emoji更全的方法
def filterEmoji(desstr, restr=' '):
    # 过滤emoji
    try:
        co = re.compile(u'[\U00010000-\U0010ffff]')
    except re.error:
        co = re.compile(u'[\uD800-\uDBFF][\uDC00-\uDFFF]')
    return co.sub(restr, desstr)


def filterBoxDrawing(desstr, restr=' '):
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


def makedir(new_dir):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)


# 去停用词
def remove_stop_words(datas):
    # 进一步去除停用词
    punc = '~`!#$%^&*()_+-=|\';":/.,?><~·！@#￥%……&*（）——+-=“：’；、。，？》《{}'
    # 获取数据
    data_all = datas

    data_removeStopWords = []

    for i in range(len(data_all)):
        dicts_1_1 = []
        dicts_1_2 = []

        dicts_2_1 = []
        dicts_2_2 = []

        dicts_3_1 = []
        dicts_4_1 = []

        # 第一步去停用词
        for words in data_all[i][0]:
            str = re.sub('@\w+|http.*\w+|\n|[%s]+' % punc, "", words)
            if str != '':
                dicts_1_1.append(str)
        dicts_1_2.append(dicts_1_1)
        dicts_1_2.append(data_all[i][1])
        # print(i, dicts_1_2)

        # 第二步去停用词
        for words in dicts_1_2[0]:
            str = bodyNorm(words)  # 去表情
            if str != '':
                dicts_2_1.append(str)
        dicts_2_2.append(dicts_2_1)
        dicts_2_2.append(dicts_1_2[1])
        # print(i, dicts_2_2)

        # 第三步去停用词
        if dicts_2_2[0]:
            dicts_3_1.append([st.lower() for st in dicts_2_2[0] if
                              st != '' and st.lower() not in stop_words.stopword.values.tolist()])
            dicts_3_1.append(dicts_2_2[1])
        # print(i, dicts_3_1)

        # 第四步去停用词
        if dicts_3_1[0]:
            dicts_4_1.append(dicts_3_1[0])
            dicts_4_1.append(dicts_3_1[1])
        # print(i, dicts_4_1)

        data_removeStopWords.append(dicts_4_1)

    # print(len(data_removeStopWords))
    # print(data_removeStopWords)

    return data_removeStopWords