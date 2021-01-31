import codecs
import os
from datetime import date, datetime

from django.core.paginator import Paginator
from django.http import JsonResponse, HttpResponse
from django.shortcuts import render, redirect

from RD.rumor_model.chenfuguan.chuanbo_guocheng import draw_pic
from login.models import User
from RD import models
from RD.Utils.dt_01 import myUtils
from RD.rumor_model.liumeiqi.rumor_detection import MyGRU
from RD.rumor_model.liumeiqi import rumor_detection
from RD.rumor_model.chenfuguan import myUtils
from RD.rumor_model.chenfuguan.myUtils import BiRNN
from django.db.models import Q
import os
import json
import torch
from RD.CX_RD import cx_model_net
from RD.CX_RD import remove_stopWords
from RD.Utils import chuanbo_guocheng as cb_gc
from datetime import datetime
import time
from abc import ABC
import jieba.analyse
from torch import nn
import torch.nn.functional as func
from gensim.models import Word2Vec
from torch.nn.utils.rnn import pad_sequence

from RD.rumor_model.my_edr import mytest


# path1 = path_bace = os.path.abspath('.')+'\\val\\'


def rumor_detect_02(request):
    """谣言检测方法2"""
    datalist = []
    # 获取输入框的值
    user_id = request.session['user_id']
    file = request.FILES.get('filename')  # 获取谣言文件对象
    # crr_selected = request.POST.get('selected')  # 选择框
    detection = request.POST.get('start_detection')
    dirs = os.listdir('RD/static/data/')  # 所有的文件名
    keyword = request.POST.get('keyword')
    # 如果选择了，处理提交的数据
    if request.method == 'POST':
        if not file is None:
            # 根据路径读取数据
            try:
                with codecs.open(os.path.join('RD/static/data/', file.name), 'wb') as f:  # 拼接上传的文件名
                    for i in request.FILES['filename'].chunks():  # 获取文件内容
                        f.write(i)
                # 将数据上传数据库
                # 读取文件
                with codecs.open(os.path.join('RD/static/data/', file.name), 'r', encoding='utf8') as f:
                    all_data = f.read()
                    all_data = json.loads(all_data)
                # -----------------------------------------------------------------------------------
                # 循环读取每一条评论上传到数据库
                for item in all_data:
                    one_src = models.Twitter()  # 一条原贴

                    # 将item转换为列表

                    # 初始化原贴
                    one_src.src_twt = item[0]  # 原题
                    one_src.true_label = int(item[1])  # 真实标签
                    one_src.user_id = user_id  # 用户id
                    one_src.detect_type = 2  # 检测类型
                    # 向数据库插入数据
                    one_src.save()
                    twitter_id = models.Twitter.objects.all().order_by('-id').first().id  # 新插入值的id
                    # 读取评论列表
                    for i in item[2:]:
                        one_comment = models.Content()  # 一条评论
                        one_comment.text = i
                        one_comment.twitter_id = twitter_id
                        one_comment.save()  # 保存到数据库
                # ------------------------------------------------------------------------------------------
                try:
                    os.remove(os.path.join('RD/static/data/', file.name))
                except:
                    message = "数据更新失败,请退出重新登录，继续检测可能会出现未知错误"
                return render(request, 'rumor_detect_02.html', locals())
                message = "上传成功！"
                return render(request, 'rumor_detect_02.html', locals())
            except Exception as e:
                print('str(Exception):\t', str(Exception))
                print('str(e):\t\t', str(e))
                print('repr(e):\t', repr(e))
                print('########################################################')  # 捕获异常
                message = "读取文件失败！"
                return render(request, 'rumor_detect_02.html', locals())
        else:
            if not detection is None:  # 检测模块
                try:
                    # 从数据库获取数据
                    all_data = models.Twitter.objects.filter(new_add=1)  # 获取新添加的数据
                    # 获取词典
                    vocab = myUtils.get_vocab_imdb()
                    # 获取停用词
                    stopwords = myUtils.get_stopword()
                    # 获取网络
                    net = myUtils.get_net()
                    count = 0  # 计数器，计算正确的个数
                    for item in all_data:
                        text = json.loads(item.src_twt)
                        # 检测
                        result = myUtils.detect(net, vocab, text['text'], stopwords)
                        # 将检测结果保存到数据库并更新new_add字段为0
                        item.label = result
                        item.new_add = 0
                        item.save()
                        # 检测结果计数
                        if result ^ item.true_label == 0:
                            count += 1
                        if result == 1:  # 是谣言的话计算传播路径并放入datalist显示到前端
                            # 计算传播路径，并根据item.id保存到IMG中
                            # 编辑数据格式
                            # 首先根据谣言id获取全部评论
                            all_comment = models.Content.objects.filter(twitter_id=item.id)
                            # 获取原帖id和获取评论数据的id
                            src_id = json.loads(item.src_twt)['id']
                            l = []  # 存储传播路径父子关系列表
                            l.append([src_id, 0])  # 初始值
                            for i in all_comment:
                                l.append([json.loads(i.text)['id'], json.loads(i.text)['in_reply_to_status_id']])
                            # 调用绘图函数
                            image_name = draw_pic(l)
                            # 将图片名保存到数据库
                            img = models.IMG()
                            img.image_path = 'RD/static/media/'
                            img.name = '/' + image_name + '.png'
                            img.twitter_id = item.id
                            img.save()
                            # 将要显示到前端的数据添加到列表
                            data = dict()
                            data['id'] = src_id
                            data['text'] = text['text']
                            data['label'] = '是谣言'
                            data['detect_time'] = datetime.now().strftime("%Y-%m-%d")  # 检测时间
                            data['img'] = '/static/media/' + image_name + '.png'  # 图片
                            datalist.append(data)

                    accuracy = count / len(all_data)
                    return render(request, 'rumor_detect_02.html', locals())
                except Exception as e:
                    print('str(Exception):\t', str(Exception))
                    print('str(e):\t\t', str(e))
                    print('repr(e):\t', repr(e))
                    print('########################################################')  # 捕获异常
                    return render(request, 'rumor_detect_02.html', locals())
            else:
                if not keyword is None:
                    if len(keyword) == 0:
                        message = "请先输入关键词"
                        return render(request, 'rumor_detect_02.html', locals())
                    else:
                        datalist = search_02(request, keyword)
                        if len(datalist):
                            message = "检索成功！共得到" + str(len(datalist)) + "条数据"
                            return render(request, 'rumor_detect_02.html', locals())
                        else:
                            message = '未查询到包含相关关键词的贴子'
                            return render(request, 'rumor_detect_02.html', locals())
                return render(request, 'rumor_detect_02.html', locals())
    else:
        return render(request, 'rumor_detect_02.html')


def rumor_detect_00(request):
    return render(request, 'rumor_detect_03.html')


def rumor_detect_03(request):
    """长期谣言检测"""
    user_id = request.session['user_id']  # 用户id
    file = request.FILES.get('filename')  # 获取谣言文件对象
    jiance = request.POST.get('jiance')
    keyword = request.POST.get('keyword')
    if request.method == 'POST':
        if not file is None:
            # 根据路径读取数据
            try:
                with codecs.open(os.path.join('RD/static/cx_data/', file.name), 'wb') as f:  # 拼接上传的文件名
                    for i in request.FILES['filename'].chunks():  # 获取文件内容
                        f.write(i)

            except Exception as e:
                print('str(Exception):\t', str(Exception))
                print('str(e):\t\t', str(e))
                print('repr(e):\t', repr(e))
                print('########################################################')  # 捕获异常
                message = "读取文件失败！"
                return render(request, 'rumor_detect_03.html', locals())
        elif not jiance is None:
            # 获取数据
            data_path = os.path.join('RD', 'static', 'cx_data')
            for sDirs in os.listdir(data_path):
                with open(os.path.join(data_path, sDirs), 'r', encoding='utf-8') as f:
                    original_data = json.load(f)
                f.close()

            # 提取原数据中原贴的text内容和标签
            text_label_data = []
            for i in range(len(original_data)):
                each_data = [[json.loads(original_data[i][0])['text']], original_data[i][1]]
                text_label_data.append(each_data)

            # 做分词处理
            word_list = []  # 该集合包含text内容、标签
            for i in range(len(text_label_data)):
                word_list_each = [text_label_data[i][0][0].split(' '), text_label_data[i][1]]
                word_list.append(word_list_each)

            # 去除停用词
            data_handle = remove_stopWords.remove_stop_words(word_list)
            # 进行谣言检测
            # 统计正确的个数
            correct_number = 0
            # 统计所有谣言的相关信息
            rumor_data = []
            for i in range(len(data_handle)):
                rumor_data_each = dict()  # 每条谣言的相关信息
                # 调用谣言检测函数
                judge = cx_model_net.detection(data_handle[i])
                # 将数据集保存到数据库
                try:
                    one_src = models.Twitter()  # 一条原贴

                    # 初始化原贴
                    one_src.src_twt = json.loads(original_data[i][0])  # 原题
                    one_src.true_label = int(json.loads(original_data[i][1]))  # 真实标签
                    one_src.user_id = user_id  # 用户id
                    one_src.detect_type = 3  # 检测类型
                    one_src.label = int(judge)
                    # 向数据库插入数据
                    one_src.save()
                    twitter_id = models.Twitter.objects.all().order_by('-id').first().id  # 新插入值的id
                    # 读取评论列表
                    for t in original_data[i][2:]:
                        one_comment = models.Content()  # 一条评论
                        one_comment.text = t
                        one_comment.twitter_id = twitter_id
                        one_comment.save()  # 保存到数据库
                except Exception as e:
                    print('str(Exception):\t', str(Exception))
                    print('str(e):\t\t', str(e))
                    print('repr(e):\t', repr(e))
                    print('########################################################')  # 捕获异常
                    return render(request, 'rumor_detect_02.html', locals())

                if judge == original_data[i][1]:
                    correct_number += 1
                if judge == '1':
                    rumor_data_each['id'] = json.loads(original_data[i][0])['id']  # 谣言id
                    rumor_data_each['text'] = json.loads(original_data[i][0])['text']  # 谣言内容
                    rumor_data_each['label'] = '1'  # 检测结果
                    # 获取检测的当前时间
                    time_str = datetime.strftime(datetime.now(), '%Y-%m-%d')
                    rumor_data_each['detect_time'] = time_str  # 检测时间

                    # 画传播路径图
                    cb_data_all = []
                    for j in range(len(original_data[i])):
                        cb_data_id = []
                        if j == 0:
                            cb_data_id.append(json.loads(original_data[i][j])['id'])
                            cb_data_id.append(0)
                        elif 2 <= j < len(original_data[i]):
                            cb_data_id.append(json.loads(original_data[i][j])['id'])
                            cb_data_id.append(json.loads(original_data[i][j])['in_reply_to_status_id'])
                        if cb_data_id:
                            cb_data_all.append(cb_data_id)
                    # 调用画图函数，画传播过程图
                    image_name = cb_gc.draw_pic(cb_data_all)
                    # 将图片名保存到数据库
                    img = models.IMG()
                    img.image_path = 'RD/static/media/'
                    img.name = '/' + image_name + '.png'
                    img.twitter_id = models.Twitter.objects.all().order_by('-id').first().id  # 新插入值的id
                    img.save()
                    # 传播路径图的地址
                    img_address = '/static/media/' + image_name + '.png'

                    rumor_data_each['img'] = img_address  # 该谣言传播路径图的存储地址
                    # 存储该谣言的相关信息
                    rumor_data.append(rumor_data_each)

            # 将检测结果保存到数据库
            # save_data(rumor_data)
            # 计算准确率
            accuracy = round(correct_number / len(data_handle), 2)
            accuracy_text = '本次检测的准确率为：' + str(accuracy)
            # print(accuracy)
            return render(request, 'rumor_detect_03.html', {'data_list': rumor_data, 'accuracy_text': accuracy_text})
        elif not keyword is None:
            if len(keyword) == 0:
                message = "请先输入关键词"
                return render(request, 'rumor_detect_03.html', locals())
            else:
                data_list = search_02(request, keyword)
                if len(data_list):
                    message = "检索成功！共得到" + str(len(data_list)) + "条数据"
                    return render(request, 'rumor_detect_03.html', locals())
                else:
                    message = '未查询到包含相关关键词的贴子'
                    return render(request, 'rumor_detect_03.html', locals())
    return render(request, 'rumor_detect_03.html', locals())


def redirect2dt_01(request):
    return redirect('RD.views.rumor_detect_02')


# liumeiq


def read_list(file):
    with open(file, 'rb') as load_f:
        load_dict = json.load(load_f)
    return load_dict


def search_02(request, keyword):
    datalist = []
    try:
        posts = models.Twitter.objects.filter(Q(user_id=request.session['user_id']) & Q(src_twt__contains=keyword))
        # print(posts)
        for post in posts:
            if post.label == 1:
                data = dict()
                data['id'] = post.id
                data['text'] = json.loads(post.src_twt)["text"]
                data['label'] = str(post.label)
                data['detect_time'] = datetime.now().strftime("%Y-%m-%d")
                data['img'] = '/static/media/' + models.IMG.objects.get(twitter_id=post.id).name
                datalist.append(data)
        # print(datalist)
        return datalist
    except Exception as e:
        print('str(Exception):\t', str(Exception))
        print('str(e):\t\t', str(e))
        print('repr(e):\t', repr(e))
        print('########################################################')  # 捕获异常
        return datalist


def show_page(request, num):
    posts = models.Chinese.objects.filter(Q(user=request.session['user_id']) & Q(new_add=1) & Q(label=1))
    # 处理成LayUi官方文档的格式
    datalist = []
    for post in posts[:num]:
        data = dict()
        data['id'] = post.id
        data['text'] = post.source_seq
        "data['label'] = post.label"
        # data['detect_time'] = post.create_time.strftime("%Y-%m-%d %H:%M:%S")
        # data['img'] = models.IMG.objects.get(twitter_id=1).name
        datalist.append(data)
    return datalist


def search(request, keyword):
    datalist = []
    try:
        posts = models.Chinese.objects.filter(Q(source_seq__contains=keyword) & Q(new_add=1))
        # print(posts)
        for post in posts:
            data = dict()
            data['id'] = post.id
            data['text'] = post.source_seq
            "data['label'] = post.label"
            # data['detect_time'] = post.create_time.strftime("%Y-%m-%d %H:%M:%S")
            # data['img'] = models.IMG.objects.get(twitter_id=1).name
            datalist.append(data)
        # print(datalist)
        return datalist
    except:
        return datalist


def rumor_detect_01(request):
    """谣言检测方法1"""
    # 获取输入框的值
    user_id = request.session['user_id']
    file = request.FILES.get('filename')  # 获取谣言文件对象
    # crr_selected = request.POST.get('selected')  # 选择框
    detection = request.POST.get('start_detection')
    dirs = os.listdir('RD/static/data/')  # 所有的文件名
    keyword = request.POST.get('keyword')
    start_search = request.POST.get('start_search')
    # 如果选择了，处理提交的数据

    if not file is None:
        # 根据路径读取数据
        try:
            try:
                os.remove(os.path.join('RD/static/data/', user_id + '.json'))
            except:
                pass
            with open(os.path.join('RD/static/data/', user_id + '.json'), 'wb') as f:
                for i in request.FILES['filename'].chunks():  # 获取文件内容
                    f.write(i)
            message_file = "上传成功！"
            return render(request, 'rumor_detect_01.html', locals())
        except IOError:
            message_file = "读取文件失败！"
            return render(request, 'rumor_detect_01.html', locals())
    else:
        if not detection is None:
            try:
                try:
                    posts = read_list(os.path.join('RD/static/data/', user_id + '.json'))
                except:
                    message = "还未上传数据！请先上传数据"
                posts = read_list(os.path.join('RD/static/data/', user_id + '.json'))
                model_d = torch.load(os.path.join('RD/rumor_model/liumeiqi/', 'RD_GRU_model02.pt'))  # 加载模型
                if torch.cuda.is_available():
                    model_d = model_d.cuda()
                labels = []
                labels_pre = []
                for post in posts:
                    labels.append(post[0])
                    label = rumor_detection.run_val(post, model_d)
                    labels_pre.append(label)
                    '''try:
                        original_post = models.Chinese()

                        original_post.source_seq = post[1]['text']
                        original_post.source = post[1]
                        original_post.comment = post[2:]
                        original_post.label = int(label)
                        original_post.detect_type = 1
                        original_post.new_add = 1
                        user = User.objects.filter(user_id=request.session['user_id']).first()
                        original_post.user = user
                        original_post.true_label = int(post[0])
                        # time.sleep(5)
                        original_post.save()
                    except Exception as e:
                        message = '文件格式错误，上传数据库失败'
                        print('str(Exception):\t', str(Exception))
                        print('str(e):\t\t', str(e))
                        print('repr(e):\t', repr(e))
                        print('########################################################')  # 捕获异常'''
                num_c = 0
                num_post = len(labels)
                num_post = 0
                for i in range(num_post):
                    if labels[i] == labels_pre[i]:
                        num_c += 1
                    if labels_pre[i] == '1':
                        num_post += 1
                accuracy = num_c / num_post
                datalist = show_page(request, num_post)
                if len(datalist):
                    message = "检测成功！共检测到" + str(len(datalist)) + "条谣言"
                else:
                    message = '检测的所有帖子均为非谣言'
                    return render(request, 'rumor_detect_01.html', locals())
                try:
                    os.remove(os.path.join('RD/static/data/', user_id + '.json'))
                except:
                    message = "数据更新失败,请退出重新登录，继续检测可能会出现未知错误"
                return render(request, 'rumor_detect_01.html', locals())
            except IOError:
                return render(request, 'rumor_detect_01.html', locals())
        else:
            if (not keyword is None) and (not start_search is None):
                if len(keyword) == 0:
                    message = "请先输入关键词"
                    return render(request, 'rumor_detect_01.html', locals())
                else:
                    datalist = search(request, keyword)
                    if len(datalist):
                        message = "检索成功！共得到" + str(len(datalist)) + "条数据"
                        return render(request, 'rumor_detect_01.html', locals())
                    else:
                        message = '未查询到包含相关关键词的贴子'
                        return render(request, 'rumor_detect_01.html', locals())
            return render(request, 'rumor_detect_01.html', locals())


def rumor_detect_04(request):
    """谣言检测方法1"""
    # 获取输入框的值
    user_id = request.session['user_id']
    file = request.FILES.get('filename')  # 获取谣言文件对象
    # crr_selected = request.POST.get('selected')  # 选择框
    detection = request.POST.get('start_detection')
    dirs = os.listdir('RD/static/data/')  # 所有的文件名
    keyword = request.POST.get('keyword')
    start_search = request.POST.get('start_search')
    # 如果选择了，处理提交的数据

    if not file is None:
        # 根据路径读取数据
        try:
            try:
                os.remove(os.path.join('RD/static/data/', user_id + '.json'))
            except:
                pass
            with open(os.path.join('RD/static/data/', user_id + '.json'), 'wb') as f:
                for i in request.FILES['filename'].chunks():  # 获取文件内容
                    f.write(i)
            message_file = "上传成功！"
            return render(request, 'rumor_detect_04.html', locals())
        except IOError:
            message_file = "读取文件失败！"
            return render(request, 'rumor_detect_04.html', locals())
    else:
        if not detection is None:
            try:
                try:
                    posts = read_list(os.path.join('RD/static/data/', user_id + '.json'))
                except:
                    message = "还未上传数据！请先上传数据"
                model_d = torch.load(os.path.join('RD/rumor_model/liumeiqi/', 'ERD_best.pkl'))  # 加载模型
                print('范春旭0')
                posts = read_list(os.path.join('RD/static/data/', user_id + '.json'))
                print("范春旭1")
                labels, labels_pre = mytest.run_eval(os.path.join('RD/static/data/', user_id + '.json'), model_d)
                print(labels)
                print(labels_pre)
                print("范春旭2")
                i = 0
                for post in posts:
                    try:

                        original_post = models.Chinese()

                        original_post.source_seq = post[1]['text']
                        original_post.source = post[1]
                        original_post.comment = post[2:]
                        original_post.label = int(labels_pre[i])
                        i += 1
                        original_post.detect_type = 1
                        original_post.new_add = 1
                        user = User.objects.filter(user_id=request.session['user_id']).first()
                        original_post.user = user
                        original_post.true_label = int(post[0])
                        # time.sleep(5)
                        original_post.save()
                    except Exception as e:
                        message = '文件格式错误，上传数据库失败'
                        print('str(Exception):\t', str(Exception))
                        print('str(e):\t\t', str(e))
                        print('repr(e):\t', repr(e))
                        print('########################################################')  # 捕获异常
                num_c = 0
                num = len(labels)
                num_post = 0
                for i in range(num):
                    if labels[i] == labels_pre[i]:
                        num_c += 1
                    if labels_pre[i] == '1':
                        num_post += 1
                accuracy = num_c / num
                datalist = show_page(request, num_post)
                if len(datalist):
                    message = "检测成功！共检测到" + str(len(datalist)) + "条谣言"
                else:
                    message = '检测的所有帖子均为非谣言'
                    return render(request, 'rumor_detect_01.html', locals())
                try:
                    os.remove(os.path.join('RD/static/data/', user_id + '.json'))
                except:
                    message = "数据更新失败,请退出重新登录，继续检测可能会出现未知错误"
                return render(request, 'rumor_detect_04.html', locals())
            except Exception as e:
                print('str(Exception):\t', str(Exception))
                print('str(e):\t\t', str(e))
                print('repr(e):\t', repr(e))
                print('########################################################')  # 捕获异常
                return render(request, 'rumor_detect_04.html', locals())
        else:
            if (not keyword is None) and (not start_search is None):
                if len(keyword) == 0:
                    message = "请先输入关键词"
                    return render(request, 'rumor_detect_04.html', locals())
                else:
                    datalist = search(request, keyword)
                    if len(datalist):
                        message = "检索成功！共得到" + str(len(datalist)) + "条数据"
                        return render(request, 'rumor_detect_04.html', locals())
                    else:
                        message = '未查询到包含相关关键词的贴子'
                        return render(request, 'rumor_detect_04.html', locals())
            return render(request, 'rumor_detect_04.html', locals())
