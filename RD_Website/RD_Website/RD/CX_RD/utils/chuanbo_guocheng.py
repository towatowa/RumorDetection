import matplotlib.pyplot as plt
import string
import os

# 图片保存路径
images_path = os.path.join('RD', 'static', 'media')


# -------数据预处理---------
def preprocess(data):
    # 求最大值
    def find_max(q):
        m = []
        for z in range(len(q)):
            if len(q[z]) == 3:
                m.append(q[z][2])
        return max(m)

    # 给数据分组
    data[0].append(0)
    index = 1
    s = 0
    for i in range(len(data)):
        for j in range(1, len(data)):
            if data[i][0] == data[j][1] and len(data[i]) == 3:
                data[j].append(data[i][2] + 1)
                s = index
        index = s + 1

    # 确定每条数据的坐标，及坐标轴的范围
    max_data = find_max(data)

    # 如果数据中有没有分好组的数据，对这些数据进行分组
    for i in range(len(data)):
        if len(data[i]) != 3:
            data[i].append(max_data + 1)

    index = 1
    s = 0
    for i in range(len(data)):
        for j in range(1, len(data)):
            if data[i][0] == data[j][1] and len(data[i]) == 3:
                data[j][2] = data[i][2] + 1
                s = index
        index = s + 1

    # 再次确定坐标轴的范围
    max_data = find_max(data)

    # X为获得x坐标轴的范围
    X = max_data + 1
    Y_axis = []

    # 确定每条数据的坐标
    # data[0].append([0, 0])
    for i in range(0, max_data + 1):
        k = 1
        if i == 0:
            data[i].append([0, 0])
        else:
            for j in range(len(data)):
                if i == data[j][2]:
                    data[j].append([i, k])
                    k += 1
        Y_axis.append(k)

    # Y为获得y坐标轴的范围
    Y = max(Y_axis)
    '''
    for i in range(len(data)):
        print(i, data[i])
    '''
    return data, X, Y


# -------画图---------
def draw_pic(data_all):
    # 处理数据
    data, X, Y = preprocess(data_all)

    # ---------- 画图 ----------
    fig, ax = plt.subplots()
    plt.xlim(0, X)  # x轴的刻度范围
    plt.ylim(0, Y)  # y轴的刻度范围

    # 折线图
    # 连接各个点
    for i in range(1, len(data)):
        for j in range(0, len(data)):
            if data[i][1] == data[j][0]:
                start = (data[j][3][0], data[i][3][0])
                end = (data[j][3][1], data[i][3][1])
                # ax.plot(start, end, color='royalblue', lw=2.5)
                ax.annotate('', xy=(data[i][3][0], data[i][3][1]),
                            xytext=(data[j][3][0], data[j][3][1]),
                            arrowprops=dict(arrowstyle='->', color='b'))

    # 折线图上的散点
    for i in range(len(data)):
        # ax.text(data[i][3][0], data[i][3][1], str(data[i][0]))
        ax.scatter(data[i][3][0], data[i][3][1], marker='o', c='firebrick')

    # 给每个点做标注
    po_annotation = []
    for i in range(len(data)):
        # 标注点的坐标
        point, = plt.plot(data[i][3][0], data[i][3][1], 'o', color='firebrick')
        # 标注框偏移量
        offset1 = 20
        offset2 = 20
        # 标注框
        bbox = dict(boxstyle='round', fc='salmon', alpha=0.6)
        # 标注箭头
        arrowprops = dict(arrowstyle='->', connectionstyle='arc3,rad=0.')
        # 标注信息
        # str_info = ''.join(c for c in str(data[i][0]) if c not in string.punctuation)
        str_info = str(data[i][0])
        annotation = plt.annotate(str_info, xy=(data[i][3][0], data[i][3][1]),
                                  xytext=(-offset1, offset2), textcoords='offset points',
                                  bbox=bbox, arrowprops=arrowprops, size=10)
        # 默认鼠标未指向时不显示标注信息
        annotation.set_visible(False)
        po_annotation.append([point, annotation])

    # 定义鼠标响应函数
    def on_move(event):
        visibility_changed = False
        for points, annotations in po_annotation:
            should_be_visible = (points.contains(event)[0] == True)
            if should_be_visible != annotations.get_visible():
                visibility_changed = True
                annotations.set_visible(should_be_visible)
        if visibility_changed:
            plt.draw()

    # 鼠标移动事件
    on_move_id = fig.canvas.mpl_connect('motion_notify_event', on_move)

    # plt.show()
    # 保存图片
    plt.savefig(os.path.join(images_path, str(data[0][0])))
    return str(data[0][0])
