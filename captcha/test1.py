# -*- coding: utf-8 -*-
# python version 3.6.4
import cv2
import numpy as np
import copy


def RGB2HSI(rgb_img):
    """
    这是将RGB彩色图像转化为HSI图像的函数
    :param rgm_img: RGB彩色图像
    :return: HSI图像
    """
    # 保存原始图像的行列数
    row = rgb_img.shape[0]
    col = rgb_img.shape[1]
    # 对原始图像进行复制
    hsi_img = rgb_img.copy()
    # 对图像进行通道拆分
    B, G, R = cv2.split(rgb_img)
    # 把通道归一化到[0,1]
    [B, G, R] = [i / 255.0 for i in ([B, G, R])]
    H = np.zeros((row, col))  # 定义H通道
    I = (R + G + B) / 3.0  # 计算I通道
    S = np.zeros((row, col))  # 定义S通道
    for i in range(row):
        den = np.sqrt((R[i] - G[i]) ** 2 + (R[i] - B[i]) * (G[i] - B[i]))
        thetha = np.arccos(0.5 * (R[i] - B[i] + R[i] - G[i]) / den)  # 计算夹角
        h = np.zeros(col)  # 定义临时数组
        # den>0且G>=B的元素h赋值为thetha
        h[B[i] <= G[i]] = thetha[B[i] <= G[i]]
        # den>0且G<=B的元素h赋值为thetha
        h[G[i] < B[i]] = 2 * np.pi - thetha[G[i] < B[i]]
        # den<0的元素h赋值为0
        h[den == 0] = 0
        H[i] = h / (2 * np.pi)  # 弧度化后赋值给H通道
    # 计算S通道
    for i in range(row):
        min = []
        # 找出每组RGB值的最小值
        for j in range(col):
            arr = [B[i][j], G[i][j], R[i][j]]
            min.append(np.min(arr))
        min = np.array(min)
        # 计算S通道
        S[i] = 1 - min * 3 / (R[i] + B[i] + G[i])
        # I为0的值直接赋值0
        S[i][R[i] + B[i] + G[i] == 0] = 0
    # 扩充到255以方便显示，一般H分量在[0,2pi]之间，S和I在[0,1]之间
    hsi_img[:, :, 0] = H * 255
    hsi_img[:, :, 1] = S * 255
    hsi_img[:, :, 2] = I * 255
    return hsi_img


def HSI2RGB(hsi_img):
    """
    这是将HSI图像转化为RGB图像的函数
    :param hsi_img: HSI彩色图像
    :return: RGB图像
    """
    # 保存原始图像的行列数
    row = np.shape(hsi_img)[0]
    col = np.shape(hsi_img)[1]
    # 对原始图像进行复制
    rgb_img = hsi_img.copy()
    # 对图像进行通道拆分
    H, S, I = cv2.split(hsi_img)
    # 把通道归一化到[0,1]
    [H, S, I] = [i / 255.0 for i in ([H, S, I])]
    R, G, B = H, S, I
    for i in range(row):
        h = H[i] * 2 * np.pi
        # H大于等于0小于120度时
        a1 = h >= 0
        a2 = h < 2 * np.pi / 3
        a = a1 & a2  # 第一种情况的花式索引
        tmp = np.cos(np.pi / 3 - h)
        b = I[i] * (1 - S[i])
        r = I[i] * (1 + S[i] * np.cos(h) / tmp)
        g = 3 * I[i] - r - b
        B[i][a] = b[a]
        R[i][a] = r[a]
        G[i][a] = g[a]
        # H大于等于120度小于240度
        a1 = h >= 2 * np.pi / 3
        a2 = h < 4 * np.pi / 3
        a = a1 & a2  # 第二种情况的花式索引
        tmp = np.cos(np.pi - h)
        r = I[i] * (1 - S[i])
        g = I[i] * (1 + S[i] * np.cos(h - 2 * np.pi / 3) / tmp)
        b = 3 * I[i] - r - g
        R[i][a] = r[a]
        G[i][a] = g[a]
        B[i][a] = b[a]
        # H大于等于240度小于360度
        a1 = h >= 4 * np.pi / 3
        a2 = h < 2 * np.pi
        a = a1 & a2  # 第三种情况的花式索引
        tmp = np.cos(5 * np.pi / 3 - h)
        g = I[i] * (1 - S[i])
        b = I[i] * (1 + S[i] * np.cos(h - 4 * np.pi / 3) / tmp)
        r = 3 * I[i] - g - b
        B[i][a] = b[a]
        G[i][a] = g[a]
        R[i][a] = r[a]
    rgb_img[:, :, 0] = B * 255
    rgb_img[:, :, 1] = G * 255
    rgb_img[:, :, 2] = R * 255
    return rgb_img


def classify(ll, distance):
    ll.sort()
    new = list()
    j = 0
    i = 1
    while i < len(ll):
        if ll[i] - ll[j] <= distance:
            i = i + 1
            if i == len(ll):
                new.append(ll[j:])
        else:
            new.append(ll[j:i])
            j = copy.deepcopy(i)
            i += 1
    return new


def eraseline(rgb_img, near_pixel=3):
    # 转换成HSI模式后，H维度的数字表示人眼所见的颜色，对此维度做聚类
    hsi_img = RGB2HSI(rgb_img)
    d = dict()
    s = set()
    for x in range(hsi_img.shape[0]):
        for y in range(hsi_img.shape[1]):
            s.add(hsi_img[x, y, 0])  # 得到所有颜色的分类
    for each in s:
        d[each] = list()
    for x in range(hsi_img.shape[0]):
        for y in range(hsi_img.shape[1]):
            d[hsi_img[x, y, 0]].append((x, y))  # 得到同一颜色所在的所有的坐标

    fenlei_list = classify(list(d.keys()), near_pixel)
    fenlei_cord_dict = dict()
    fenlei_len_list = list()
    # 同一类颜色，坐标点数 填进列表
    for i, colorlist in enumerate(fenlei_list):
        fenlei_cord_dict[i] = list()
        for color in colorlist:
            fenlei_cord_dict[i] += d[color]
        fenlei_len_list.append((i, len(fenlei_cord_dict[i])))
    fenlei_len_list = sorted(fenlei_len_list, key=lambda x: x[1])
    newimg = np.full(rgb_img.shape, 255, dtype='uint8')
    for cctuple in fenlei_len_list[-5:-1]:
        for cccord in fenlei_cord_dict[cctuple[0]]:
            newimg[cccord[0], cccord[1], 0] = rgb_img[cccord[0], cccord[1], 0]
            newimg[cccord[0], cccord[1], 1] = rgb_img[cccord[0], cccord[1], 1]
            newimg[cccord[0], cccord[1], 2] = rgb_img[cccord[0], cccord[1], 2]
    return newimg


if __name__ == '__main__':
    # 利用opencv读入图片
    file = 'y.jpg'
    rgb_img = cv2.imread(file, cv2.IMREAD_COLOR)
    # 去除连线
    rgb_img_no_line = eraseline(rgb_img)
    cv2.imwrite('after.jpg', rgb_img_no_line)