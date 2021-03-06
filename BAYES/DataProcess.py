import os
import re
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from skimage import data, filters
from skimage.feature import local_binary_pattern
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

# 定义数据路径
labelPath = 'face/'  # 标签路径
labelFile = ['faceDR', 'faceDS']  # 标签文件
rawDataPath = 'face/rawdata/'  # 图像路径
outputImgPath = 'out/'  # 输出图像路径

# LBP参数的设置
radius = 3  # LBP算法中范围半径的取值
n_points = 8 * radius  # 领域像素点数


# 将二进制的图像数据集转为矩阵，同时负责压缩某些图像，输入DataName（序号），输出矩阵
def getFid(DataName):
    with open(rawDataPath + DataName, 'r') as file:
        array = np.fromfile(file, 'B')
        size = array.size
        reshapeSie = int(size ** 0.5)
        array = array.reshape(reshapeSie, reshapeSie)
        # 将图像从 512*512 压缩到 128*128
        if array.shape == (512, 512):
            array = cv2.resize(array, (128, 128))

        return array


# 过滤全黑的图像数据（灰度值全为0）
def imgDataProcess():
    imgData = []
    skip = []
    for dataName in os.listdir(rawDataPath):
        fid = getFid(dataName)
        if np.all(fid == 0):
            skip.append(dataName)
            continue
        imgData.append(dataName)
    print('\n图像数据为纯黑：\n', skip)
    imgData = pd.DataFrame(imgData, columns=['Seq'])
    return imgData


# 标签数据处理，将去掉缺少标签和完善道具标签栏，返回选择的序号与修改好的标签
def labelDataProcess():
    labelData = []
    miss = []
    for dataName in labelFile:
        with open(labelPath + dataName, 'r') as file:
            for line in file:
                match = re.findall(r"(\d+)|\b_missing.descriptor|\b_(?:sex|age|[rf]ace)\s+(\w+)|_prop..\(([^()]*).\)\)",
                                   line)
                labelList = list(["".join(x) for x in match])

                if len(labelList) == 2:
                    miss.append(labelList[0])
                    continue
                elif len(labelList) == 6:
                    pass
                else:
                    labelList.append('NoProp')

                labelData.append(labelList)
    print('\n标签数据丢失：\n', miss)
    labelData = pd.DataFrame(labelData, columns=['Seq', 'Sex', 'Age', 'Race', 'Face', 'Prop'])
    return labelData


# 合并 经过图像数据处理后筛选出的序号 与 经过标签处理后筛选出的序号，并保存为'pdData.csv'
def combineSeq(imgData, labelData):
    Different = list(set(imgData.iloc[:, 0]) ^ set(labelData.iloc[:, 0]))  # 找出图像数据与标签数据不同的序号
    Seq = [imgData.iloc[:, 0], labelData.iloc[:, 0]][len(imgData) > len(labelData)]  # [A,B][True]返回B

    Data = pd.DataFrame(columns=['Seq', 'Sex', 'Age', 'Race', 'Face', 'Prop'])
    for i in Seq:
        if i in Different:
            print("序号", i, "缺少图像数据或标签数据")
            continue
        a = imgData[imgData['Seq'].isin([i])]
        b = labelData[labelData['Seq'].isin([i])]
        Data = Data.append(pd.merge(a, b, how='left', on='Seq'))
    Data.index = np.arange(len(Data))
    Data.to_csv('pdData.csv', encoding='utf-8')


# 提取合并的序号对应的图像数据（128 * 128）
def combineSeqFid(DataFrame):
    img = np.zeros((len(DataFrame), 128, 128))
    k = range(len(DataFrame))
    for (i, j) in zip(DataFrame['Seq'], k):
        fid = getFid(str(i))
        img[j] = fid
    return img


# 扁平化合并序号对应的图像数据（一维向量）
def combineSeqData(DataFrame):
    data = np.zeros((len(DataFrame), 128 * 128))
    k = range(len(DataFrame))
    for (i, j) in zip(DataFrame['Seq'], k):
        fid = getFid(str(i))
        data[j] = fid.flatten()
    return data


# 将图像数据转为jpg格式的图片
def outputImages(outputImagesPath=outputImgPath):
    if not os.path.exists(outputImagesPath):  # 检查要输出图像的目录outputImagesPath是否存在
        os.mkdir(outputImagesPath)  # 不存在，创建该目录
    for Img in os.listdir(rawDataPath):
        fid = getFid(Img)
        im = Image.fromarray(fid)
        im.save(os.path.join(outputImagesPath, Img) + '.jpg')


# LBP特征
def getLBPFid(DataName):
    array = getFid(DataName)
    lbp = local_binary_pattern(array, n_points, radius)
    return lbp


# 合并LBP特征图像数据并扁平化
def combineLBPSeqData(DataFrame):
    data = np.zeros((len(DataFrame), 128 * 128))
    k = range(len(DataFrame))
    for (i, j) in zip(DataFrame['Seq'], k):
        fid = getLBPFid(str(i))
        data[j] = fid.flatten()
    return data


# 预置一些数据预处理方法：标准化、正则化、二值化
# Standardization
def standardilize(DataFrame):
    x_standard = preprocessing.StandardScaler().fit(DataFrame)
    x = x_standard.transform(DataFrame)
    return x


# Regularization
def regularization(DataFrame):
    x_normalizer = preprocessing.Normalizer().fit(DataFrame)
    x = x_normalizer.transform(DataFrame)
    return x


# Binarization
def binarizer(Dataframe):
    x_binarizer = preprocessing.Binarizer().fit(Dataframe)
    x = x_binarizer.transform(Dataframe)
    return x


# PCA
def pca(Dataframe):
    x_pca = PCA(n_components=76, svd_solver='auto', whiten=True).fit(Dataframe)
    x = x_pca.transform(Dataframe)
    return x


# KNN的参数设置
# kNN(k=1)
def kNN1(Dataframe, y_):
    neigh = KNeighborsClassifier(n_neighbors=1, weights='distance',
                                 algorithm='auto', leaf_size=30, p=2,
                                 metric='minkowski', metric_params=None,
                                 n_jobs=None)
    neigh.fit(Dataframe[:3000], y_[:3000])
    # Take the first 3000 training set images for training
    right = 0
    wrong = 0
    for i in range(3001, 3969):
        # Take the latter 969 training set images for testing
        l = neigh.predict([Dataframe[i]])
        if l == y_[i]:
            right = right + 1
        else:
            wrong = wrong + 1
    accuracy = right / 969
    wrong_rate = wrong / 969
    return [accuracy, wrong_rate]


# kNN(k=3)
def kNN2(Dataframe, y_):
    neigh = KNeighborsClassifier(n_neighbors=3, weights='distance',
                                 algorithm='auto', leaf_size=30, p=2,
                                 metric='minkowski', metric_params=None,
                                 n_jobs=None)
    neigh.fit(Dataframe[:3000], y_[:3000])
    right = 0
    wrong = 0
    for i in range(3001, 3969):
        l = neigh.predict([Dataframe[i]])
        if l == y_[i]:
            right = right + 1
        else:
            wrong = wrong + 1
    accuracy = right / 969
    wrong_rate = wrong / 969
    return [accuracy, wrong_rate]


# kNN(k=5)
def kNN3(Dataframe, y_):
    neigh = KNeighborsClassifier(n_neighbors=5, weights='distance',
                                 algorithm='auto', leaf_size=30, p=2,
                                 metric='minkowski', metric_params=None,
                                 n_jobs=None)
    neigh.fit(Dataframe[:3000], y_[:3000])
    right = 0
    wrong = 0
    for i in range(3001, 3969):
        l = neigh.predict([Dataframe[i]])
        if l == y_[i]:
            right = right + 1
        else:
            wrong = wrong + 1
    accuracy = right / 969
    wrong_rate = wrong / 969
    return [accuracy, wrong_rate]


# kNN(k=7)
def kNN4(Dataframe, y_):
    neigh = KNeighborsClassifier(n_neighbors=7, weights='distance',
                                 algorithm='auto', leaf_size=30, p=2,
                                 metric='minkowski', metric_params=None,
                                 n_jobs=None)
    neigh.fit(Dataframe[:3000], y_[:3000])
    right = 0
    wrong = 0
    for i in range(3001, 3969):
        l = neigh.predict([Dataframe[i]])
        if l == y_[i]:
            right = right + 1
        else:
            wrong = wrong + 1
    accuracy = right / 969
    wrong_rate = wrong / 969
    return [accuracy, wrong_rate]
