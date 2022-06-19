import gc
import os
import datetime
import xlrd
import cv2
import skimage.feature
import skimage.segmentation
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import Perceptron
from sklearn.model_selection import cross_val_score
from skimage.feature import hog
from skimage import exposure, filters

# 提取标签数据
DatasetPath = './face/'
name = ['faceDR', 'faceDS']  # 存放标签的文件名
error = []
labelraw_1 = []  # 元素是英文 funny smiling serious 和 missing
labelnum_1 = []  # 元素 funny smiling serious 和 missing 分别将标签标号为1,2,3,4
for i in name:
    with open(DatasetPath + i, 'r') as file1:
        for line in file1:
            num = int(line[1:5])
            if 'missing' in line:
                error.append(num)
            else:
                label1 = line.split('_face ', 1)[1].split(') (_prop')[0]  # 通过分割文本提取表情对应的文本，如smiling
                labelraw_1.append(label1)

# 根据表情对应的文本生成标签，元素1,2,3分别代表 'funny'，'smiling'，'serious'
for element in labelraw_1:
    if element == 'funny':
        labelnum_1.append(1)
    if element == 'smiling':
        labelnum_1.append(2)
    if element == 'serious':
        labelnum_1.append(3)

# ----------  对于rawdata而言  ------------
# 提取图像数据
ImgPath_1 = './face/rawdata/'
data = []
inferior = []
filelist_1 = os.listdir(ImgPath_1)
bigerror1 = []  # 存放尺寸过大图像的编号
rawdata_1 = np.zeros((3991, 16384))  # 存放图像的像素

i = 0
for name1 in filelist_1:
    with open(ImgPath_1 + str(name1), 'r') as file1:
        array1 = np.fromfile(file1, 'B')
        if array1.size == 128 * 128:
            rawdata_1[i] = array1
            i += 1
        else:
            bigerror1.append(name1)
# 删除尺寸过大的图片
for a in bigerror1:
    index = filelist_1.index(str(a))
    del labelnum_1[index]

# ----------  对于IMG而言  ------------
ImgPath_2 = './face/IMG/'
data = []
inferior = []
filelist_2 = os.listdir(ImgPath_2)
path = os.path.join(ImgPath_2)
rawdata_2 = np.zeros((3690, 16384))  # 经过dlib处理过的图像


labelraw_2 = []  # 元素是英文 funny smiling serious 和 missing
labelnum_2 = []  # 元素 funny smiling serious 和 missing 分别将标签标号为1,2,3,4
wb = xlrd.open_workbook("E:\大学课程\大三\Facial-Recognition-Project-main\other\data.xls")  # 打开文件并返回一个工作簿
sheet_num = wb.nsheets  # 获取excel里面的sheet的数量
sheet_names = wb.sheet_names()  # 获取到Excel里面所有的sheet的名称列表，即使没有sheet也能用。
sheet = wb.sheet_by_index(0)  # 通过索引的方式获取到某一个sheet，现在是获取的第一个sheet页，也可以通过sheet的名称进行获取，sheet_by_name('sheet名称')
rows = sheet.nrows  # 获取sheet页的行数，一共有几行
columns = sheet.ncols  # 获取sheet页的列数，一共有几列
# 获取第二列的数据
col_data = sheet.col_values(1)

# 获取单元格的数据
for m in range(0, len(col_data)):
    if col_data[m] == 'funny':
        labelnum_2.append(1)
    if col_data[m] == 'smiling':
        labelnum_2.append(2)
    if col_data[m] == 'serious':
        labelnum_2.append(3)


# 提取图像hog特征
def use_hog(x):
    hogdata = np.zeros((x.shape[0], x.shape[1]))  #
    for i in range(x.shape[0]):
        image = x[i].reshape(int(x.shape[1] ** 0.5), int(x.shape[1] ** 0.5))
        fd, hog_image = hog(image, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(1, 1), visualize=True)
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
        hogdata[i] = hog_image_rescaled.reshape(x.shape[1])
    return hogdata


# 提取图像lbp特征
def use_lbp(x):
    lbpdata = np.zeros((x.shape[0], x.shape[1]))
    for i in range(x.shape[0]):
        image = x[i].reshape(int(x.shape[1] ** 0.5), int(x.shape[1] ** 0.5))
        img_lbp = skimage.feature.local_binary_pattern(image, 8, 1.0, method='default')
        img_lbp = img_lbp.astype(np.uint8)
        lbpdata[i] = img_lbp.reshape(x.shape[1])
    return lbpdata


# PCA主成分分析降维
def usePCA(x, n_components):
    pca = PCA(n_components=n_components)
    pca.fit(x)
    x_pca_data = pca.transform(x)
    return x_pca_data


# 标准化
def stander(x_data):
    sc = StandardScaler()
    sc.fit(x_data)
    x_all_std = sc.transform(x_data)
    return x_all_std


# 训练多层感知机分类器
def trainBPmodel(size1, size2, max_iter, learning_rate, X_train, X_test, y_train, y_test, X_all_std):
    start = datetime.datetime.now()
    clf = MLPClassifier(solver='lbfgs', learning_rate='adaptive', learning_rate_init=learning_rate,
                        hidden_layer_sizes=(size1, size2), random_state=1, max_iter=max_iter)
    clf.fit(X_train, y_train)
    end = datetime.datetime.now()
    print('spend time:', end - start, 's')
    y_pred1 = clf.predict(X_train)
    right_classified = (y_pred1 == y_train).sum()
    print("bp trainAcc: ", right_classified / y_pred1.size)
    print("bp testAcc: ", clf.score(X_test, y_test))
    scores = cross_val_score(clf, X_all_std, y_1, cv=5)
    print("cross_val_score Acc: ", scores.sum() / scores.size)
    cross_val_acc = scores.sum() / scores.size
    return cross_val_acc


# 训练感知机分类器
def trainppmodel(times, learning_rate, X_train, X_test, y_train, y_test, X_all_std):
    clf = Perceptron(max_iter=times, eta0=learning_rate, random_state=1)  # 40
    clf.fit(X_train, y_train)
    y_pred1 = clf.predict(X_train)
    right_classified = (y_pred1 == y_train).sum()
    print("pp trainAcc: ", right_classified / y_pred1.size)
    print("pp testAcc: ", clf.score(X_test, y_test))
    scores = cross_val_score(clf, X_all_std, y_1, cv=5)
    print("score: ", scores)
    print("cross_val_score Acc: ", scores.sum() / scores.size)


# 选择使用hog或pca处理图像
def processdata(x, y, n_components, usepca, usehog):
    if usehog == 1:
        x_data = use_hog(x)
    if usehog == 0:
        x_data = x
    if usepca == 1:
        x_pca_data = usePCA(x_data, n_components)
    if usepca == 0:
        x_pca_data = x_data
    X_all_std = stander(x_pca_data)
    X_train, X_test, y_train, y_test = train_test_split(X_all_std, y, test_size=0.3, random_state=1, stratify=y)
    return X_train, X_test, y_train, y_test, X_all_std


# 打印图像
def showimage(image_data, size1, size2):
    image = image_data.reshape(size1, size2)
    plt.imshow(image)
    plt.show()


# 截取人脸图像中嘴巴部分的图片
def cutimage(X):
    X_cut = np.zeros((X.shape[0], 40, 40))
    X_reshape = X.reshape(X.shape[0], 128, 128)
    for i in range(X.shape[0]):
        for j in range(70, 110):
            for k in range(50, 90):
                X_cut[i][j - 70][k - 50] = X_reshape[i][j][k]
    X_cut_reshape = X_cut.reshape(X.shape[0], 1600)
    return X_cut_reshape


# 用两个过滤器对图像进行简单卷积
def extraimage(x, Step):
    x_reshape = x.reshape(x.shape[0], int(x.shape[1] ** 0.5), int(x.shape[1] ** 0.5))
    step = Step
    size1 = x.shape[0]  # 图像个数
    size4 = 3
    size5 = 3
    size2 = int(((x.shape[1] ** 0.5) - size4) / step + 1)  # output行数
    size3 = int(((x.shape[1] ** 0.5) - size5) / step + 1)  # output列数
    filter1 = np.array([[2, 0, 2], [0, 3, 0], [2, 0, 2]])
    filter2 = np.array([[0, 3, 0], [1, 0, 1], [0, 4, 0]])
    cutdata = np.zeros((size1, size4, size5))
    conv = [[0 for q in range(0)] for w in range(size1)]
    for i in range(size1):
        for j in range(size2):  # 过滤器竖着扫
            for k in range(size3):  # 过滤器横着扫
                for l in range(size4):
                    for h in range(size5):
                        cutdata[i][l][h] = x_reshape[i][l + j * step][h + k * step]
                num = (cutdata[i] * filter1 + cutdata[i] * filter2).sum()
                conv[i].append(num)
    conv = np.array(conv)
    return conv


X_1 = rawdata_1  # X是展平后的原始图片数据，维度为（3991，16384） 即3991张图片每张有16384个像素
X_2 = rawdata_2  # X是展平后的原始图片数据，维度为（3690，16384） 即3690张图片每张有16384个像素
y_1 = np.array(labelnum_1)  # y是图片的表情标签，元素有1,2,3，分别代表 'funny' ’smiling' 和 'serious'
y_2 = np.array(labelnum_2)  # y是图片的表情标签，元素有1,2,3，分别代表 'funny' ’smiling' 和 'serious'
