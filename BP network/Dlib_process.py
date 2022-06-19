import dlib  # 人脸识别的库dlib
import numpy as np  # 数据处理的库numpy
import cv2  # 图像处理的库OpenCv
import os

# dlib预测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('model/shape_predictor_68_face_landmarks.dat')

# 读取图像的路径
path_read = "face/output"
num = 0
for file_name in os.listdir(path_read):
    # aa是图片的全路径
    aa = (path_read + "/" + file_name)
    # 读入的图片的路径中含非英文
    img = cv2.imdecode(np.fromfile(aa, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    # 获取图片的宽高
    img_shape = img.shape
    img_height = img_shape[0]
    img_width = img_shape[1]

    # 用来存储生成的单张人脸的路径
    path_save = "face/IMG"
    # dlib检测
    dets = detector(img, 1)
    print("人脸数：", len(dets))

    if len(dets) == 0:
        print("该输入图片未检测到人脸，将其写入文本中---")
        print(aa, file=open("./other/fail.txt", "a"))

    for k, d in enumerate(dets):
        if len(dets) > 1:
            continue
        num = num + 1
        # 计算矩形大小
        # (x,y), (宽度width, 高度height)
        pos_start = tuple([d.left(), d.top()])
        pos_end = tuple([d.right(), d.bottom()])

        # 计算矩形框大小
        height = d.bottom() - d.top()
        width = d.right() - d.left()

        # 根据人脸大小生成空的图像
        img_blank = np.zeros((height, width, 3), np.uint8)
        for i in range(height):
            if d.top() + i >= img_height:  # 防止越界
                continue
            for j in range(width):
                if d.left() + j >= img_width:  # 防止越界
                    continue
                img_blank[i][j] = img[d.top() + i][d.left() + j]
        img_blank = cv2.resize(img_blank, (128, 128), interpolation=cv2.INTER_CUBIC)

        cv2.imencode('.jpg', img_blank)[1].tofile(path_save + "\\" + str(num) + ".jpg")  # 正确方法
