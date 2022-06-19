import dlib
import numpy
from skimage import io
import cv2

predictor_path = "E:/shape_predictor_68_face_landmarks.dat"
faces_path = "E:/大学课程/大三/Facial-Recognition-Project-main/face/IMG/1223.jpg"

'''加载人脸检测器、加载官方提供的模型构建特征提取器'''
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

win = dlib.image_window()
img = io.imread(faces_path)

win.clear_overlay()
win.set_image(img)

dets = detector(img, 1)
print("Number of faces detected: {}".format(len(dets)))

for k, d in enumerate(dets):
    shape = predictor(img, d)
    landmark = numpy.matrix([[p.x, p.y] for p in shape.parts()])
    print("face_landmark:")
    print(landmark)  # 打印关键点矩阵
    win.add_overlay(shape)  # 绘制特征点
    # for idx, point in enumerate(landmark):
    #     pos = (point[0, 0], point[0, 1])
    #     cv2.putText(img, str(idx), pos, fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
    #                 fontScale=0.3, color=(0, 255, 0))
    #     # cv2.circle(img, pos, 3, color=(0, 255, 0))
    win.set_image(img)
    dlib.hit_enter_to_continue()