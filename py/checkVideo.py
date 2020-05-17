import cv2
import numpy as np

#face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
#eye_cascade = cv2.CascadeClassifier('haarcascade_lefteye_2splits.xml')
num_plate = cv2.CascadeClassifier('cascade.xml')

camera = cv2.VideoCapture(0)#cv2.VideoCapture(path)

#无限循环
while(True):
    #获取视频及返回状态
    ret, img = camera.read()
    #将获取的视频转化为灰色
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #检测视频中的人脸，并用vector保存人脸的坐标、大小（用矩形表示）
    faces = num_plate.detectMultiScale(gray, 1.3, 5)
    #faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    #脸部检测
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        #检测视频中脸部的眼睛，并用vector保存眼睛的坐标、大小（用矩形表示）
        #eyes = eye_cascade.detectMultiScale(roi_gray)
        #眼睛检测
        
        #for (ex,ey,ew,eh) in eyes:
        #    cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        
    #显示原图像
    cv2.imshow('img',img)
    #按q键退出while循环
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

    