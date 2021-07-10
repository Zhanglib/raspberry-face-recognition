# -*- coding: utf-8 -*-
import os
import cv2
import sys
import gc
import time
import datetime
import numpy as np
from model_train import Model
from picscal import rotate
#第四部，实现人脸识别
def unevenLightCompensate(img, blockSize, min_num,max_num):
    gray = img
    average = np.mean(gray)

    rows_new = int(np.ceil(gray.shape[0] / blockSize))
    cols_new = int(np.ceil(gray.shape[1] / blockSize))

    blockImage = np.zeros((rows_new, cols_new), dtype=np.float32)
    for r in range(rows_new):
        for c in range(cols_new):
            rowmin = r * blockSize
            rowmax = (r + 1) * blockSize
            if (rowmax > gray.shape[0]):
                rowmax = gray.shape[0]
            colmin = c * blockSize
            colmax = (c + 1) * blockSize
            if (colmax > gray.shape[1]):
                colmax = gray.shape[1]

            imageROI = gray[rowmin:rowmax, colmin:colmax]
            temaver = np.mean(imageROI)
            blockImage[r, c] = temaver
    blockImage = blockImage - average
    blockImage2 = cv2.resize(blockImage, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_CUBIC)
    # blockImage2[200:800,200:500]=0
    gray2 = gray.astype(np.float32)
    dst = gray2 - blockImage2
    for i in range(len(dst)):
        for j in range(len(dst[0])):
            if dst[i][j]<min_num:
                dst[i][j]=min_num
            elif dst[i][j]>max_num:
                dst[i][j]=max_num
    dst = dst.astype(np.uint8)
    # dst = cv2.GaussianBlur(dst, (3, 3), 0)

    return dst

if __name__ == '__main__':
    if len(sys.argv) != 1:
        print("Usage:%s camera_id\r\n" % (sys.argv[0]))
        sys.exit(0)

    # 加载模型
    model = Model()
    MODEL_PATH = '/home/pi/Desktop/face_recognition_opencv/model.h5'
    #MODEL_PATH=MODEL_PATH.encode('utf-8')
    model.load_model(file_path=MODEL_PATH)

    # 框住人脸的矩形边框颜色
    color = (0, 255, 0)

    # 捕获指定摄像头的实时视频流
    cap = cv2.VideoCapture(0)

    # 人脸识别分类器本地存储路径
    cascade_path = "/home/pi/Desktop/face_recognition_opencv/haarcascade_frontalface_alt2.xml"

    cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
    cascade.load('/home/pi/Desktop/face_recognition_opencv/haarcascade_frontalface_alt2.xml')  # 一定要告诉编译器文件所在的具体位置

    # 循环检测识别人脸
    while True:
        ret, frame = cap.read()  # 读取一帧视频
        frame=rotate(frame,180)
       # cv2.imshow("识别朕", frame)
      #  img_norm = frame/255.0  
       # img_gamma = np.power(img_norm,0.7)*255.0
        #img_gamma = img_gamma.astype(np.uint8)
        #frame = cv2.cvtColor(img_gamma,cv2.COLOR_BGR2HSV)
    
        #dst2 = unevenLightCompensate(frame[:,:,2], 1000,0,255)
        #frame[:,:,2] = dst2
        #qframe = cv2.cvtColor(frame,cv2.COLOR_HSV2BGR)
    

        if ret is True:

            # 图像灰化，降低计算复杂度
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #cv2.imshow("hui",frame_gray)
        else:
            continue
        # 使用人脸识别分类器，读入分类器
        # = cv2.CascadeClassifier(cascade_path)
        cascade = cv2.CascadeClassifier(cascade_path)

        # 利用分类器识别出哪个区域为人脸q
        faceRects = cascade.detectMultiScale(frame_gray, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
        if len(faceRects) > 0:
            for faceRect in faceRects:
                x, y, w, h = faceRect

                # 截取脸部图像提交给模型识别这是谁
                image = frame[y - 10: y + h + 10, x - 10: x + w + 10]
                faceID = model.face_predict(image)
                
                print("faceID", faceID)
                # 如果是“我”
                if faceID == 0:
                    cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, thickness=2)

                    # 文字提示是谁
                    time1=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S');
                    
                    file=open("/home/pi/Downloads/apache-tomcat-7.0.109/webapps/Server/data.txt",'a+')
                    file.writelines("\nzhanglibeing   ")
                    file.write(str(time1))
                
                    file.close()
                    
                    cv2.putText(frame, 'zhanglibeing',
                                (x + 30, y + 30),  # 坐标
                                cv2.FONT_HERSHEY_SIMPLEX,  # 字体
                                1,  # 字号
                                (255, 0, 255),  # 颜色
                                2)  # 字的线宽
                    #cv2.imshow("识别朕", frame)
                    os.system("python3 /home/pi/Desktop/face_recognition_opencv/code_v1/pwm.py")
                    #time.sleep(3)
                    os.system("python3 /home/pi/Desktop/face_recognition_opencv/code_v1/pwm2.py")
                    #time.sleep(10)
                elif faceID==2:
                    cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, thickness=2)
                    cv2.putText(frame, 'asy',
                                (x + 30, y + 30),  # 坐标
                                cv2.FONT_HERSHEY_SIMPLEX,  # 字体
                                1,  # 字号
                                (255, 0, 0),  # 颜色
                                2)  # 字的线宽
                else:
                    
                    cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, thickness=2)
                    cv2.putText(frame, 'unknown',
                                (x + 30, y + 30),  # 坐标
                                cv2.FONT_HERSHEY_SIMPLEX,  # 字体
                                1,  # 字号
                                (255, 0, 0),  # 颜色
                                2)  # 字的线宽
                    #cv2.imshow("识别朕", frame)
                    pass

        cv2.imshow("识别朕", frame)

        # 等待10毫秒看是否有按键输入
        k = cv2.waitKey(10)
        # 如果输入q则退出循环
        if k & 0xFF == ord('q'):
            break

    # 释放摄像头并销毁所有窗口
    cap.release()
    cv2.destroyAllWindows()
    