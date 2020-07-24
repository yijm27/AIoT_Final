import cv2
import numpy as np
#import matplotlib.pylab as plt
import sys
import os
import math
import Jetson.GPIO as GPIO
current_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_path)
from Modules.Pca9685 import Pca9685
from Modules.Sg90 import Sg90
from datetime import datetime
import matplotlib.pyplot as plt

#%% 이미지 캡처
cap = cv2.VideoCapture('../Img/test_1.mp4')
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 400)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 250)
#%%
a = 0
b = 0
error_theta = 0
error_sum = 0
pre_error_theta = 0
kp = 1
ki = 0.003
kd = 0.005
data = []
data2 = []
last_time = round(datetime.utcnow().timestamp() * 1000)
pca9685 = Pca9685()
sv = Sg90(pca9685, 0)
sv.angle(90)

while(cap.isOpened()):
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img_w = frame.shape[1]
    img_h = frame.shape[0]
    scale = img_w + img_h

    # 가우시안 블러
    blur_frame = cv2.GaussianBlur(gray, (3,3), 0)

    # 캐니 엣지
    canny_frame = cv2.Canny(blur_frame, 40, 210)

    # ROI
    if img_h > img_w:
        temp = img_h
        temp2 = img_w

    else:
        temp = img_w
        temp2 = img_h

    vertices = np.array([[(0, img_h),(0, img_h/2),(img_w,img_h/2),(img_w, img_h)]], dtype=np.int32)
    mask = np.zeros_like(canny_frame)  # mask = img와 같은 크기의 빈 이미지
    color = 255

    # vertices에 정한 점들로 이뤄진 다각형부분(ROI 설정부분)을 color로 채움
    cv2.fillPoly(mask, vertices, color)

    # 이미지와 color로 채워진 ROI를 합침
    ROI_image = cv2.bitwise_and(canny_frame, mask)
    #print(ROI_image.shape)

    # 교점 검출용 선
    cross_y = int(img_h * 2 / 3)
    # cv2.line(img_copy, (0, cross_y), (img_w,cross_y), (0,255,0), 3)

    # 확률 허프 변환
    thr = int(scale / 10)
    houghLines = cv2.HoughLinesP(ROI_image, 1, np.pi/180, 30, minLineLength= 10, maxLineGap= 20)
    cross_xs = []
    temps = []

    if houghLines is not None:
        for index, line in enumerate(houghLines):
            if (line[0][2] - line[0][0]) != 0:
                temp = (line[0][3] - line[0][1]) / (line[0][2] - line[0][0])
                if temp != 0 and 80 > temp and abs(temp) > 0.3:
                    cv2.line(frame, (line[0][0], line[0][1]), (line[0][2], line[0][3]), (0, 0, 255), 2)

                    # 직선의 방정식
                    cross_x = (cross_y - line[0][3]) / temp + line[0][2]
                    cross_xs.append(cross_x)
                    temps.append(temp)

    # 중심점 찾기
    sub = a
    if len(cross_xs) is not 0:
        left_spot = (min(cross_xs))
        right_spot = (max(cross_xs))

        if (right_spot - left_spot) > 50:
            middle_spot = (right_spot - left_spot) / 2 + left_spot
            a = middle_spot - left_spot
            b = middle_spot
            #print(middle_spot)  ### 중심좌표 -> (middle_spot, cross_y)
            cv2.circle(frame, (int(middle_spot), cross_y), 10, (0, 0, 255), -1)
            cv2.circle(frame, (int(img_w/2), cross_y), 10, (0,255,0), -1)
        else:
            if (left_spot + sub) <= (b+50): # 중심에서 50정도
               # print("sub: {}".format(sub))
                cv2.circle(frame, (int(left_spot) + int(sub), cross_y), 10, (255, 0, 255), -1)
            else :
                #print("sub2: {}".format(sub))
                cv2.circle(frame, (int(right_spot) - int(sub), cross_y), 10, (255, 255, 0), -1)
    middle_wgap = int(img_w/2 - middle_spot)
    middle_hgap = int(img_h - cross_y)

    # 중심 좌표 찾기
    if middle_wgap < 0:
        middle_wgap = -middle_wgap
        cv2.line(frame, (int(middle_spot), cross_y), (int(img_w / 2), cross_y), (0,0,0), 3)
        error_theta = (180/np.pi) *np.arctan(middle_wgap/ middle_hgap)

    elif middle_wgap == 0:
        error_theta = 0

    else:
        cv2.line(frame, (int(middle_spot), cross_y), (int(img_w / 2), cross_y), (255, 255, 255), 3)
        error_theta = (180/np.pi) *np.arctan(middle_wgap/ middle_hgap)
        error_theta = -1 * error_theta


############### PID Controller
    cur_time = round(datetime.utcnow().timestamp() * 1000) # ms 얻기
    dt = (cur_time - last_time) * 0.001
    print("dt:{}s".format(dt))
    error_sum = error_sum+ error_theta

    p_value = kp * error_theta
    i_value = ki * error_sum
    d_value = kd * (error_theta-pre_error_theta)/dt
    target_theta = p_value + i_value + d_value
    print("target_theta:{}".format(target_theta))

#################### 시간 갱신###############
    pre_error_theta = error_theta
    last_time = round(datetime.utcnow().timestamp() * 1000)
    if target_theta > 50 :
        target_theta = 50
    elif target_theta < -50:
        target_theta = -50
    real_theta = 90 - target_theta
   # real_theta = 90 - error_theta
    if real_theta > 140:
        real_theta = 140
    elif real_theta < 40:
        real_theta = 40

    sv.angle(real_theta)
    print(real_theta)
    data.append(str(error_theta))
    data2.append(str(real_theta-90))
  #  print(data)
########################### 출력 #################################
    if frame is not None:
        cv2.imshow('frame', frame)
    else:
        break
    if cv2.waitKey(33) & 0xFF == ord('q'):
        break
f = open('data.csv', 'w')
for i in range(len(data)):
    f.write(str(i) + ',' + data[i] + ',' + data2[i] + '\n')
f.close()
cap.release()
cv2.destroyAllWindows()
