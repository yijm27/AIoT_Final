import cv2
import numpy as np
import matplotlib.pylab as plt

# image -> grayscale
img = cv2.imread('../Img/road2.png')
dst = img.copy()
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 경계 값 지정 없이 오츠 알고리즘
t, otsu_img = cv2.threshold(img, -1, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# 구조화 커널 생성
k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

# 열림 연산 적용
gradient = cv2.morphologyEx(otsu_img, cv2.MORPH_GRADIENT, k)

# houghLines
thr = 90
houghLines = cv2.HoughLinesP(gradient, 0.8, np.pi/180, thr, minLineLength= 20, maxLineGap= 40)
for line in houghLines:
    cv2.line(dst, (line[0][0], line[0][1]), (line[0][2], line[0][3]), (0, 0, 255), 2)

imgs = {'Original': img,'Otsu':otsu_img, 'Gradient':gradient}

for i, (key,value) in enumerate(imgs.items()):
    plt.subplot(1,3,i+1)
    plt.title(key)
    plt.imshow(value, cmap='gray')
    plt.xticks([]); plt.yticks([])

plt.show()