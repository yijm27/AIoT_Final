import cv2
import numpy as np
import matplotlib.pylab as plt

blk_size = 9
C = 3
# image -> grayscale
img = cv2.imread('../Img/road3.jpg', cv2.IMREAD_GRAYSCALE)

blur = cv2.GaussianBlur(img,(5,5), 0) # 가우시안 필터

mean_img = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blk_size, C)
gaussian_img = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,blk_size, C)
t, otsu_img = cv2.threshold(blur, -1, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)


# 경계 값 지정 없이 오츠 알고리즘
# mean_img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blk_size, C)
# gaussian_img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,blk_size, C)
# t, otsu_img = cv2.threshold(img, -1, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

imgs = {'Original': img,'otsu':otsu_img,'gaussian:': gaussian_img, 'mean': mean_img}
for i, (key,value) in enumerate(imgs.items()):
    plt.subplot(1,4,i+1)
    plt.title(key)
    plt.imshow(value, cmap='gray')
    plt.xticks([]); plt.yticks([])

plt.show()