import cv2
import numpy as np
import matplotlib.pylab as plt

img = cv2.imread('../Img/road2.png',cv2.IMREAD_GRAYSCALE)

# 이미지 크기
img_h = img.shape[0]
img_w = img.shape[1]

t, otsu_img = cv2.threshold(img, -1, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

edge = cv2.Canny(otsu_img, 0, 255)


blur = cv2.GaussianBlur(edge,(5,5), 0) # 가우시안 필터
blur2 = cv2.bilateralFilter(blur, 5, 75, 75)
imgs = {'Original': img,'edge':edge}
for i, (key,value) in enumerate(imgs.items()):
    plt.subplot(1,2,i+1)
    plt.title(key)
    plt.imshow(value, cmap='gray')
    plt.xticks([]); plt.yticks([])

plt.show()
