import cv2
import numpy as np
import matplotlib.pylab as plt

# 이미지 읽기
img = cv2.imread('../Img/road3.jpg',cv2.IMREAD_GRAYSCALE)

# 이미지 크기
img_h = img.shape[0]
img_w = img.shape[1]

# 구조화 커널 생성
k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

# 열림 연산 적용
gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, k)

#출력
imgs = {'Original':img, 'Merged': gradient}
for i, (key,value) in enumerate(imgs.items()):
    plt.subplot(1,2,i+1)
    plt.title(key)
    plt.imshow(value, cmap='gray')
    plt.xticks([]); plt.yticks([])
plt.show()