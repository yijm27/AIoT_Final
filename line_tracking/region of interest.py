import cv2
import numpy as np
import matplotlib.pylab as plt

img = cv2.imread('../Img/road1.png',cv2.IMREAD_GRAYSCALE)
print(img)
# Image Size
img_h = img.shape[0]
img_w = img.shape[1]

print(img_h)
print(img_w)

reg_img = img[int(img_h/3) : int(img_h), 0:int(img_w) ]
imgs = {'Original': img, 'ROI': reg_img}

for i, (key,value) in enumerate(imgs.items()):
    plt.subplot(1,2,i+1)
    plt.title(key)
    plt.imshow(value, cmap='gray')
    plt.xticks([]); plt.yticks([])
plt.show()