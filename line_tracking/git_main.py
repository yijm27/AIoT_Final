import numpy as np
import cv2
import matplotlib.pylab as plt

img = cv2.imread("../Img/road5.jpg")
img2 = img.copy()

imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(imgray, 50, 200)

lines = cv2.HoughLinesP(edges, 1, np.pi/180, 10, None, 20, 2)
for line in lines:
    # 검출된 선 그리기
    x1, y1, x2, y2 = line[0]
    cv2.line(img, (x1,y1), (x2, y2), (255,0,0), 3)

imgs = {'Original': img}
for i, (key,value) in enumerate(imgs.items()):
    plt.subplot(1,1,i+1)
    plt.title(key)
    plt.imshow(value, cmap='gray')
    plt.xticks([]); plt.yticks([])

plt.show()