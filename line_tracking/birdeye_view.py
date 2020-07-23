import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread('../Img/road2.png', cv2.IMREAD_GRAYSCALE) # Read the test img
img = cv2.resize(img, dsize=(600, 300))
img_h = img.shape[0]
img_w = img.shape[1]

print(img.shape)
src = np.float32([[0, img_h], [150, img_h], [0, 0], [img_w, 0]])
dst = np.float32([[200, img_h], [243, img_h], [0, 0], [img_w, 0]])
M = cv2.getPerspectiveTransform(src, dst) # The transformation matrix
Minv = cv2.getPerspectiveTransform(dst, src) # Inverse transformation


reg_img = img[int(img_h/3) : int(img_h), 0:int(img_w) ]
warped_img = cv2.warpPerspective(reg_img, M, (img_w, img_h)) # Image warping
imgs = {'Original': reg_img,'bird_eye_view':warped_img}
for i, (key,value) in enumerate(imgs.items()):
    plt.subplot(1,2,i+1)
    plt.title(key)
    plt.imshow(value, cmap='gray')
    plt.xticks([]); plt.yticks([])
plt.show()