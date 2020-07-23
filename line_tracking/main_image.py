import cv2
import numpy as np
import matplotlib.pylab as plt
import math
###################################
# 1.  가우시안블러
# 1-2. otsu threshold
# 3.  에지 검출
# 4.  관심영역 잘라내기
# 5.  확률 허프 변환 직선 찾기
###################################

# 이미지 불러오기
img = cv2.imread('../Img/test8.PNG')

img_copy = img.copy()
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img_w = img.shape[1]
img_h = img.shape[0]
scale = img_w + img_h
print(img_w, img_h)

# # otsu 스레쉬홀딩
# t, otsu_img = cv2.threshold(img_gray, -1, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# 가우시안 블러
blur_img = cv2.GaussianBlur(img_gray, (3,3), 0)

# # 가우시안 스레쉬홀딩
# blk_size = 9
# C = 3
# gaussian_img = cv2.adaptiveThreshold(blur_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,blk_size, C)



# 캐니 엣지
canny_img = cv2.Canny(blur_img, 50, 210)

# ROI
if img_h > img_w:
    temp = img_h
    temp2 = img_w
else:
    temp = img_w
    temp2 = img_h
vertices = np.array([[(0, img_h),(0, img_h*2/3 - 50) ,(img_w, img_h*2/3 -50),(img_w, img_h)]], dtype=np.int32)
mask = np.zeros_like(canny_img)  # mask = img와 같은 크기의 빈 이미지
color = 255

# vertices에 정한 점들로 이뤄진 다각형부분(ROI 설정부분)을 color로 채움
cv2.fillPoly(mask, vertices, color)

# 이미지와 color로 채워진 ROI를 합침
ROI_image = cv2.bitwise_and(canny_img, mask)
print(ROI_image.shape)

# 교점 검출용 선
cross_y = int(img_h*2/3)
#cv2.line(img_copy, (0, cross_y), (img_w,cross_y), (0,255,0), 3)

# 확률 허프 변환
thr = int(scale / 14)
houghLines = cv2.HoughLinesP(ROI_image, 1, np.pi/180, 30, minLineLength= 10, maxLineGap= 20)
cross_xs = []
temps = []
if houghLines is not None:
    for index, line in enumerate(houghLines):
        if (line[0][2] - line[0][0]) != 0:
            temp = (line[0][3] - line[0][1]) / (line[0][2] - line[0][0])
            if temp != 0 and 80 > temp and abs(temp) > 0.3:
                cv2.line(img_copy, (line[0][0], line[0][1]),(line[0][2], line[0][3]),(0,0,255), 2)
                # 직선의 방정식
                cross_x = (cross_y-line[0][3])/temp + line[0][2]
                cross_xs.append(cross_x)
                temps.append(temp)

print (cross_xs, temps)

# 중심점 찾기
if len(cross_xs) is not 0:
    left_spot = (min(cross_xs))
    right_spot = (max(cross_xs))
    middle_spot = (right_spot - left_spot)/2 + left_spot

    print(middle_spot) ### 중심좌표 -> (middle_spot, cross_y)
    cv2.circle(img_copy, (int(middle_spot),cross_y),10, (255, 0, 255), -1)

# 중심점 찾기

if len(cross_xs) is not 0:
    left_spot = (min(cross_xs))
    right_spot = (max(cross_xs))
    middle_spot = (right_spot - left_spot) / 2 + left_spot
    a = middle_spot - left_spot
    b = middle_spot
    #print(middle_spot)  ### 중심좌표 -> (middle_spot, cross_y)
    cv2.circle(img_copy, (int(middle_spot), cross_y), 10, (0, 0, 255), -1)
    cv2.circle(img_copy, (int(img_w/2), cross_y), 10, (0,255,0), -1)
middle_wgap = int(img_w/2 - middle_spot)
middle_hgap = int(img_h - cross_y)
# 중심 좌표 찾기
if middle_wgap < 0:
    middle_wgap = -middle_wgap
    cv2.line(img_copy, (int(middle_spot), cross_y), (int(img_w / 2), cross_y), (0,0,0), 3)
    cur_theta = (180/np.pi) *np.arctan(middle_wgap/ middle_hgap)
    cur_theta =  cur_theta
    print("middle_wgap < 0: {}".format(cur_theta))
elif middle_wgap == 0:
    cur_theta = 0
else:
    cv2.line(img_copy, (int(middle_spot), cross_y), (int(img_w / 2), cross_y), (255, 255, 255), 3)
    cur_theta = (180/np.pi) *np.arctan(middle_wgap/ middle_hgap)
    cur_theta = -1 * cur_theta
    print("target_theta: {}".format(cur_theta))



"""
lines = houghLines.reshape(houghLines.shape[0]*2, 2)
h, w = img.shape[:2]

# 최고의 성능 0, 0.01, 0.01

fline = cv2.fitLine(lines, cv2.DIST_L2, 0, 0.01, 0.01)
vx, vy, x, y = fline[0], fline[1], fline[2], fline[3]
x1, y1 = int(((h-1)-y) / vy * vx + x), h-1
x2, y2 = int(((h/2 + 70)-y) / vy * vx + x), int(h/2 + 70)
result = [x1, y1, x2, y2, x, y]
print(result)
cv2.line(img_copy, (x1, y1), (x2, y2), (255, 0, 0) , 5)
"""

# 출력
img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
imgs = {'Original': img_copy, 'ROI_edge':ROI_image} # , 'Canny': canny

for i, (key,value) in enumerate(imgs.items()):
    plt.subplot(1,2,i+1)
    plt.title(key)
    plt.imshow(value, cmap='gray')
    plt.xticks([]); plt.yticks([])

plt.show()