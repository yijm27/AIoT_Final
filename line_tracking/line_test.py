import numpy as np
import cv2

def img_show_test(img):
    cv2.imshow("test", img)
    cv2.waitKey(3000)
    cv2.destroyAllWindows()
path = "C:/MyWorkspace/tensorflow/Advanced-lane-finding-master/test_images/test3.jpg"
img = cv2.imread(path)
h, w = img.shape[:2]
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray,(3,3),0)
edge = cv2.Canny(blur,70, 210)

# img_show_test(edge)
vertices = np.array(
    [
        [
            (50, h),
            (w / 2 - 50, h / 2 + 70),
            (w / 2 + 75, h / 2 + 70),
            (w - 50, h)
        ]
    ], dtype=np.int32

)
mask = np.zeros_like(edge)

#%%
cv2.fillPoly(mask, vertices, (255,255,255))
# img_show_test(mask)
roi = cv2.bitwise_and(edge, mask)

img_show_test(roi)
#%%
# lines = cv2.HoughLinesP(roi, 1, 1 * np.pi / 180, 30, minLineLength=10, maxLineGap=30)

def draw_line(img, lines, color=[0,0,255], thickness=3):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_trans(img, r, theta, threshold, min_len, max_gap):
    lines = cv2.HoughLinesP(
        img, r, theta, threshold,
        minLineLength=min_len,
        maxLineGap=max_gap
    )
    line_img = np.zeros(
        (img.shape[0], img.shape[1], 3), dtype=np.uint8
    )
    draw_line(line_img, lines)

    return line_img, lines
#%%
img_hough, lines = hough_trans(
    img=roi,
    r=1, theta=1*np.pi /180,
    threshold=30, min_len=10, max_gap=20
)

line_cnt = np.squeeze(lines)
#%%
img_show_test(img_hough)
#%% 기울기 세팅 -> 기울기 양수가 왼쪽, 기울기 음수가 오른쪽
x1, y1, x2, y2 = line_cnt[:,0], line_cnt[:,1], line_cnt[:,2], line_cnt[:,3]
slope_degree = (np.arctan2(y1-y2, x1-x2) * 180) / np.pi
#%% 160 보다 큰거 자르기
line_limit_160 = line_cnt[np.abs(slope_degree)<160]
slope_degree_under_160 = slope_degree[np.abs(slope_degree)<160]
#%% 95도 수직이상 자르기
line_limit_95_160 = line_limit_160[np.abs(slope_degree_under_160) > 95]
slop_degree_95_160 = slope_degree_under_160[np.abs(slope_degree_under_160) > 95]
#%%
temp_img = np.zeros_like(img)
def draw(img, lines):
    for line in lines:
        cv2.line(img, (line[0], line[1]), (line[2], line[3]), [255,255,255], 3)
draw(temp_img, line_limit_95_160)
img_show_test(temp_img)
#%% 대표직선 추출 -> fit라인 포인트 좌표 배열 입력
l_line, r_line = line_limit_95_160[(slop_degree_95_160>0),:], line_limit_95_160[(slop_degree_95_160<0), :]
#%%
def fitline(img, lines):
    lines = lines.reshape(lines.shape[0]*2, 2)
    h, w = img.shape[:2]
    # 최고의 성능 0, 0.01, 0.01
    fline = cv2.fitLine(lines, cv2.DIST_L2, 0, 0.01, 0.01)
    vx, vy, x, y = fline[0], fline[1], fline[2], fline[3]
    x1, y1 = int(((h-1)-y) / vy * vx + x), h-1
    x2, y2 = int(((h/2 + 70)-y) / vy * vx + x), int(h/2 + 70)
    result = [x1, y1, x2, y2, x, y]
    return result
#%%
l_fit_line = fitline(temp_img, l_line)
r_fit_line = fitline(temp_img, r_line)

x_r, y_r = l_fit_line[-2:]
x_l, y_l = r_fit_line[-2:]
#%%
temp = np.zeros_like(temp_img)
def draw_fit_line(img, lines):
    cv2.line(img, (lines[0], lines[1]), (lines[2], lines[3]), color=(255, 0, 0), thickness=7)
draw_fit_line(temp, l_fit_line)
draw_fit_line(temp, r_fit_line)

cv2.drawMarker(temp, (r_fit_line[0], r_fit_line[1]), [255, 255, 255], markerSize=20)
cv2.drawMarker(temp, (r_fit_line[2], r_fit_line[3]), [255, 255, 255], markerSize=20)
cv2.drawMarker(temp, (l_fit_line[0], l_fit_line[1]), [255, 255, 255], markerSize=20)
cv2.drawMarker(temp, (l_fit_line[2], l_fit_line[3]), [255, 255, 255], markerSize=20)
cv2.drawMarker(temp, (x_r, y_r), [255,255,255], markerSize=20)
cv2.drawMarker(temp, (x_l, y_l), [255, 255, 255], markerSize=20)
img_show_test(temp)
#%%

