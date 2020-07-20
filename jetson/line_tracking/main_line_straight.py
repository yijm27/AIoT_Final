import cv2  # opencv 사용
import numpy as np
import sys
project_path = "/home/jetson/MyWorkspace/jetson_inference"
sys.path.append(project_path)
def cvt_gray(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img
def gauassian_blur(img, kernel_size):
    img = cv2.GaussianBlur(img, (kernel_size,kernel_size), 0)
    return img
def canny(img, low_threshold, high_threshold):
    img = cv2.Canny(img, low_threshold, high_threshold)
    return img

def region_of_interest(img, vertices, c3=(255,255,255), c1=255):
    mask = np.zeros_like(img)

    if len(img.shape) > 2:
        color = c3
    else:
        color = c1

    cv2.fillPoly(mask, vertices, color)
    img_ROI = cv2.bitwise_and(img, mask)
    return img_ROI

def draw_lines(img, lines, color = [0, 0, 255], thickness=2):
    # (샘플, 1, (x1, y1, x2, y2))
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def draw_fit_line(img, lines, color=[255, 0, 0], thickness=10):  # 대표선 그리기
    cv2.line(img, (lines[0], lines[1]), (lines[2], lines[3]), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):

    lines = cv2.HoughLinesP(
        img,
        rho, theta,
        threshold,
        minLineLength=min_line_len,
        maxLineGap=max_line_gap
    )

    img_line = np.zeros(
        (img.shape[0], img.shape[1], 3), dtype=np.uint8
    )

    draw_lines(img_line, lines)

    return img_line, lines
def weighted_img(initial_img, img, a=1, b=1., y=0.):
    img = cv2.addWeighted(initial_img, a, img, b, y)
    return img
def fitline(img, f_lines):

    lines = np.squeeze(f_lines)
    lines = lines.reshape(lines.shape[0]*2, 2)
    rows, cols = img.shape[:2]
    output = cv2.fitLine(lines, cv2.DIST_L2, 0, 0.01, 0.01)

    vx, vy, x, y = output[0], output[1], output[2], output[3]
    x1, y1 = int(((img.shape[0] - 1) - y) / vy * vx + x), img.shape[0] - 1
    x2, y2 = int(((img.shape[0] / 2 + 100) - y) / vy * vx + x), int(img.shape[0] / 2 + 100)

    result = [x1, y1, x2, y2, x, y]
    return result


# img_path = "D:/MyWorkspace/datasets/line/slope_test.jpg"
# image = cv2.imread(img_path)
video_path = project_path + "/resources/video/challenge_video.mp4"
video = cv2.VideoCapture(video_path)

while video.isOpened():
    ret, image = video.read()
    if not ret:
        break

    h, w = image.shape[:2]
    img_gray = cvt_gray(image)
    img_blur = gauassian_blur(img_gray, kernel_size=3)
    img_edge = canny(img_blur, low_threshold=70, high_threshold=210)
    vertices = np.array(
        [
            [
                (50, h),
                (w / 2 - 45, h / 2 + 60),
                (w / 2 + 45, h / 2 + 60),
                (w - 50, h)
            ]
        ],
        dtype=np.int32
    )

    img_roi = region_of_interest(img_edge, vertices)

    img_hough_line, lines = hough_lines(
        img=img_roi,
        rho=1,
        theta=1 * np.pi / 180,
        threshold=30, min_line_len=10, max_line_gap=20
    )

    # lines = cv2.HoughLinesP(
    #         img_roi,
    #         1, 1 * np.pi / 180,
    #         30,
    #         minLineLength=10,
    #         maxLineGap=20
    #     )
    # x1, y1, x2, y2 검출
    line_arr = np.squeeze(lines)
    # 기울기 구하기
    x1 = line_arr[:, 0]
    y1 = line_arr[:, 1]
    x2 = line_arr[:, 2]
    y2 = line_arr[:, 3]
    slope_degree = (np.arctan2(y1 - y2, x1 - x2) * 180) / np.pi
    # 수평 기울기 제한
    line_arr = line_arr[np.abs(slope_degree)<160]
    slope_degree = slope_degree[np.abs(slope_degree)<160]
    #수직 기울기 제한
    line_arr = line_arr[np.abs(slope_degree)>95]
    slope_degree = slope_degree[np.abs(slope_degree)>95]
    # 필터링된 직선 버리기
    L_lines, R_lines = line_arr[(slope_degree>0),:], line_arr[(slope_degree<0),:]
    temp = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    L_lines, R_lines = np.expand_dims(L_lines, axis=1), R_lines[:,None]

    left_fit_line = fitline(temp, L_lines)
    right_fit_line = fitline(temp, R_lines)

    x_r, y_r = right_fit_line[-2:]
    x_l, y_l = left_fit_line[-2:]

    draw_fit_line(temp, right_fit_line)
    draw_fit_line(temp, left_fit_line, [255, 0, 0], 7)
    # cv2.line(temp,
    #          (right_fit_line[0], right_fit_line[1]),
    #          (right_fit_line[2], right_fit_line[3]), [255,0,0], 5)
    # cv2.line(temp,
    #          (left_fit_line[0], left_fit_line[1]),
    #          (left_fit_line[2], left_fit_line[3]), [255, 0, 0], 5)
    cv2.drawMarker(temp, (right_fit_line[0], right_fit_line[1]), [255, 255, 255], markerSize=20)
    cv2.drawMarker(temp, (right_fit_line[2], right_fit_line[3]), [255, 255, 255], markerSize=20)
    cv2.drawMarker(temp, (left_fit_line[0], left_fit_line[1]), [255, 255, 255], markerSize=20)
    cv2.drawMarker(temp, (left_fit_line[2], left_fit_line[3]), [255, 255, 255], markerSize=20)
    cv2.drawMarker(temp, (x_r, y_r), [255,255,255], markerSize=20)
    cv2.drawMarker(temp, (x_l, y_l), [255, 255, 255], markerSize=20)
    fps = video.get(cv2.CAP_PROP_FPS)
    cv2.putText(temp, "fps: "+ str(int(fps)), (50,50), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=1.0, color=(0,0,255), thickness=3)

    # # %%직선 그리기
    # draw_lines(temp, L_lines)
    # draw_lines(temp, R_lines)

    result = weighted_img(temp, image) # 원본 이미지에 검출된 선 overlap
    cv2.imshow('result', result) # 결과 이미지 출력

    if cv2.waitKey(int(1000/fps)) == 27:
        break
video.release()
cv2.destroyAllWindows()
#%%
# img_result = weighted_img(line_arr, image)
# img_result = cv2.resize(img_result,dsize=None,fx=0.5, fy=0.5)
# img_line = np.zeros(
#         (image.shape[0], image.shape[1], 3), dtype=np.uint8
#     )
#
# for point in line_arr:
#     cv2.line(img_line, (point[0],point[1]), (point[2], point[3]), [255,255,255], thickness=3)
# cv2.imshow('result', img_line)
# cv2.waitKey(0)
