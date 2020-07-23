import cv2
import numpy as np

def Gray_change(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def Otsu_threshold(image):
    return cv2.threshold(image, -1, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU) # return t, otsu_img

def Mean_threshold(image, blk_size, C):
    return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blk_size, C)

def Gaussian_threshold(image, blk_size, C):
    return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,blk_size, C)

def Gaussianblur(image):
    return cv2.GaussianBlur(image, (3,3), 0)

def Canny_edge(image, low_threshold, high_threshold):
    return cv2.Canny(image, low_threshold, high_threshold)

def Detect_white(image):
    mark = np.copy(image) # image 복사

    blue_thr = 120
    green_thr = 120
    red_thr = 120
    bgr_thr = [blue_thr, green_thr, red_thr]

    thresholds = (image[:,:,0] < bgr_thr[0]) \
                | (image[:,:,1] < bgr_thr[1]) \
                | (image[:,:,2] < bgr_thr[2])
    mark[thresholds] = [0,0,0]

    return mark

def ROI_image(image,color3=(255,255,255),color1=255):
    vertices = np.array(
        [[(0, height), (0, height / 2 ), (width/2, height/2),(width, height / 2), (width, height)]],
        dtype=np.int32)
    mask = np.zeros_like(image)  # mask = img와 같은 크기의 빈 이미지
    if len(image.shape) > 2:  # Color 이미지(3채널)라면 :
        color = color3
    else:  # 흑백 이미지(1채널)라면 :
        color = color1

        # vertices에 정한 점들로 이뤄진 다각형부분(ROI 설정부분)을 color로 채움
    cv2.fillPoly(mask, vertices, color)

    # 이미지와 color로 채워진 ROI를 합침
    ROI_image = cv2.bitwise_and(image, mask)
    return ROI_image


if __name__ == "__main__":
################################################### RGB 흰선
    image = cv2.imread("../Img/test1.PNG")
    height, width = image.shape[:2]
    mark_image = Detect_white(image)
    roi_image = ROI_image(mark_image, (0, 0, 255))
################################################### Gray 흰선
    gray = Gray_change(image)

    canny_image = Canny_edge(roi_image, 70, 210)



    cv2.imshow('result', image)
    cv2.waitKey(0)