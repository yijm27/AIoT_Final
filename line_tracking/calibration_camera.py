import numpy as np
import cv2
import matplotlib.image as mpimg
import glob

# Read in and make a list of calibration images
path = "C:/MyWorkspace/tensorflow/Advanced-lane-finding-master/camera_cal"
images = glob.glob(path + '/calibration*.jpg')

# Array to store object points and image points from all the images

objpoints = []  # 3D points in real world space
imgpoints = []  # 2D points in image plane

def calibration():
    """오픈 CV 도큐먼트"""
    # Prepare object points
    objp = np.zeros((6 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)  # x,y coordinates

    for fname in images:
        print("실행")
        img = cv2.imread(fname)
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
        # If corners are found, add object points, image points
        if ret == True:
            imgpoints.append(corners)
            objpoints.append(objp)
        else:
            continue
    # mtx-메트릭스, dist - 왜곡계수
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    return mtx, dist

def undistort(img, mtx, dist):
    """ undistort image """
    print("실행")
    return cv2.undistort(img, mtx, dist, None, mtx)
if __name__ == "__main__":
    mtx, dist = calibration()

    img = cv2.imread(images[0])
    cv2.imshow("before", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    img = undistort(img, mtx, dist)
    cv2.imshow("cal", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
