import os
import sys
import cv2
import threading
import time
#%% 프로젝트 폴더를 sys.path에 추가(Jetson Nano에서 직접 사용하기 위해)
project_path = "/home/jetson/MyWorkspace/jetson_inference"
sys.path.append(project_path)
# 영상정보 탐지와 네트워킹 센싱 등 여러가지 동시에 해야하기에 스레드 사용 -> 미리 객체 감지를 스레드 처리되도록 TrtThread를 만들어 놓음
# 우리는 추론에 쓰레딩 코드 작성 없어도 쓰레드 처리됨
from utils.trt_ssd_object_detect import TrtThread, BBoxVisualization
from utils.coco_label_map import CLASSES_DICT

#%% 감지 결과 활용(처리)
def handleDetectedObject(trtThread, condition):
    # 전체 스크린 플래그 변수
    full_scrn = False

    # 초당 프레임 수
    fps = 0.0

    # 시작 시간
    tic = time.time()

    # 바운딩 박스 시각화 객체
    vis = BBoxVisualization(CLASSES_DICT)

    # TrtThread가 실행 중 일때 반복 실행
    while trtThread.running:
        # 모든 활용 스레드가 동기화 데이터를 추론했다면 다음 notify를 실행하도록 하기 위해 with
        # with 가 실행될 동안 trt는 새로운 감지를 하지 않는다.

        with condition:
            # 감지 결과가 있을 때까지 대기

            condition.wait()
            # 감지 결과 얻기 (객체 확률, 레이블 번호)
            img, boxes, confs, clss = trtThread.getDetectResult()

            # 객체 감지 결과 출력
        # 감지 결과 출력
        img = vis.drawBboxes(img, boxes, confs, clss)
        # 초당 프레임 수 드로잉
        img = vis.drawFps(img, fps)
        # 이미지를 윈도우에 보여주기
        cv2.imshow("detect_from_video", img)
        # 초당 프레임 수 계산
        toc = time.time()
        curr_fps = 1.0 / (toc-tic)
        # fps 의 흐름을 보여주기 위해
        fps = curr_fps if fps ==0.0 else (fps*0.95 + curr_fps*0.05)
        tic = toc

        # 키보드 입력을 위해 1ms 동안 대기, 입력이 없으면 -1을 리턴
        key = cv2.waitKey(1)
        if key == 27:
            # esc
            break
        # ord 는 아스키 코드로 변환
        elif key == ord("F") or key == ord("f"):
            # F 나 f를 눌렸을 경우 전체 스크린 토글 기능
            full_scrn = not full_scrn
            if full_scrn:
                cv2.setWindowProperty("detect_from_video", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            else:
                cv2.setWindowProperty("detect_from_video", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
#%% 메인 함수
# 추론 작업(객체 감지, TrtThread가 담당, 입력이미지를 지속적으로 받아서 추론 결과를 main에서 이용, 생성자 스레드) + 
# 응용 및 활용(자율 주행, 네트워크 전송 -> 메인 스레드가 담당, 소비자 스레드) -> 동기화 동기화(sychronize)
def main():
    #엔진 파일 경로
    enginePath = project_path + "/models/ssd_mobilenet_v2_coco_2018_03_29/tensorrt_fp16.engine"
    #입력 이미지 얻기
    videoCapture = cv2.VideoCapture(project_path + "/resources/video/challenge_video.mp4")
    # 감지 결과(생산)와 처리(소비)를 동기화를 위한 Condition 얻기 -> trt 스레드에서 객체를 인지하면 notify->
    # 메인은 wait상태에서 notify 보고 스레드 실행
    condition = threading.Condition()
    # TrtThread 객체 생성
    trtThread = TrtThread(enginePath, TrtThread.INPUT_TYPE_VIDEO, videoCapture, 0.7, condition)
    # 감지 시작
    trtThread.start()
    # 이름있는 윈도우 만들기
    cv2.namedWindow("detect_from_video", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("detect_from_video", 240, 180)
    cv2.setWindowTitle("detect_from_video", "detect_from_video")
    print(condition)
    print(0)
    # 감지 결과 처리(활용)
    handleDetectedObject(trtThread, condition)
    print(1)
    # 감지 중지
    trtThread.stop()
    # 캡쳐 중지
    videoCapture.release()
    cv2.destroyAllWindows()
#%% 최상위 스크립트 실행

if __name__ == "__main__":
    main()
