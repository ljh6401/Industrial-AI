import cv2 as cv
import numpy as np
import sys

cap = cv.VideoCapture(0, cv.CAP_DSHOW) # 카메라 연결 시도

if not cap.isOpened():
    sys.exit("카메라 연결 실패")

frames=[]
while True:
    ret, frame = cap.read() # 비디오를 구성하는 프레임 획득

    if not ret:
        print("프레임 획득 실패시 루프 탈출")
        break

    cv.imshow('Video display', frame)

    key = cv.waitKey(1)
    if key == ord('c'):         # c를 누르면 프레임을 리스트에 추가
        frames.append(frame)
    if key == ord('q'):         # q를 누르면 종료
        break
q
cap.release()
cv.destroyAllWindows()

if len(frames) > 0: # 수집된 영상이 존재할 경우
    imgs=frames[0]
    for i in range(1, min(3, len(frames))): # 최대 3개까지 붙임
        imgs = np.hstack((imgs, frames[i]))

cv.imshow('Video display', imgs)

cv.waitKey()
cv.destroyAllWindows()