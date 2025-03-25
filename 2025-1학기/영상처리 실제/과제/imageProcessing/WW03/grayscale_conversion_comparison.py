import cv2 as cv
import numpy as np
import time

# 영상처리 성능평가할때 유용한 코드
# 첫 번째 방법: for문을 사용한 직접 변환
def my_cvtGray1(bgr_img):
    g = np.zeros((bgr_img.shape[0], bgr_img.shape[1]))
    for r in range(bgr_img.shape[0]):
        for c in range(bgr_img.shape[1]):
            g[r, c] = 0.114 * bgr_img[r, c, 0] + 0.587 * bgr_img[r, c, 1] + 0.299 * bgr_img[r, c, 2]
    return np.uint8(g)

# 두 번째 방법: NumPy 벡터 연산을 활용한 변환
def my_cvtGray2(bgr_img):
    g = np.zeros((bgr_img.shape[0], bgr_img.shape[1]))
    g = 0.114 * bgr_img[:, :, 0] + 0.587 * bgr_img[:, :, 1] + 0.299 * bgr_img[:, :, 2]
    return np.uint8(g)

# 이미지 불러오기
img = cv.imread('../imgSet/girl_laughing.jpg')

# 첫 번째 방법 실행 시간 측정
start = time.time()
my_cvtGray1(img)
print('My time1:', time.time() - start)  # for문을 사용한 변환 시간

# 두 번째 방법 실행 시간 측정
start = time.time()
my_cvtGray2(img)
print('My time2:', time.time() - start)  # NumPy 연산을 활용한 변환 시간

# OpenCV 내장 함수 실행 시간 측정
start = time.time()
cv.cvtColor(img, cv.COLOR_BGR2GRAY)
print('OpenCV time:', time.time() - start)  # OpenCV 변환 시간
