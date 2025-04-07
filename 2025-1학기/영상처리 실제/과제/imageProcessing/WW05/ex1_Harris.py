import cv2 as cv
import numpy as np

# 입력 영상: 9x9의 간단한 패턴 배열 정의
img = np.array([
    [0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0],
    [0,0,1,0,0,0,0,0,0],
    [0,0,1,1,0,0,0,0,0],
    [0,0,1,1,1,0,0,0,0],
    [0,0,1,1,1,1,0,0,0],
    [0,0,1,1,1,1,1,0,0],
    [0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0]
], dtype=np.float32)

# x, y 방향 필터 정의 (Sobel 유사)
ux = np.array([[-1, 0, 1]])
uy = np.array([-1, 0, 1]).transpose()

# 3x3 가우시안 커널 생성 및 외적 연산으로 2D 커널 생성
k = cv.getGaussianKernel(3, 1)
g = np.outer(k, k.transpose())

# x, y 방향 미분
dy = cv.filter2D(img, cv.CV_32F, uy)
dx = cv.filter2D(img, cv.CV_32F, ux)

# 도함수의 제곱 및 곱
dyy = dy * dy
dxx = dx * dx
dyx = dy * dx

# 가우시안 필터로 평활화 (노이즈 제거)
gdyy = cv.filter2D(dyy, cv.CV_32F, g)
gdxx = cv.filter2D(dxx, cv.CV_32F, g)
gdyx = cv.filter2D(dyx, cv.CV_32F, g)

# 해리스 코너 응답값 계산 (특징 가능성)
C = (gdyy * gdxx - gdyx * gdyx) - 0.04 * (gdyy + gdxx) * (gdyy + gdxx)

# 비최대 억제를 통한 특징점 추출
for j in range(1, C.shape[0]-1):
    for i in range(1, C.shape[1]-1):
        if C[j, i] > 0.1 and np.sum(C[j, i] > C[j-1:j+2, i-1:i+2]) == 8:
            img[j, i] = 9  # 특징점 위치에 9로 표시

np.set_printoptions(precision=2)
print(dy)
print(dx)
print(dyy)
print(dxx)
print(dyx)
print(gdyy)
print(gdxx)
print(gdyx)
print(C)
print(img)


# 확대된 영상 만들기 (16배로)
popping = np.zeros([160,160], np.uint8)
for j in range(0, 160):
    for i in range(0, 160):
        popping[j, i] = np.uint8((C[j//16, i//16] + 0.06) * 700)

# 결과 영상 출력
cv.imshow('Image Display2', popping)
cv.waitKey()
cv.destroyAllWindows()
