import cv2 as cv
import numpy as np

img = cv.imread('../imgSet/soccer.jpg')
img = cv.resize(img, dsize=(0, 0), fx=0.4, fy=0.4)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.putText(gray, 'soccer', (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
cv.imshow('Original', gray)

# 가우시안 스무딩
smooth = np.hstack((
    cv.GaussianBlur(gray, (5, 5), 0.0),
    cv.GaussianBlur(gray, (9, 9), 0.0),
    cv.GaussianBlur(gray, (15, 15), 0.0)
))
cv.imshow('Smooth', smooth)

# 엠보싱 커널 정의
femboss = np.array([
    [-1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0]
])

gray16 = np.int16(gray)
emboss = np.uint8(np.clip(cv.filter2D(gray16, -1, femboss) + 128, 0, 255))  # 정상 변환
emboss_bad = np.uint8(cv.filter2D(gray16, -1, femboss) + 128)               # 클립 미적용
emboss_worse = cv.filter2D(gray, -1, femboss)                               # 타입 변환 없이 적용

cv.imshow('Emboss', emboss)
cv.imshow('Emboss_bad', emboss_bad)
cv.imshow('Emboss_worse', emboss_worse)

cv.waitKey()
cv.destroyAllWindows()
