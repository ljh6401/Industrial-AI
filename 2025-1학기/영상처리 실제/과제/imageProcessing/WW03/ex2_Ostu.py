import cv2 as cv
import sys

img = cv.imread('../imgSet/soccer.jpg')

if img is None:
    sys.exit('파일을 찾을 수 없습니다.')

# R 채널에 대해 Otsu 알고리즘을 사용하여 이진화 수행
t, bin_img = cv.threshold(img[:, :, 2], 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
print('오츄 알고리즘이 찾은 최적 임계값 =', t)

cv.imshow('R channel', img[:, :, 2])                   # R 채널 영상
cv.imshow('R channel binarization', bin_img)           # R 채널 이진화 영상

cv.waitKey()
cv.destroyAllWindows()
