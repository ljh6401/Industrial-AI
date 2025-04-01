import cv2 as cv
import numpy as np

img = cv.imread('./img/soccer.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
canny = cv.Canny(gray, 100, 200)

# 외곽선 추출 (CHAIN_APPROX_NONE: 모든 윤곽 좌표를 저장)
contour, hierarchy = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

lcontour = []
for i in range(len(contour)):
    if contour[i].shape[0] > 100:  # 길이 필터링
        lcontour.append(contour[i])

# 경계선을 색으로 그림
cv.drawContours(img, lcontour, -1, (0, 255, 0), 3)

cv.imshow('Original with contours', img)
cv.imshow('Canny', canny)
cv.waitKey(0)
cv.destroyAllWindows()
