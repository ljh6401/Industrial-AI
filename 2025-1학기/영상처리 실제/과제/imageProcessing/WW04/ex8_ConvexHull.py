import skimage
import numpy as np
import cv2 as cv

orig = skimage.data.horse()
img = 255 - np.uint8(orig) * 255
cv.imshow('Horse', img)

# 외곽선 추출
contours, hierarchy = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
img2 = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
cv.drawContours(img2, contours, -1, (255, 0, 255), 2)
cv.imshow('Horse with contour', img2)

# 모멘트 및 중심좌표
contour = contours[0]
m = cv.moments(contour)
area = cv.contourArea(contour)
cx = m['m10'] / m['m00']
cy = m['m01'] / m['m00']
perimeter = cv.arcLength(contour, True)
roundness = (4 * np.pi * area) / (perimeter * perimeter)

print('면적:', area, '\n중심:', cx, cy, '\n둘레:', perimeter, '\n둥근 정도:', roundness)

# 다각형 근사
img3 = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
approx = cv.approxPolyDP(contour, 8, True)
cv.drawContours(img3, [approx], -1, (0, 255, 0), 2)

# 블록 헐 (Convex Hull)
hull = cv.convexHull(contour)
cv.drawContours(img3, [hull], -1, (0, 0, 255), 2)

cv.imshow('Horse with line segments and convex hull', img3)
cv.waitKey(0)
cv.destroyAllWindows()
