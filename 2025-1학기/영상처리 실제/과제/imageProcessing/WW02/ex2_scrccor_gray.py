import cv2 as cv
import sys

img = cv.imread("../imgSet/soccer.jpg")

if img is None:
    sys.exit("Could not read the image")

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray_small=cv.resize(gray, dsize=(0,0), fx=0.5, fy=0.5)

cv.imwrite("../imgSet/soccer_gray.jpg", gray)
cv.imwrite("../imgSet/soccer_gray_small.jpg", gray_small)


cv.imshow('Color Image', img)

cv.imshow('Gray Image', gray)

cv.imshow('Gray Small Image', gray_small)

cv.waitKey()
cv.destroyAllWindows()