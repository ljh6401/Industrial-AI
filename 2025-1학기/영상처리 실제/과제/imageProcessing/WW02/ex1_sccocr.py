import cv2 as cv
import sys

img = cv.imread("../imgSet/soccer.jpg")

if img is None:
    sys.exit("Could not read the image")

cv.imshow('Image', img)

cv.waitKey()
cv.destroyAllWindows()