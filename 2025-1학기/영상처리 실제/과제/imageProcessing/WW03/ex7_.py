import cv2 as cv

img = cv.imread('../imgSet/rose.jpg')
patch = img[250:350, 170:270, :]

# 원본 이미지에 빨간색 사각형 그리기
img = cv.rectangle(img, (170, 250), (270, 350), (255, 0, 0), 3)

# 다양한 보간법으로 패치 크기 변경
patch1 = cv.resize(patch, dsize=(0, 0), fx=5, fy=5, interpolation=cv.INTER_NEAREST)
patch2 = cv.resize(patch, dsize=(0, 0), fx=5, fy=5, interpolation=cv.INTER_LINEAR)
patch3 = cv.resize(patch, dsize=(0, 0), fx=5, fy=5, interpolation=cv.INTER_CUBIC)

# 결과 출력
cv.imshow('Original', img)
cv.imshow('Resize nearest', patch1)
cv.imshow('Resize bilinear', patch2)
cv.imshow('Resize bicubic', patch3)

cv.waitKey()
cv.destroyAllWindows()
