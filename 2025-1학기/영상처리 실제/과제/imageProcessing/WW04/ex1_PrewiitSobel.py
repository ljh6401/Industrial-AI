import cv2 as cv

img = cv.imread('./img/soccer.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# 1차 미분 (Sobel 필터 적용)
grad_x = cv.Sobel(gray, cv.CV_32F, 1, 0, ksize=3)
grad_y = cv.Sobel(gray, cv.CV_32F, 0, 1, ksize=3)

# 절댓값 후 8비트로 변환 (음수 제거, 스케일 조정)
sobel_x = cv.convertScaleAbs(grad_x)
sobel_y = cv.convertScaleAbs(grad_y)

# 에지 강도 계산: 두 방향 결과를 가중합으로 병합
edge_strength = cv.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)

cv.imshow('Original', gray)
cv.imshow('sobel x', sobel_x)
cv.imshow('sobel y', sobel_y)
cv.imshow('edge strength', edge_strength)

cv.waitKey(0)
cv.destroyAllWindows()
