import cv2 as cv
import numpy as np
import time

# 모델 영상 일부 크롭 (버스 영역)
img1 = cv.imread('../imgSet/mot_color70.jpg')[190:350, 440:560]
gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)

# 장면 영상 전체
img2 = cv.imread('../imgSet/mot_color83.jpg')
gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

# SIFT 특징점 검출
sift = cv.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

# FLANN 매칭 객체 생성 및 KNN 매칭 (최근접 2개)
start = time.time()
flann_matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
knn_match = flann_matcher.knnMatch(des1, des2, 2)

# 좋은 매칭 선별 (ratio test, 최근접 거리 비율 비교)
T = 0.7
good_match = []
for nearest1, nearest2 in knn_match:
    if nearest1.distance / nearest2.distance < T:
        good_match.append(nearest1)

print('매칭에 걸린 시간:', time.time() - start)

# 매칭 결과 그리기
img_match = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], 3), dtype=np.uint8)
cv.drawMatches(img1, kp1, img2, kp2, good_match, img_match, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# 출력
cv.imshow('Good Matches', img_match)
cv.waitKey()
cv.destroyAllWindows()
