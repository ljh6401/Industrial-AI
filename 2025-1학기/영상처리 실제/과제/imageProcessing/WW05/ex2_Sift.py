import cv2 as cv

# 컬러 이미지 읽기
img = cv.imread('../imgSet/mot_color70.jpg')
# 그레이스케일로 변환
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# SIFT 객체 생성
sift = cv.SIFT_create()

# 특징점과 기술자 검출
kp, des = sift.detectAndCompute(gray, None)

# 특징점을 이미지 위에 그리기
gray = cv.drawKeypoints(gray, kp, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# 결과 영상 출력
cv.imshow('sift', gray)
cv.waitKey()
cv.destroyAllWindows()
