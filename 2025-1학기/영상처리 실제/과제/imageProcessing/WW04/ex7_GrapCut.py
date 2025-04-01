import cv2 as cv
import numpy as np

img = cv.imread('./img/soccer.jpg')
img_show = img.copy()
mask = np.zeros(img.shape[:2], np.uint8)
mask[:] = cv.GC_PR_BGD  # 초기 마스크는 전부 "아마도 배경"

# 마우스 이벤트로 브러시 구현
def painting(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        cv.circle(mask, (x,y), 9, cv.GC_FGD, -1)
    elif event == cv.EVENT_RBUTTONDOWN:
        cv.circle(mask, (x,y), 9, cv.GC_BGD, -1)
    elif event == cv.EVENT_MOUSEMOVE and flags & cv.EVENT_FLAG_LBUTTON:
        cv.circle(mask, (x,y), 9, cv.GC_FGD, -1)
    elif event == cv.EVENT_MOUSEMOVE and flags & cv.EVENT_FLAG_RBUTTON:
        cv.circle(mask, (x,y), 9, cv.GC_BGD, -1)

cv.namedWindow('Painting')
cv.setMouseCallback('Painting', painting)

while True:
    cv.imshow('Painting', img_show)
    if cv.waitKey(1) == ord('q'):
        break

# GrabCut 실행
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)
cv.grabCut(img, mask, None, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_MASK)
mask2 = np.where((mask == cv.GC_BGD) | (mask == cv.GC_PR_BGD), 0, 1).astype('uint8')
grab = img * mask2[:, :, np.newaxis]

cv.imshow('Grab cut image', grab)
cv.waitKey(0)
cv.destroyAllWindows()
