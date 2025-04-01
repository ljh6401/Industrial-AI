import skimage
import numpy as np
import cv2 as cv
import time

coffee = skimage.data.coffee()

start = time.time()
slic = skimage.segmentation.slic(coffee, compactness=20, n_segments=600, start_label=1)
g = skimage.future.graph.rag_mean_color(coffee, slic, mode='similarity')
ncut = skimage.future.graph.cut_normalized(slic, g)

print(coffee.shape, '분할 시간:', time.time() - start, '초')

# 결과 시각화
marking = skimage.segmentation.mark_boundaries(coffee, ncut)
ncut_coffee = np.uint8(marking * 255)

cv.imshow('Normalized cut', cv.cvtColor(ncut_coffee, cv.COLOR_RGB2BGR))
cv.waitKey(0)
cv.destroyAllWindows()
