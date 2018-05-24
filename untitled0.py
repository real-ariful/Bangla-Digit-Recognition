# Image Processing

import numpy as np
import cv2

img = cv2.imread('real.jpg', cv2.IMREAD_COLOR)

img[55,55] = [255,255,255]
px = img[55,55]
print(px)

#Region of Image
roi = img[100:150, 100:150]
print(roi)
img[100:150, 100:150] =[255, 255,255]

watch_face = img[100:200, 100:200]
img[0:100, 0:100] = watch_face

cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

