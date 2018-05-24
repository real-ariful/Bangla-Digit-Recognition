#Data Prepossing
#Gaussian Adaptive threshold
#%%
import cv2
import numpy as np

img = cv2.imread('spring2018_2.jpg')
retval, threshold = cv2.threshold(img, 140, 255, cv2.THRESH_BINARY)
cv2.imshow('original',img)
cv2.imshow('threshold',threshold)
cv2.waitKey(0)
cv2.destroyAllWindows()

#%%

#Gray scale
import cv2
import numpy as np

grayscaled = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
retval, threshold = cv2.threshold(grayscaled, 140, 255, cv2.THRESH_BINARY)
cv2.imshow('original',img)
cv2.imshow('threshold',threshold)
cv2.waitKey(0)
cv2.destroyAllWindows()

#%% adaptive thresholding
import cv2
import numpy as np

th = cv2.adaptiveThreshold(grayscaled, 220, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
cv2.imshow('original',img)
cv2.imshow('Adaptive threshold',th)
cv2.waitKey(0)
cv2.destroyAllWindows()

#%% Otsu's Threshold
retval2,threshold2 = cv2.threshold(grayscaled,120,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imshow('original',img)
cv2.imshow('Otsu threshold',threshold2)
cv2.waitKey(0)
cv2.destroyAllWindows()

#%%
