#%% GRAYSCALE image
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('Images/1.png',cv2.IMREAD_GRAYSCALE)
#img = cv2.imread('watchgray.png',cv2.IMREAD_GRAYSCALE)

plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
#plt.plot([200,300,400],[100,200,300],'c', linewidth=5)
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
#cv2.imwrite('watchgray.png',img)

#%%
print(img.shape)
print(img.dtype)

#%%Four Steps
#Negate
def neg(x):
    return (255-x)
def negative(y):
    y[0]= neg(y[0])
    y[1]= neg(y[1])
    y[2]= neg(y[2])
    return y

import cv2
import numpy as np

img = cv2.imread('Images/1.png')
 
neg = negative(img)
imagem = cv2.bitwise_not(img)
#cv2.imshow('image',img)
#cv2.imshow('Scaling',res)
#cv2.imshow('negative',neg)
cv2.imshow('new neg',imagem)
cv2.waitKey(0)
cv2.destroyAllWindows()

#%%Random work from scaling-does not work

def scalu(x):
    maxx = x.max(); lenx =x.shape[0]
    minx = x.min(); leny = x.shape[1]
    for i in range(0,lenx):
        for j in range(0,leny):
            x[i][j] = (x[i][j]-minx)/(maxx-minx)

import cv2
img = cv2.imread('Images/1.png')


#scmat = np.array(0,)
scal = scalu(img)
cv2.imshow('Scaling',scal)
cv2.waitKey(0)
cv2.destroyAllWindows()

#%%
#1. Scaling
res = cv2.resize(imagem,None,fx=1, fy=1, interpolation = cv2.INTER_CUBIC)
#cv2.imshow('image',img)
cv2.imshow('Scaling',res)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(res.shape)
print(res.dtype)

#%%
import cv2
def range01(M):
    ((M-min(M))/(max(M)-min(M)))

img = cv2.imread('Images/1.png')
imagem = cv2.bitwise_not(img)

#scmat = np.array(0,)
scal = range01(img)
cv2.imshow('Scaling',scal)
cv2.waitKey(0)
cv2.destroyAllWindows()
#%%
#2. Threshold to remove background noise

import cv2
import numpy as np

grayscaled = cv2.cvtColor(imagem,cv2.COLOR_BGR2GRAY)
retval, threshold = cv2.threshold(grayscaled, 120, 240, cv2.THRESH_BINARY)
cv2.imshow('original',imagem)
cv2.imshow('threshold',threshold)
cv2.waitKey(0)
cv2.destroyAllWindows()

#%%
#3. Centralize-Testing
import cv2
img = cv2.imread("Images/1.png", 0)

x,y,w,h = cv2.boundingRect(img)


#%%

import numpy as np
im = np.zeros((20, 20))
#Datermine Centre of Mass
from scipy import ndimage
com = ndimage.measurements.center_of_mass(img)
print(com)

#Translation distances in x and y
x_trans = int(img.shape[0]//2 - com[0])
y_trans = int(img.shape[1]//2 - com[1])

#Pad and remove pixels from image to perform translation

if x_trans >0:
    im2 = np.pad(img, ((x_trans,0), (0,0)), mode ='constant')
    im2 = im2[:img.shape[0]-x_trans, :]
else:
    im2 = np.pad(img, ((0, -x_trans), (0,0)), mode ='constant')
    im2 = im2[:img.shape[0]-x_trans, :]
  
if y_trans >0:
    im3 = np.pad(im2, ((0,0), (y_trans,0)), mode ='constant')
    im3 = im3[:, :img.shape[0]-y_trans]
else:
    im3 = np.pad(im2, ((0,0), (0,-y_trans)), mode ='constant')
    im3 = im3[:, -y_trans]

print(ndimage.measurements.center_of_mass(im3))
#%%

#4. Framing

#----------------------------------------------------------------------
#%% Preprocessing of MNIST
import numpy as np
import matplotlib.pyplot as plt

images = np.zeros((4,784))
correct_vals = np.zeros((4,10))

import cv2
#read the image
gray = cv2.imread("Images/3.png", 0)
#----My added lines
#blur = cv2.GaussianBlur(gray,(5,5),0)
#ret3,gray = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#ret,gray = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
#--------
gray = cv2.resize(255-gray, (28,28))

(thresh, gray) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU )
#cv2.imwrite("Images/2-process.png", gray)
plt.imshow(gray)
flatten = gray.flatten() / 255.0

#%%
while np.sum(gray[0]) == 0:
    gray = gray[1:]

while np.sum(gray[:, 0]) == 0:
    gray = np.delete(gray, 0,1)

while np.sum(gray[-1]) == 0:
    gray = gray[:-1]
    
while np.sum(gray[:,-1]) == 0:
    gray = np.delete(gray,-1,-1)

rows, cols = gray.shape
#%%
if rows > cols:
    factor = 26.0/rows
    rows = 26
    cols = int(round(cols*factor))
    gray = cv2.resize(gray, (cols, rows))

else:
    factor = 26.0/cols
    cols =26
    rows = int(round(rows*factor))
    gray = cv2.resize(gray, (cols, rows))
    

#%% Adding padding'
import math
colsPadding =(int(math.ceil((28-cols)/2.0)), int(math.floor((28-cols)/2.0)))

rowsPadding =(int(math.ceil((28-cols)/2.0)), int(math.floor((28-rows)/2.0)))

gray = np.lib.pad(gray, (rowsPadding, colsPadding), 'constant')

shiftx, shifty = getBestShift(gray)
shifted = shift(gray, shiftx, shifty)
gray = shifted

plt.imshow(gray)
#%%

def getBestShift(img):
    cy, cx = ndimage.measurements.center_of_mass(img)
    
    rows, cols = img.shape
    shiftx = np.round(cols/2.0-cx).astype(int)
    shifty = np.round(rows/2.0-cx).astype(int)
    
    return shiftx, shifty

def shift(img, sx, sy):
    rows, cols = img.shape
    M= np.float32([[1,0,sx], [0,1,sy]])
    shifted = cv2.warpAffine(img,M, (cols, rows))
    return shifted