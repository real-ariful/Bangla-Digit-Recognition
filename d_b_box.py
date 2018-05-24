# Importing necessary libraries
import numpy as np
import pandas as pd
import os
import glob
import cv2 # the opencv library
import matplotlib.pyplot as plt

FIG_WIDTH=16 # Width of figure
HEIGHT_PER_ROW=3 # Height of each row when showing a figure which consists of multiple rows

project_dir='D:/Python Course_buet/Final Project/Numta_Workshop - Copy/'

# All the images have .png extension. We can get all the filepaths that have .png extensions by using the 
# os.glob.glob() function 
os.listdir(os.path.join(project_dir,'Numta_Workshop/'))
# Notice that I am using the os.path.join() function to create the filepaths instead of writing them down explicitly with a 
# filepath separator ('\\' for windows '/' for linux). This allows us to run this notebook both in windows and linux 
# environment without manually changing the filepath separator

paths_train_a=glob.glob(os.path.join(project_dir,'Numta_Workshop/','training-a','*.png'))
paths_train_b=glob.glob(os.path.join(project_dir,'Numta_Workshop/','training-b','*.png'))
paths_train_c=glob.glob(os.path.join(project_dir,'Numta_Workshop/','training-c','*.png'))
paths_train_d=glob.glob(os.path.join(project_dir,'Numta_Workshop/','training-d','*.png'))
paths_train_e=glob.glob(os.path.join(project_dir,'Numta_Workshop/','training-e','*.png'))

paths_test_a=glob.glob(os.path.join(project_dir,'Numta_Workshop/','testing-a','*.png'))
paths_test_b=glob.glob(os.path.join(project_dir,'Numta_Workshop/','testing-b','*.png'))
paths_test_c=glob.glob(os.path.join(project_dir,'Numta_Workshop/','testing-c','*.png'))
paths_test_d=glob.glob(os.path.join(project_dir,'Numta_Workshop/','testing-d','*.png'))
paths_test_e=glob.glob(os.path.join(project_dir,'Numta_Workshop/','testing-e','*.png'))

path_label_train_a=os.path.join(project_dir,'Numta_Workshop/','training-a.csv')
path_label_train_b=os.path.join(project_dir,'Numta_Workshop/','training-b.csv')
path_label_train_c=os.path.join(project_dir,'Numta_Workshop/','training-c.csv')
path_label_train_d=os.path.join(project_dir,'Numta_Workshop/','training-d.csv')
path_label_train_e=os.path.join(project_dir,'Numta_Workshop/','training-e.csv')

#%% Let's read an image from dataset A as grayscale image and plot it.

img_gray=cv2.imread(paths_train_a[0],cv2.IMREAD_GRAYSCALE) # read image, image size is 180x180
plt.imshow(img_gray,cmap='gray')
plt.show()

#%% Inversion
img_gray=255-img_gray
plt.imshow(img_gray,cmap='gray')
plt.show()

#%% Threshold

(thresh, img_bin) = cv2.threshold(img_gray, 128, 255, cv2.THRESH_OTSU)
plt.imshow(img_bin,cmap='gray')
plt.title('Threshold: {}'.format(thresh))
plt.show()

#%% Applying contours

_,contours,_ = cv2.findContours(img_bin.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#Let's draw the contours on the grayscaleimage using the drawContours() function. This function alters the source image. So we are going to pass a copy of the grayscale image each time we want a draw a contour on it.

for i in range(len(contours)):
    img_gray_cont=img_gray.copy() 
    cv2.drawContours(img_gray_cont, contours, i,255, 1)
    plt.imshow(img_gray_cont,cmap='gray')
    plt.title('Contour: {}'.format(i+1))
    plt.show()
    
#%% Largest area contour
    
countours_largest = sorted(contours, key=lambda x: cv2.contourArea(x))[-1]

bb=cv2.boundingRect(countours_largest)
#Let's observe the quality of the bounding box.

# pt1 and pt2 are terminal coordinates of the diagonal of the rectangle
pt1=(bb[0],bb[1]) # upper coordinates 
pt2=(bb[0]+bb[2],bb[1]+bb[3]) # lower coordinates
img_gray_bb=img_gray.copy()
cv2.rectangle(img_gray_bb,pt1,pt2,255,3)
plt.imshow(img_gray_bb,cmap='gray')
plt.show()

#%% Padding, make sure contour does not go out of image

PIXEL_ADJ=3 # the number of padding pixels
pt1=[bb[0],bb[1]]
pt2=[bb[0]+bb[2],bb[1]+bb[3]]

if pt1[0]-PIXEL_ADJ>=0: # upper x coordinate
    pt1[0]=pt1[0]-PIXEL_ADJ
else: pt1[0]=0
if pt1[1]-PIXEL_ADJ>=0: # upper y coordinate
    pt1[1]=pt1[1]-PIXEL_ADJ
else: pt1[1]=0

if pt2[0]+PIXEL_ADJ<=img_gray.shape[0]: # lower x coordinate
    pt2[0]=pt2[0]+PIXEL_ADJ
else: pt2[0]=img_gray.shape[0]
if pt2[1]+PIXEL_ADJ<=img_gray.shape[1]: # lower y coordinate
    pt2[1]=pt2[1]+PIXEL_ADJ
else: pt2[1]=img_gray.shape[1]
pt1=tuple(pt1)
pt2=tuple(pt2)

img_gray_bb=img_gray.copy()
cv2.rectangle(img_gray_bb,pt1,pt2,255,3)
plt.imshow(img_gray_bb,cmap='gray')
plt.show()

#%%% Cropping an image
import cv2

crop_img = img_bin[pt1[1]:pt2[1], pt1[0]:pt2[0]]
plt.imshow(crop_img, cmap='gray')
plt.show()

#%% Resizing the image
#%%Resizing to 28 x28
resize_img = cv2.resize(crop_img  , (28 , 28))
plt.imshow(resize_img, cmap='gray')
plt.show()


#%%  number 7
#---------------------------------------------------------------
img_gray=cv2.imread(paths_train_a[61],cv2.IMREAD_GRAYSCALE) # read image, image size is 180x180
plt.imshow(img_gray,cmap='gray')
plt.show()

#%%
img_gray=255-img_gray
plt.imshow(img_gray,cmap='gray')
plt.show()
#%%
(thresh, img_bin) = cv2.threshold(img_gray, 128, 255, cv2.THRESH_OTSU)
plt.imshow(img_bin,cmap='gray')
plt.title('Threshold: {}'.format(thresh))
plt.show()
#%%

_,contours,_ = cv2.findContours(img_bin.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for i in range(len(contours)):
    img_gray_cont=img_gray.copy() 
    cv2.drawContours(img_gray_cont, contours, i,255, 1)
    plt.imshow(img_gray_cont,cmap='gray')
    plt.title('Contour: {}'.format(i+1))
    plt.show()
    
#%% kernels
kernel = np.ones((5,5),np.uint8)
img_dilate=cv2.dilate(img_bin,kernel,iterations = 1)
plt.imshow(img_dilate,cmap='gray')
plt.show()

#%% we want to erode the edges of the digit and reduce its thicknes

img_erode=cv2.erode(img_dilate,kernel,iterations = 1)
plt.imshow(img_erode,cmap='gray')
plt.show()

#%%

def img_bb(path):
    # shows image with bounding box
    img_gray=cv2.imread(path,cv2.IMREAD_GRAYSCALE) # read image, image size is 180x180
    img_gray=255-img_gray
    
    (thresh, img_bin) = cv2.threshold(img_gray, 128, 255, cv2.THRESH_OTSU)
    
    kernel = np.ones((5,5),np.uint8)
    img_dilate=cv2.dilate(img_bin,kernel,iterations = 1)
    img_erode=cv2.erode(img_dilate,kernel,iterations = 1)
    
    _,contours,_ = cv2.findContours(img_erode.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    countours_largest = sorted(contours, key=lambda x: cv2.contourArea(x))[-1]
    bb=cv2.boundingRect(countours_largest)
    
    pt1=[bb[0],bb[1]]
    pt2=[bb[0]+bb[2],bb[1]+bb[3]]
    # pt1 and pt2 are terminal coordinates of the diagonal of the rectangle
    if pt1[0]-PIXEL_ADJ>=0: # upper x coordinate
        pt1[0]=pt1[0]-PIXEL_ADJ
    else: pt1[0]=0
    if pt1[1]-PIXEL_ADJ>=0: # upper y coordinate
        pt1[1]=pt1[1]-PIXEL_ADJ
    else: pt1[1]=0
    if pt2[0]+PIXEL_ADJ<=img_gray.shape[0]: # lower x coordinate
        pt2[0]=pt2[0]+PIXEL_ADJ
    else: pt2[0]=img_gray.shape[0]
    if pt2[1]+PIXEL_ADJ<=img_gray.shape[0]: # lower y coordinate
        pt2[1]=pt2[1]+PIXEL_ADJ
    else: pt2[1]=img_gray.shape[0]
    pt1=tuple(pt1)
    pt2=tuple(pt2)
    
    img_gray_bb=img_gray.copy()
    cv2.rectangle(img_gray_bb,pt1,pt2,255,3)
    
    return img_gray_bb
    
def imshow_group(X,n_per_row=10):
    # shows a group of images
    n_sample=len(X)
    j=np.ceil(n_sample/n_per_row)
    fig=plt.figure(figsize=(FIG_WIDTH,HEIGHT_PER_ROW*j))
    for i,img in enumerate(X):
        plt.subplot(j,n_per_row,i+1)
        plt.imshow(img,cmap='gray')
        plt.axis('off')
    plt.show()
    
#%%
    
X_sample_a=[img_bb(path) for path in paths_train_a[:100]]
imshow_group(X_sample_a)

#%% The results are quite satisfactory. Let's see if the above procedure works for dataset B.
paths_train_b=glob.glob(os.path.join(project_dir,'Final_DB','training-b','*.png'))

X_sample_b=np.array([img_bb(path) for path in paths_train_b[:100]])
imshow_group(X_sample_b)

#%% Dataset c

X_sample_c=np.array([img_bb(path) for path in paths_train_c[:100]])
imshow_group(X_sample_c)

#%% Dataset d

X_sample_d=np.array([img_bb(path) for path in paths_train_d[:10]])
imshow_group(X_sample_d)

#%%