#Resizing*******
#______________________________________________________________
# Importing necessary libraries
import numpy as np
import pandas as pd
import os
import glob
import cv2 # the opencv library
import matplotlib.pyplot as plt

FIG_WIDTH=16 # Width of figure
HEIGHT_PER_ROW=3 # Height of each row when showing a figure which consists of multiple rows
project_dir='D:/Python Course_buet/Final Project/Numta_Workshop - Copy/Numta_Workshop/testing-f/'
#os.listdir(os.path.join(project_dir,'Numta_Workshop/'))
paths_test_f=glob.glob(os.path.join(project_dir,'*.png'))
print('length:',len(paths_test_f))


#%%
ct=1
#paths_train = paths_test_f
#os.mkdir(os.path.join(project_dir,'candc/'))
#x= imagePath.split('/')
for imagePath in paths_train:
    
    img=cv2.imread(imagePath,cv2.IMREAD_GRAYSCALE) # read image, image size is 180x180
    
    resize_img = cv2.resize(img  , (28 , 28))
    #plt.imshow(resize_img, cmap='gray')
    #plt.show()
        
    
    imageDir2 = os.path.join(project_dir,'candc/', imagePath.split('\\')[1])
    cv2.imwrite(imageDir2,resize_img)
    ct += 1
    print(ct)
    if ct == len(paths_train):
        print("Complete")
        break

#%%3. Centralizing *******     #__Testing-f images
# Converting all images to .png
#!/usr/bin/env python
from glob import glob                                                           
import cv2 
project_dir='D:/Python Course_buet/Final Project/Numta_Workshop - Copy/Numta_Workshop/testing-f/'

jpgs = glob(project_dir+'*.jpg')

for j in range(0,len(jpgs)):
    img = cv2.imread(jpgs[j])
    cv2.imwrite(jpgs[j][:-3] + 'png', img)
    os.remove(jpgs[j])
    
  #%%____________________________________________________________
# Importing necessary libraries
import numpy as np
import pandas as pd
import os
import glob
import cv2 # the opencv library
import matplotlib.pyplot as plt

FIG_WIDTH=16 # Width of figure
HEIGHT_PER_ROW=3 # Height of each row when showing a figure which consists of multiple rows

image = 'f00056.png'
i=9

project_dir='D:/Python Course_buet/Final Project/Numta_Workshop - Copy/Numta_Workshop/testing-f/'
#os.listdir(os.path.join(project_dir,'Numta_Workshop/'))
paths_test_f=glob.glob(os.path.join(project_dir,'*.png'))
print('length:',len(paths_test_f))

img = cv2.imread(project_dir+image)
plt.imshow(img, cmap ='gray'); plt.show()


#%%
#ct=1
#paths_train = paths_test_f
#os.mkdir(os.path.join(project_dir,'candc/'))


#for imagePath in paths_train:
 #%% Otsu’s Binarization
import cv2
import numpy as np
from matplotlib import pyplot as plt
image = 'f00097.png'
plt.title("Original"); plt.imshow(img,'gray'); plt.show()
img = cv2.imread(project_dir+image, 0)


#img = cv2.medianBlur(img,5)
# global thresholding
ret1,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)

# Otsu's thresholding
ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  cv2.THRESH_BINARY,11,2)
# Otsu's thresholding after Gaussian filtering

blur = cv2.GaussianBlur(img,(5,5),0)
ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
ret,thresh3 = cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
ret,thresh4 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO)
plt.imshow(th1,'gray'); plt.show()
plt.imshow(th2,'gray'); plt.show()
plt.title("Otsu");plt.imshow(th3,'gray'); plt.show()
plt.title("tozero");plt.imshow(thresh4,'gray'); plt.show()

#%%
bina = 255-thresh4

plt.imshow(bina,'gray')
plt.show()
img_bin=bina

#%%   
    _,contours,_ = cv2.findContours(img_bin.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        img_gray_cont=img_bin.copy() 
        cv2.drawContours(img_gray_cont, contours, i,255, 1)
        plt.imshow(img_gray_cont,cmap='gray')
        plt.title('Contour: {}'.format(i+1))
        plt.show()  

    if len(contours) > 4:
        kernel = np.ones((5,5),np.uint8)
        img_dilate=cv2.dilate(img_bin,kernel,iterations = 1)
        plt.title("Dilated")
        plt.imshow(img_dilate,cmap='gray')
        plt.show()

        #Reduce Thickness
        img_erode=cv2.erode(img_dilate,kernel,iterations = 1)
        plt.imshow(img_erode,cmap='gray')
        plt.show()
        #_,contours,_ = cv2.findContours(img_erode.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    else:
        img_erode = img_bin
     
    #img_erode = img_dilate
    #closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    #Largest area contour
    
    countours_largest = sorted(contours, key=lambda x: cv2.contourArea(x))[-1]

    bb=cv2.boundingRect(countours_largest)
    #Let's observe the quality of the bounding box.
    # pt1 and pt2 are terminal coordinates of the diagonal of the rectangle
    pt1=(bb[0],bb[1]) # upper coordinates 
    pt2=(bb[0]+bb[2],bb[1]+bb[3]) # lower coordinates
    img_gray_bb=img_erode.copy()
    cv2.rectangle(img_gray_bb,pt1,pt2,0,1)
    plt.title("Eroded")
    plt.imshow(img_gray_bb,cmap='gray')
    plt.show()

#%%--------------------------------------------------   
#    # Padding, make sure contour does not go out of image
#    PIXEL_ADJ=3 # the number of padding pixels
#    pt1=[bb[0],bb[1]]
#    pt2=[bb[0]+bb[2],bb[1]+bb[3]]
#    
#    if pt1[0]-PIXEL_ADJ>=0: # upper x coordinate
#        pt1[0]=pt1[0]-PIXEL_ADJ
#    else: pt1[0]=0
#    if pt1[1]-PIXEL_ADJ>=0: # upper y coordinate
#        pt1[1]=pt1[1]-PIXEL_ADJ
#    else: pt1[1]=0
#    
#    if pt2[0]+PIXEL_ADJ<=img_gray_bb.shape[0]: # lower x coordinate
#        pt2[0]=pt2[0]+PIXEL_ADJ
#    else: pt2[0]=img_gray_bb.shape[0]
#    if pt2[1]+PIXEL_ADJ<=img_gray_bb.shape[1]: # lower y coordinate
#        pt2[1]=pt2[1]+PIXEL_ADJ
#    else: pt2[1]=img_gray_bb.shape[1]
#    pt1=tuple(pt1)
#    pt2=tuple(pt2)
#    
#    img_gray_bb=img_dilate.copy()
#    cv2.rectangle(img_gray_bb,pt1,pt2,255,3)
#    plt.imshow(img_gray_bb,cmap='gray')
#    plt.show()
#%%    
    # Cropping an image
    crop_img = img_erode[pt1[1]:pt2[1], pt1[0]:pt2[0]]
    plt.imshow(crop_img, cmap='gray')
    plt.show()
#%%
   #Resizing to 28 x28
    resize_img = cv2.resize(crop_img  , (28 , 28))
    plt.imshow(resize_img, cmap='gray')
    plt.show()
        
 #%%   
    #os.mkdir(os.path.join(project_dir,'candc/'))
    #x= imagePath.split('/')
    imageDir2 = os.path.join(project_dir,'candc/', image)
    cv2.imwrite(imageDir2,crop_img)
    #ct += 1
    #print(ct)
    #if ct == len(paths_train):
        #print("Complete")
        #break
    i+=1
#%%Resizing to 28 x28
#resize_img = cv2.resize(img  , (28 , 28))
#cv2.imshow('img' , resize_img)
#x = cv2.waitKey(0)
#cv2.destroyWindow('img')
    
#Rotation
#img = cv2.imread('messi5.jpg',0)
rows,cols = bina.shape
 
M = cv2.getRotationMatrix2D((cols/2,rows/2),-15,1)
dst = cv2.warpAffine(bina,M,(cols,rows))

cv2.imshow('img',dst)

cv2.waitKey(0)
cv2.destroyAllWindows()
#%%
plt.imshow(dst[72:218, 254:347],cmap='gray')

