#My ideas
#Find the best preprocessing techniques
#Comparing all, it seems Otsu's Thresholding along with Gaussian Filtered 
#Image can be very helpful for the pre-processing of images
#Preprocessing Steps

#%% #1. Otsu's Thresholding with Gaussian Filtered
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('3.png',0)

# Otsu's thresholding after Gaussian filtering
blur = cv2.GaussianBlur(img,(5,5),0)
ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# plot all the images and their histograms
#plt.hist(img.ravel(),256,[0,256]); plt.show()
plt.imshow(th3,'gray')
plt.title('Otsu’s Binarization')

#%%Opening all images 
# import the necessary packages
import cv2
import numpy as np
from matplotlib import pyplot as plt
import cv2
import os, os.path
 
#image path and valid extensions
imageDir = "D:/Python Course_buet/Final Project/Numta_Workshop - Copy/Numta_Workshop/training-a/" #specify your path here
image_path_list = []
valid_image_extensions = [".jpg", ".jpeg", ".png", ".tif", ".tiff"] 
valid_image_extensions = [item.lower() for item in valid_image_extensions]
 
#create a list all files in directory and
#append files with a vaild extention to image_path_list
for file in os.listdir(imageDir):
    extension = os.path.splitext(file)[1]
    if extension.lower() != '.db':
        if extension.lower() not in valid_image_extensions:
            continue
        image_path_list.append(os.path.join(imageDir, file))
#%% not working,need to solve it
#loop through image_path_list to open each image
i=1
for imagePath in image_path_list:
    image = cv2.imread(imagePath, 0)
    
    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(image,(5,5),0)
    ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # plot all the images and their histograms
    #plt.hist(img.ravel(),256,[0,256]); plt.show()
    plt.imshow(th3,'gray')
    plt.title('Otsu’s Binarization')
    print(imagePath)
    #cv2.imwrite(imagePath)
    
    if image is not None:
        #cv2.imshow(imagePath, image)
        print("Completed: ", i)
    elif image is None:
        print ("Error loading: " + imagePath)
        #end this loop iteration and move on to next image
        continue
    i += 1
        #key = cv2.waitKey(0)
    #if key == 27: # escape
        #break
    x= imagePath.split('/')
    imageDir2 = os.path.join(imageDir,'otsu/', x[6])
    #cv2.imwrite(imageDir2,th3)
    
# close any open windows
#cv2.destroyAllWindows()


#%% 2. Binary or negative-BINARY_INV
 
imageDir = "D:/Python Course_buet/Final Project/Numta_Workshop - Copy/Numta_Workshop/training-a/" #specify your path here    
imageDir2 = "D:/Python Course_buet/Final Project/Numta_Workshop - Copy/Numta_Workshop/training-a/otsu/" #specify your path here
image_path_list = []
valid_image_extensions = [".jpg", ".jpeg", ".png", ".tif", ".tiff"] 
valid_image_extensions = [item.lower() for item in valid_image_extensions]
 
#create a list all files in directory and
#append files with a vaild extention to image_path_list
for file in os.listdir(imageDir2):
    extension = os.path.splitext(file)[1]
    if extension.lower() != '.db':
        if extension.lower() not in valid_image_extensions:
            continue
        image_path_list.append(os.path.join(imageDir2, file))

i=1
for imagePath in image_path_list:
    image = cv2.imread(imagePath)
    
    #Binary Inversion
    ret,thresh2 = cv2.threshold(image,127,255,cv2.THRESH_BINARY_INV)
    
    # plot all the images and their histograms
    #plt.hist(img.ravel(),256,[0,256]); plt.show()
    plt.imshow(thresh2,'gray')
    plt.title('Binary Inversion')
    #print(imagePath)
    #cv2.imwrite(imagePath)
    
    if image is not None:
        #cv2.imshow(imagePath, image)
        print("Completed: ", i)
    elif image is None:
        print ("Error loading: " + imagePath)
        #end this loop iteration and move on to next image
        continue
    i += 1
        #key = cv2.waitKey(0)
    #if key == 27: # escape
        #break
    
    x= imagePath.split('/')
    imageDir3 = os.path.join(imageDir,'bin_inv/', x[7])
    cv2.imwrite(imageDir3,thresh2)
    if i == 19703:
        break
# close any open windows
#cv2.destroyAllWindows()

#%%
    
import cv2
import numpy as np
from matplotlib import pyplot as plt

path = 'D:/Python Course_buet/Final Project/Numta_Workshop - Copy/Numta_Workshop/training-a/otsu/'
img = cv2.imread(path+'a16000.png', 0)
ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
ret,thresh2 = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
ret,thresh3 = cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
ret,thresh4 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO)
ret,thresh5 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)

titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]

for i in range(0,6):
    plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])

plt.show()

#%%3. Centralizing *******
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
paths_train = paths_test_f
#os.mkdir(os.path.join(project_dir,'candc2/'))
#x= imagePath.split('/')
for imagePath in paths_train:
    
    img=cv2.imread(imagePath,cv2.IMREAD_GRAYSCALE) # read image, image size is 180x180
    #plt.imshow(img_gray,cmap='gray')
    #plt.show()

     #Inversion
    img_gray=255-img
    #plt.imshow(img_gray,cmap='gray')
    #plt.show()
    
    blur = cv2.GaussianBlur(img_gray,(5,5),0)
    (thresh, img_bin) = cv2.threshold(blur,128,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    plt.imshow(img_bin,'gray')
    plt.title('Otsu’s Binarization')

     
    _,contours,_ = cv2.findContours(img_bin.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        img_gray_cont=img_bin.copy() 
        cv2.drawContours(img_gray_cont, contours, i,255, 1)
        #plt.imshow(img_gray_cont,cmap='gray')
        #plt.title('Contour: {}'.format(i+1))
        #plt.show()  

    if len(contours) > 4:
        kernel = np.ones((5,5),np.uint8)
        img_dilate=cv2.dilate(img_bin,kernel,iterations = 1)
        #plt.imshow(img_dilate,cmap='gray')
        #plt.show()

        #Reduce Thickness
        img_erode=cv2.erode(img_dilate,kernel,iterations = 1)
        #plt.imshow(img_erode,cmap='gray')
        #plt.show()
    else:
        img_erode = img_bin
        
    #Largest area contour
   
    countours_largest = sorted(contours, key=lambda x: cv2.contourArea(x))[-1]

    bb=cv2.boundingRect(countours_largest)
    #Let's observe the quality of the bounding box.
    # pt1 and pt2 are terminal coordinates of the diagonal of the rectangle
    #pt1=(bb[0],bb[1]) # upper coordinates 
    #pt2=(bb[0]+bb[2],bb[1]+bb[3]) # lower coordinates
    #img_gray_bb=img_erode.copy()
    #cv2.rectangle(img_gray_bb,pt1,pt2,255,3)
    #plt.imshow(img_gray_bb,cmap='gray')
    #plt.show()
    
    # Padding, make sure contour does not go out of image
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
    
    #img_gray_bb=img_erode.copy()
    #cv2.rectangle(img_gray_bb,pt1,pt2,255,3)
    #plt.imshow(img_gray_bb,cmap='gray')
    #plt.show()
    
    # Cropping an image
    crop_img = img_bin[pt1[1]:pt2[1], pt1[0]:pt2[0]]
    #plt.imshow(crop_img, cmap='gray')
    #plt.show()
    
   #Resizing to 28 x28
    resize_img = cv2.resize(crop_img  , (28 , 28))
    #plt.imshow(resize_img, cmap='gray')
    #plt.show()
        
    
    #os.mkdir(os.path.join(project_dir,'candc/'))
    #x= imagePath.split('/')
    imageDir2 = os.path.join(project_dir,'candc2/', imagePath.split('\\')[1])
    cv2.imwrite(imageDir2,resize_img)
    ct += 1
    print(ct)
    if ct == len(paths_train):
        print("Complete")
        break

#%%Resizing to 28 x28
#resize_img = cv2.resize(img  , (28 , 28))
#cv2.imshow('img' , resize_img)
#x = cv2.waitKey(0)
#cv2.destroyWindow('img')