# Importing necessary libraries
import numpy as np
import os
import glob
import cv2
import matplotlib.pyplot as plt
import pandas as pd

# Declare fontsize variables which will be used while plotting the data
FS_AXIS_LABEL=14
FS_TITLE=17
FS_TICKS=12
FIG_WIDTH=16

project_dir='D:/Python Course_buet/Final Project/Numta_Workshop - Copy/'

#%%
os.listdir(os.path.join(project_dir,'Numta_Workshop/'))
# Notice that I am using the os.path.join() function to create the filepaths instead of writing them down explicitly with a 
# filepath separator ('\\' for windows '/' for linux). This allows us to run this notebook both in windows and linux 
# environment without manually changing the filepath separator

#%%
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

#%% Some Utility Functions
def get_img(path,mode=cv2.IMREAD_GRAYSCALE):
    # read image (if no read mode is defined, the image is read in grayscale)
     return cv2.imread(path,mode)   
def imshow_group(paths,n_per_row=10):
    # plot multiple digits in one figure, by default 10 images are plotted per
    n_sample=len(paths)
    j=np.ceil(n_sample/n_per_row)
    fig=plt.figure(figsize=(20,2*j))
    for i, path in enumerate(paths):
        img=get_img(path)
        plt.subplot(j,n_per_row,i+1)
        plt.imshow(img,cmap='gray')  
        plt.title(img.shape)
        plt.axis('off')
    return fig
def get_key(path):
    # separate the key from the filepath of an image
    return path.split(sep=os.sep)[-1]

#%% Check a few samples from dataset A

paths=np.random.choice(paths_train_a,size=40)
fig=imshow_group(paths)
fig.suptitle('Samples from {} training images in dataset A'.format(len(paths_train_a)), fontsize=FS_TITLE)
plt.show()

#%% Shape statistics of dataset A
#Train-A
shapes_train_a_sr=pd.Series([get_img(path).shape for path in paths_train_a])
shapes_train_a_sr.value_counts()
#%% Test-A

shapes_test_a_sr=pd.Series([get_img(path).shape for path in paths_test_a])
shapes_test_a_sr.value_counts()

#%% Class distribution statistics of dataset A
#Train-A
#
#Let's read the .csv file which contains the labels of dataset A. 
#We are using read_csv() function from the pandas library which will return the 
#content of the .csv file in a dataframe.

df_train_a=pd.read_csv(path_label_train_a)
df_train_a.head() # Observe first five rows 

#%%

df_train_a=df_train_a.set_index('filename')
df_train_a.head() # Observe first five rows 

#%%

labels_train_a_sr=pd.Series([df_train_a.loc[get_key(path)]['digit'] for path in paths_train_a])
labels_train_a_sr_vc=labels_train_a_sr.value_counts()

#%%

plt.figure(figsize=(FIG_WIDTH,5))
labels_train_a_sr_vc.plot(kind='bar')
plt.xticks(rotation='horizontal',fontsize=FS_TICKS)
plt.yticks(fontsize=FS_TICKS)
plt.xlabel('Digits', fontsize=FS_AXIS_LABEL)
plt.ylabel('Frequency', fontsize=FS_AXIS_LABEL)
plt.title('Train A\nMean frequency of digits per class: {}, Standard Deviation: {:.4f} '.format(labels_train_a_sr_vc.mean(),labels_train_a_sr_vc.std()),
         fontsize=FS_TITLE)
plt.show()

#%% Check a few samples from dataset B
paths=np.random.choice(paths_train_b,size=40)
fig=imshow_group(paths)
fig.suptitle('Samples from {} training images in dataset B'.format(len(paths_train_b)), fontsize=FS_TITLE)
plt.show()

#%% Shape statistics of dataset B

shapes_train_b_sr=pd.Series([get_img(path).shape for path in paths_train_b])
shapes_train_b_sr_vc=shapes_train_b_sr.value_counts()
plt.figure(figsize=(FIG_WIDTH,5))
shapes_train_b_sr_vc.plot(kind='bar')
plt.xlabel('Image Shapes', fontsize=FS_AXIS_LABEL)
plt.ylabel('Frequency', fontsize=FS_AXIS_LABEL)
plt.xticks(fontsize=FS_TICKS)
plt.yticks(fontsize=FS_TICKS)
plt.title('Train B\nNo. of unique shapes: {}'.format(shapes_train_b_sr_vc.count()),fontsize=FS_TITLE)
plt.show()

#%% Class distribution statistics of dataset B
#Train-B
df_train_b=pd.read_csv(path_label_train_b)
df_train_b=df_train_b.set_index('filename')
df_train_b.head()

#%%
labels_train_b_sr_vc=pd.Series([df_train_b.loc[get_key(path)]['digit'] for path in paths_train_b])
plt.figure(figsize=(FIG_WIDTH,5))
labels_train_b_sr_vc.plot(kind='bar')
plt.xticks(rotation='horizontal',fontsize=FS_TICKS)
plt.yticks(fontsize=FS_TICKS)
plt.xlabel('Digits', fontsize=FS_AXIS_LABEL)
plt.ylabel('Frequency', fontsize=FS_AXIS_LABEL)
plt.title('Train B\nMean frequency of digits per class: {}, Standard Deviation: {}'.format(labels_train_b_sr_vc.mean(),labels_train_b_sr_vc.std()),
         fontsize=FS_TITLE)
plt.show()

#%%Checking samples from dataset E
paths=np.random.choice(paths_train_e,40)
fig=imshow_group(paths)
fig.suptitle('Samples from {} training images in dataset E'.format(len(paths_train_e)), fontsize=FS_TITLE)
plt.show()
#%%
#Shape statistics of dataset E
shapes_train_e_sr=pd.Series([get_img(path).shape for path in paths_train_e])
shapes_train_e_sr.nunique()

shapes_train_e_sr_vc=shapes_train_e_sr.value_counts()
#Plotting 50 most frequently occuring shapes.

plt.figure(figsize=(FIG_WIDTH,5))
shapes_train_e_sr_vc.iloc[:50].plot(kind='bar')
plt.xticks(fontsize=10)
plt.yticks(fontsize=FS_TICKS)
plt.xlabel('Image Shapes', fontsize=FS_AXIS_LABEL)
plt.ylabel('Count', fontsize=FS_AXIS_LABEL)
plt.title('Train E\nNo. of unique shapes: {}\nPlot of 50 most frequently occuring shapes'.format(shapes_train_e_sr_vc.count()),fontsize=FS_TITLE)
plt.show()

#%% Class distribution statistics of dataset E
# Train E
df_train_e=pd.read_csv(path_label_train_e)
df_train_e=df_train_e.set_index('filename')
df_train_e.head()

#%%

labels_train_e_sr=pd.Series([df_train_e.loc[get_key(path)]['digit'] for path in paths_train_e])
labels_train_e_sr_vc=labels_train_e_sr.value_counts()
plt.figure(figsize=(FIG_WIDTH,5))
labels_train_e_sr_vc.plot(kind='bar')
plt.xticks(rotation='horizontal',fontsize=FS_TICKS)
plt.yticks(fontsize=FS_TICKS)
plt.xlabel('Digits', fontsize=FS_AXIS_LABEL)
plt.ylabel('Frequency', fontsize=FS_AXIS_LABEL)
plt.title('Train E\nMean frequency of digits per class: {}, Standard Deviation: {:.4f}'.format(labels_train_e_sr_vc.mean(),labels_train_e_sr_vc.std()),
         fontsize=FS_TITLE)
plt.show()



#%%



