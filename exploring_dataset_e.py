
#%% Exploring dataset E

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
# Setup paths
project_dir='D:/Python Course_buet/Final Project/Numta_Workshop - Copy/'

path_label_train_e=glob.glob(os.path.join(project_dir,'Numta_Workshop/','training-e','*.png'))
#path_label_test_e=os.path.join(project_dir,'Final_DB','testing-e.csv')


# Let's get the label files in a dataframe

df_train_e=os.path.join(project_dir,'Numta_Workshop/','training-e.csv')



#%%
# Let's get the label files in a dataframe 
df_train_e=pd.read_csv(df_train_e)
#df_test_e=pd.read_csv(path_label_test_e)
df_train_e.head()

#%% Distribution of digits by age
train_age_vc=df_train_e['age'].value_counts().sort_index()
#barplot
plt.figure(figsize=(FIG_WIDTH,5))
plt.suptitle('Age Distribution (Train)',fontsize=FS_TITLE)
plt.subplot(1,2,1)
train_age_vc.plot(kind='bar')
plt.xticks(rotation='horizontal',fontsize=FS_TICKS)
plt.yticks(fontsize=FS_TICKS)
plt.xlabel('Age', fontsize=FS_AXIS_LABEL)
plt.ylabel('Frequency', fontsize=FS_AXIS_LABEL)
#boxplot
plt.subplot(1,2,2)
bp_dict=plt.boxplot(df_train_e['age'])
plt.xticks([0],[''],fontsize=FS_TICKS)
plt.ylabel('Age', fontsize=FS_AXIS_LABEL)
for line in bp_dict['medians']:
    # get position data for median line (2nd quartile line)
    x, y = line.get_xydata()[1] # terminal point of median line
    plt.text(x, y, '{:.0f}'.format(y), horizontalalignment='left') 

for line in bp_dict['boxes']:
    # get position data for 1st quartile line
    x, y = line.get_xydata()[0] 
    plt.text(x,y, '{:.0f}'.format(y), horizontalalignment='right', verticalalignment='top')     
    # get position data for 3rd quartile line
    x, y = line.get_xydata()[3] 
    plt.text(x,y, '{:.0f}'.format(y), horizontalalignment='right', verticalalignment='top')
plt.show()

#%% Distribution of digits by district
train_district_vc=df_train_e['districtid'].value_counts()
plt.figure(figsize=(FIG_WIDTH,5))
train_district_vc.plot(kind='bar')
plt.yticks(fontsize=FS_TICKS)
xlabels=['Dhaka ({:.2f}%)'.format(train_district_vc.loc[1]/train_district_vc.sum()*100),
        'Comilla ({:.2f})%'.format(train_district_vc.loc[2]/train_district_vc.sum()*100)]
plt.xticks([0,1],xlabels,rotation='horizontal',fontsize=FS_TICKS)
plt.xlabel('District', fontsize=FS_AXIS_LABEL)
plt.ylabel('Frequency', fontsize=FS_AXIS_LABEL)
plt.title('District Distribution (Train)',fontsize=FS_TITLE)
plt.show()
#%%  Distribution of images by gender

train_gender_vc=df_train_e['gender'].value_counts()
plt.figure(figsize=(FIG_WIDTH,5))
train_gender_vc.plot(kind='bar')
plt.yticks(fontsize=FS_TICKS)
xlabels=['Male ({:.2f}%)'.format(train_gender_vc.loc[0]/train_gender_vc.sum()*100),
        'Female ({:.2f})%'.format(train_gender_vc.loc[1]/train_gender_vc.sum()*100)]
plt.xticks([0,1],xlabels,rotation='horizontal',fontsize=FS_TICKS)
plt.xlabel('Gender', fontsize=FS_AXIS_LABEL)
plt.ylabel('Frequency', fontsize=FS_AXIS_LABEL)
plt.title('Gender Distribution (Train)',fontsize=FS_TITLE)
plt.show()
#%%
