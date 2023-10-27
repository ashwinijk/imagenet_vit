# -*- coding: utf-8 -*-
"""imagenet-1k.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1u0_mhjZI4xprFCqLM6YkqR3xmCjXKLLr
"""
import numpy as np
from tqdm import tqdm

# Commented out IPython magic to ensure Python compatibility.
# %cd '/content/drive/MyDrive/Colab Notebooks'

IMG_DIR='/home/ashwini/Documents/calligo_tasks/posit_vit_redo/redo_for_imagenet/tiny-imagenet-200/'
IMG_ROOT=[IMG_DIR+'train/', IMG_DIR+'test/', IMG_DIR+'val/images']

import glob

source='/home/ashwini/Documents/calligo_tasks/posit_vit_redo/redo_for_imagenet/tiny-imagenet-200/val/'
#s1='tiny-imagenet-200/train/n03584254/images/'
filepath=glob.glob(source+'**')
#f1=filepath=glob.glob(s1+'**')



s1=filepath[0]+'/images/'
fp=glob.glob(s1+'**')
#fp[0].split('/')[2]

import pandas as pd
df = pd.read_csv('/home/ashwini/Documents/calligo_tasks/posit_vit_redo/redo_for_imagenet/tiny-imagenet-200/val/val_annotations.txt', delimiter = "\t",names=["Images","Labels","x","y","z","w"])

df.drop(['x', 'y','z','w'], axis=1,inplace=True)

df.to_csv("/home/ashwini/Documents/calligo_tasks/posit_vit_redo/redo_for_imagenet/tiny-imagenet-200/validation.csv")



import pandas as pd

#df=[]

for i in tqdm(range(len(filepath))):
  path=filepath[i]+'/images/'
  fp=glob.glob(path+'**')
  #df.iloc[0,0]=fp
  #df.append(fp)

# Import label encoder
from sklearn import preprocessing

# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()

# Encode labels in column 'species'.
df['Labels']= label_encoder.fit_transform(df['Labels'])

df.head()

df.to_csv("/home/ashwini/Documents/calligo_tasks/posit_vit_redo/redo_for_imagenet/tiny-imagenet-200/validation_encoded.csv")

#df1 = []

#df1['Path'][0].split('/')[2]

label=[]
for i in tqdm(range(len(df1))):
  lb=df1['Path'][i].split('/')[2]
  label.append(lb)

df1['Label']=label

df1['Path']

# Import label encoder
from sklearn import preprocessing

# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()

# Encode labels in column 'species'.
df1['Label']= label_encoder.fit_transform(df1['Label'])

df1.to_csv('/home/ashwini/Documents/calligo_tasks/posit_vit_redo/redo_for_imagenet/tiny-imagenet-200/train_label_1.csv')

import pandas as pd

# create a dataframe with a column of image paths
# df = pd.read_excel('filename.xlsx')  # replace with your filename and path
image_paths = df['Images']

# add the file path to each image path using a lambda function
df['New Image Paths'] = image_paths.apply(lambda x: 'tiny-imagenet-200/val/images/' + x)

# replace the original column with the new column
df = df.drop('Images', axis=1)  # drop the original column
df = df.rename(columns={'New Image Paths': 'Image Paths'})  # rename the new column

df.head()

df.to_csv('/home/ashwini/Documents/calligo_tasks/posit_vit_redo/redo_for_imagenet/tiny-imagenet-200/valid_label_1.csv')

p=[]
for i in tqdm(range(len(df))):
  for j in tqdm(range(len(df[i]))):
    path=df[i][j]
    p.append(path)

df1=pd.DataFrame(p, columns=['Path'])
#df1.to_csv('/content/drive/MyDrive/Colab Notebooks/tiny-imagenet-200/Paths.csv')
df1.head()

import cv2 as cv
import imageio

lr=[]
for i in tqdm(range(len(df1))):
  img=cv.imread(df1.values[i,0])
  img=cv.resize(img, (64,64))
  lr.append(img)

img_arr=np.array(lr)

