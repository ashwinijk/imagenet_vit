import glob
import os

import cv2
import numpy as np

#input = '/home/ashwini/Documents/calligo_tasks/posit_vit_redo/redo_for_imagenet/tiny_imagenet_float32/val/'
#subdirectories = [f.path for f in os.scandir(input) if f.is_dir()]  #200 subdirectories
#data_list = []

#first_10_subdirectories = subdirectories#[100:200]


#for directory in first_10_subdirectories:
#    txt_files = glob.glob(os.path.join(directory, 'images/*.txt'))
#    for file_path in txt_files:
#        print(f"Found JPG file: {file_path}")
#        data_1d = np.loadtxt(file_path)
#        data_3d = data_1d.reshape(64, 64, 3)
#        data_list.append(data_3d)
#        print(len(data_list))

#total_data_array = np.array(data_list)
#np.save("imagenet_posit8_val.npy", total_data_array)

data1 = np.load('imagenet_float32_100.npy')
#data2 = np.load('imagenet_float32_200.npy')
#data3 = np.load('imagenet_float32_val.npy')

import tensorflow as tf
data1_bfloat = tf.cast(data1, tf.bfloat16)
data1_bfloat = np.array(data1_bfloat)
np.save("imagenet_bfloat16_100.npy", data1_bfloat)


