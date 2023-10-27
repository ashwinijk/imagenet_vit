import glob

import cv2
import os

import numpy as np
import tensorflow as tf
import softposit as sp

val_restruct_dir = '/home/ashwini/Documents/calligo_tasks/posit_vit_redo/redo_for_imagenet/val_restruct/'

#input = '/home/ashwini/Documents/calligo_tasks/posit_vit_redo/redo_for_imagenet/tiny-imagenet-200/train/'
output = '/home/ashwini/Documents/calligo_tasks/posit_vit_redo/redo_for_imagenet/tiny_imagenet_float32/val/'


subdirectories = [f.path for f in os.scandir(val_restruct_dir) if f.is_dir()]
#print(subdirectories)

first_10_subdirectories = subdirectories
set=0

for directory in first_10_subdirectories:
    #print(f"Processing directory: {directory}")
    jpg_files = glob.glob(os.path.join(directory, '*.JPEG'))  #images/*.JPEG for train, *.JPEG for val
    #print('jpg files',jpg_files)
    print(set)
    set=set+1
    path_parts = directory.split(os.path.sep)

    last_word = path_parts[-1]

    new_file_path = os.path.join(output, last_word)

    #os.makedirs(new_file_path, exist_ok=True)

    # Loop through each JPG file in the subdirectory
    for jpg_file in jpg_files:
       # print(f"Found JPG file: {jpg_file}")
        jpg_parts = jpg_file.split(os.path.sep)
        last_word_jpg = jpg_parts[-1]

        image = cv2.imread(jpg_file)
        image = np.array(image/255, dtype=np.float32)
        image = image.flatten()
        new_image = []                          #
        for i in range(image.shape[0]):         #
           x = image[i]                         #
           posit_element= sp.posit8(float(x))   #
           new_image.append(posit_element)      #
        #image = (image * 255).astype(np.uint8)
        output_path = os.path.join(new_file_path, "images" ,last_word_jpg)
        output_path = output_path.replace('.JPEG', '.txt')

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        #print('processing', output_path, )
        np.savetxt(output_path, new_image, fmt="%6f") #remove %6f for floating point number
       #np.savetxt(output_path, image")