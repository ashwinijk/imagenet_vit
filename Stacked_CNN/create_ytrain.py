import glob

import cv2
import os

import numpy as np
import tensorflow as tf
import softposit as sp

#val_restruct_dir = '/home/ashwini/Documents/calligo_tasks/posit_vit_redo/redo_for_imagenet/val_restruct/'

input = '/home/ashwini/Documents/calligo_tasks/posit_vit_redo/redo_for_imagenet/tiny-imagenet-200/train/'
output = '/home/ashwini/Documents/calligo_tasks/posit_vit_redo/redo_for_imagenet/tiny_imagenet_float32/train/'

subdirectories = [f.path for f in os.scandir(output) if f.is_dir()]
class_to_id = {class_name: i for i, class_name in enumerate(subdirectories)}
#print(class_to_id)

num_intervals = 200  # Three intervals: 0-499, 500-999, 1000-1499
interval_size = 250 #500

# Create an array with the specified pattern
result_array = np.repeat(np.arange(num_intervals), interval_size)
np.save("dummy_ytain.npy", result_array)

# Print the result_array
print(result_array.shape)

