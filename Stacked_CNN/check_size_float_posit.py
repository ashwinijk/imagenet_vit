import os
import sys

import numpy as np

#float_32 = np.memmap('concatenated.npy', dtype='float32', mode='r', shape=(100000, 64, 64, 3)) #mode = w+ to write
#posit = np.memmap('concatenated_posit8.npy', dtype='float32', mode='r', shape=(100000, 64, 64, 3)) #mode = w+ to write


#float_32 = np.load('imagenet_float32_100.npy')
#posit = np.load('imagenet_posit8_100.npy')

# Create a list of 10 float instances for measurement
import cv2
import softposit as sp
import tensorflow as tf
image = cv2.imread('goldfish.JPEG')
image = image/255
image32 = np.float32(image)
imagebfloat16 = tf.cast(image32, tf.bfloat16)
imagebfloat16 = imagebfloat16.numpy()


pos = np.zeros((263, 376, 3))
for i in range(0, 263):
    for j in range(0, 376):
        for k in range(0, 3):
                convert = (image32[i][j][k])
                temp = sp.posit8(float(convert))
                pos[i][j][k] = temp

float_instances = image32 * 255 #np.float32(image32) # Initialize with 0.0, change values as needed
size_bytes = sys.getsizeof(float_instances)
size_kb = size_bytes / 1024.0


print(f"Size of float instances: {size_bytes} bytes ({size_kb:.2f} KB)")

##########################################################################################

image = cv2.imread('goldfish.JPEG')
image = image/255

pos = np.zeros((263, 376, 3))
for i in range(0, 263):
    for j in range(0, 376):
        for k in range(0, 3):
                convert = (image[i][j][k])
                temp = sp.posit8(float(convert))
                pos[i][j][k] = temp


image = pos*255
eight_bit_array = np.array(image, dtype=np.uint8)
size_bytes = sys.getsizeof(eight_bit_array)
size_kb = size_bytes / 1024.0
print(f"Size of float instances: {size_bytes} bytes ({size_kb:.2f} KB)")
###