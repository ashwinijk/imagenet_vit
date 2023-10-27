import numpy as np
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnrskimage

float_32 = np.memmap('concatenated.npy', dtype='float32', mode='r', shape=(100000, 64, 64, 3)) #mode = w+ to write
posit = np.memmap('concatenated_posit8.npy', dtype='float32', mode='r', shape=(100000, 64, 64, 3)) #mode = w+ to write

a1 = []  # Create a 1D array to store the differences
float32 = []
posit8 = []
k = 0

for i in range(3):
    for j in range(32):
        a = float_32[j][i][0][0] - posit[j][i][0][0]
        a1.append(a)
        float32.append(float_32[j][i][0][0])
        posit8.append(posit[j][i][0][0])

       # print(a1[k])
        k += 1

a1d = a1

#fig, ax = plt.subplots(figsize=(10, 7))
#ax.hist(a1d, bins=[-0.002, -0.001, 0, 0.001, 0.002])
#ax.hist(a1, bins = 40)

#plt.subplot(3, 1, 1)
#plt.hist(a1, bins=40)
#plt.xlabel("residual pixel values")
#plt.ylabel("number of pixels")
#plt.title('Difference b/n float and posit')

#plt.subplot(3, 1, 2)
#plt.hist(float32, bins=40)
#plt.xlabel("pixel values")
#plt.ylabel("number of pixels")

#plt.subplot(3, 1, 3)
#plt.hist(posit8, bins=40)
#plt.xlabel("pixel values")
#plt.ylabel("number of pixels")

#plt.show()



import cv2
import softposit as sp
import tensorflow as tf
image = cv2.imread('goldfish.JPEG')
print(image.shape)
image = image/255
image32 = np.float32(image)
imagebfloat16 = tf.cast(image32, tf.bfloat16)
image_bf = imagebfloat16.numpy()

pos = np.zeros((263, 376, 3))
for i in range(0, 263):
    for j in range(0, 376):
        for k in range(0, 3):
                convert = float(image32[i][j][k])
                temp = sp.posit8(float(convert))
                pos[i][j][k] = temp

#image = (image*255).astype(np.uint8)
#pos = (pos*255).astype(np.uint8)
#image32 = (image32 * 255).astype(np.uint8)
#image_bf = (image_bf * 255).astype(np.uint8)


#pixel_values = (pixel_values).astype(np.uint8)

#pixel_values = np.ones_like(pixel_values)
#print(pixel_values)
#cv2.imshow("OpenCV Image",pixel_values)
#cv2.waitKey(0)

plt.subplots(figsize=(10, 4))
plt.subplot(331)  # 3 rows, 1 column, plot 1
#plt.subplot(131)  # 3 rows, 1 column, plot 1
plt.imshow(image32)
plt.title('Float32')
plt.xlabel("number of pixels")
plt.ylabel("number of pixels")

plt.subplot(332)  # 3 rows, 1 column, plot 2
#plt.subplot(132)  # 3 rows, 1 column, plot 2
plt.imshow(((image_bf/255)*255).astype(np.uint8))
plt.title('Bfloat16')
plt.xlabel("number of pixels")
plt.ylabel("number of pixels")

plt.subplot(333)  # 3 rows, 1 column, plot 3
#plt.subplot(133)  # 3 rows, 1 column, plot 3
plt.imshow(pos)
plt.title('Posit8')
plt.xlabel("number of pixels")
plt.ylabel("number of pixels")

#plt.show()


plt.subplot(334)
pixel_values = image32.flatten()
plt.hist(pixel_values)
plt.xlabel("pixel value Float32")
plt.ylabel("number of pixels")

plt.subplot(335)
pixel_values = image_bf.flatten()
plt.xlabel("pixel value Bfloat16")
plt.ylabel("number of pixels")
plt.hist(pixel_values)

plt.subplot(336)
pixel_values = pos.flatten()
plt.xlabel("pixel value Posit8")
plt.ylabel("number of pixels")
plt.hist(pixel_values)


plt.subplot(337)
pixel_values = abs(image.flatten()-image32.flatten())
relative_error = np.sum(pixel_values)/np.sum(abs(image.flatten())) * 100
print(relative_error, 'relative error image32')
plt.xticks(np.arange(0.001, 1, 0.001), rotation=45)  # Adjust tick positions and rotation
plt.xlabel("Difference in pixel value")
plt.ylabel("number of pixels")
plt.hist(abs(pixel_values), bins=20, range=(0, 0.002))

plt.subplot(338)
pixel_values = abs(image32.flatten()-image_bf.flatten())
relative_error = np.sum(pixel_values)/np.sum(abs(image32.flatten())) * 100
print(relative_error, 'relative error image_bf')
plt.xticks(np.arange(0.001, 1, 0.001), rotation=45)  # Adjust tick positions and rotation
plt.xlabel("Difference in pixel value")
plt.ylabel("number of pixels")
plt.hist(abs(pixel_values), bins=20, range=(0, 0.002))

plt.subplot(339)
pixel_values = abs(image32.flatten()-pos.flatten())
relative_error = np.sum(pixel_values)/np.sum(abs(image32.flatten())) * 100
print(relative_error, 'relative error posit')
plt.xticks(np.arange(0.001, 1, 0.001), rotation=45)  # Adjust tick positions and rotation
plt.xlabel("Difference in pixel value")
plt.ylabel("number of pixels")
plt.hist(abs(pixel_values), bins=20, range=(0, 0.013))

# Adjust spacing between subplots
#plt.tight_layout()

# Display the subplot
plt.show()

from matplotlib import pyplot as plt

plt.bar(['Float64'],[2317.83],
#plt.bar([2317.83,1158.98, 579.56,289.85 ],[30,40,10,80],
label='Float64',color='c',width=.5)

plt.bar(['Float32'],[1158.98],
label='Float32', color='g',width=.5)

#plt.bar([2317.83,1158.98, 579.56,289.85 ], [30,40,10,80],
plt.bar(['Bfloat16'],[579.56],
label='Bfloat16', color='r',width=.5)

plt.bar(['Posit8'],[289.85],
label='Posit8', color='y',width=.5)

plt.legend()
plt.xlabel('Data Types')
plt.ylabel('Kbytes')
plt.title('Space occupied by Float64, Float32, Bfloat16 and Posit8')
plt.show()