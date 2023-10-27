import cv2
import numpy as np
import softposit as sp
import tensorflow as tf
from skimage.metrics import structural_similarity as ssim

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

image = (image*255).astype(np.uint8)
pos = (pos*255).astype(np.uint8)
image32 = (image32 * 255).astype(np.uint8)
image_bf = (image_bf * 255).astype(np.uint8)
#########################################################################
import cv2 as cv
def ed2(img):
    #img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(img, threshold1=50, threshold2=100)  # Adjust thresholds as needed
    edge_pixel_count = np.count_nonzero(edges)
    total_pixels = img.shape[0] * img.shape[1]
    edge_density = (edge_pixel_count / total_pixels) * 100
    return edge_density

imagedensity = ed2(image32)
print(imagedensity, 'density image')

imagedensity = ed2(image_bf)
print(imagedensity, 'density bfloat16')

imagedensity = ed2(pos)
print(imagedensity, 'density pos')

mse = np.mean((image32 - image) ** 2)
print(mse, 'mse float64 and Float32')
psnr = 10 * np.log10((255.0 ** 2) / mse)
print(psnr,  'psnr float64 and Float32')
array1 = image.flatten()
array2 = image32.flatten()
array1 = np.array(array1)
array2 = np.array(array2)
ssim_value = ssim(array1, array2, multichannel=True, data_range=array1.max() - array1.min())
print(ssim_value,  'ssim float64 and Float32')

mse = np.mean((image_bf - image32) ** 2)
print(mse, 'mse bfloat16 and Float32')
psnr = 10 * np.log10((255.0 ** 2) / mse)
print(psnr,  'psnr bfloat16 and Float32')
array1 = image.flatten()
array2 = image32.flatten()
array1 = np.array(array1)
array2 = np.array(array2)
ssim_value = ssim(array1, array2, multichannel=True, data_range=array1.max() - array1.min())
print(ssim_value,  'ssim bfloat16 and Float32')


mse = np.mean((image32 - pos) ** 2)
print(mse, 'mse pos and Float32')
psnr = 10 * np.log10((255.0 ** 2) / mse)
print(psnr,  'psnr pos and Float32')
array1 = image.flatten()
array2 = image32.flatten()
array1 = np.array(array1)
array2 = np.array(array2)
ssim_value = ssim(array1, array2, multichannel=True, data_range=array1.max() - array1.min())
print(ssim_value,  'ssim pos and Float32')


###########################################################################################3

observed_values = image32#np.array([10, 20, 30, 40, 50])
predicted_values =  image_bf #np.array([12, 18, 32, 45, 48])
non_zero_indices = observed_values != 0
relative_errors = np.abs(observed_values[non_zero_indices] - predicted_values[non_zero_indices]) / np.abs(observed_values[non_zero_indices])
mean_relative_error = np.mean(relative_errors)
print("Mean Relative Error:", mean_relative_error)