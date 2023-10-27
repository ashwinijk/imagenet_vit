import tensorflow as tf
import os

from func_get_posit import *

os.system('clear')

print(tf.__version__)

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout, GlobalMaxPooling2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.models import Model
import softposit as sp

cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

#Split them into train & test
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train, y_test = y_train.flatten(), y_test.flatten()

print("x_train.shape:", x_train.shape)
print("y_train.shape", y_train.shape)
print('x_train datatype', x_train.dtype)
print(' datatype', y_train.dtype)

print(' x_train datatype', x_train.dtype)

# number of classes
K = len(set(y_train))
print("number of classes:", K)

i = Input(shape=x_train[0].shape)

x = Conv2D(32, (3, 3), activation='relu', padding='same')(i)
x = BatchNormalization()(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)

x = Flatten()(x)
x = Dropout(0.2)(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(K, activation='softmax')(x)

model = Model(i, x)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# conversion of weights and biases from float to posit #

# Extract weights for the convolutional layer #

weights_posit_1 = model.layers[1].get_weights()[0]
bias_posit_1 = model.layers[1].get_weights()[1]

weights_posit_3 = model.layers[3].get_weights()[0]
bias_posit_3 = model.layers[3].get_weights()[1]

weights_posit_6 = model.layers[6].get_weights()[0]
bias_posit_6 = model.layers[6].get_weights()[1]

weights_posit_8 = model.layers[8].get_weights()[0]
bias_posit_8 = model.layers[8].get_weights()[1]

weights_posit_11 = model.layers[11].get_weights()[0]
bias_posit_11 = model.layers[11].get_weights()[1]

weights_posit_13 = model.layers[13].get_weights()[0]
bias_posit_13 = model.layers[13].get_weights()[1]


#for i in range(0,3):
#    for j in range(0,3):
#        for k in range(0,3):
#            for l in range(0,32):
#                print(weights_posit_1[i][j][k][l])

# conversion of weights and biases from float to posit #

for i in range(0, 3):
    for j in range(0, 3):
        for k in range(0, 3):
            for l in range(0, 32):
                convert = float(weights_posit_1[i][j][k][l])
                temp = sp.posit16(float(convert))
                weights_posit_1[i][j][k][l] = temp

for i in range(0, 3):
    for j in range(0, 3):
        for k in range(0, 32):
            for l in range(0, 32):
                convert = float(weights_posit_3[i][j][k][l])
                temp = sp.posit16(float(convert))
                weights_posit_3[i][j][k][l] = temp

for i in range(0, 3):
    for j in range(0, 3):
        for k in range(0, 32):
            for l in range(0, 64):
                convert = float(weights_posit_6[i][j][k][l])
                temp = sp.posit16(float(convert))
                weights_posit_6[i][j][k][l] = temp

for i in range(0, 3):
    for j in range(0, 3):
        for k in range(0, 64):
            for l in range(0, 64):
                convert = float(weights_posit_8[i][j][k][l])
                temp = sp.posit16(float(convert))
                weights_posit_8[i][j][k][l] = temp

for i in range(0, 3):
    for j in range(0, 3):
        for k in range(0, 64):
            for l in range(0, 128):
                convert = float(weights_posit_11[i][j][k][l])
                temp = sp.posit16(float(convert))
                weights_posit_11[i][j][k][l] = temp

for i in range(0, 3):
    for j in range(0, 3):
        for k in range(0, 128):
            for l in range(0, 128):
                convert = float(weights_posit_13[i][j][k][l])
                temp = sp.posit16(float(convert))
                weights_posit_13[i][j][k][l] = temp

# setting weights and biases after converting to posit  in layer 1 #
l1 = []
ww1 = weights_posit_1  # weights
bb1 = bias_posit_1  # biases
l1.append(ww1)
l1.append(bb1)
ls1 = model.layers[1].set_weights(l1)

# setting weights and biases after converting to posit  in layer 3 #
l3 = []
ww3 = weights_posit_3  # weights
bb3 = bias_posit_3  # biases
l3.append(ww3)
l3.append(bb3)
ls3 = model.layers[3].set_weights(l3)

# setting weights and biases after converting to posit  in layer 6 #
l6 = []
ww6 = weights_posit_6  # weights
bb6 = bias_posit_6  # biases
l6.append(ww6)
l6.append(bb6)
ls6 = model.layers[6].set_weights(l6)

# setting weights and biases after converting to posit  in layer 8 #
l8 = []
ww8 = weights_posit_8  # weights
bb8 = bias_posit_8  # biases
l8.append(ww8)
l8.append(bb8)
ls8 = model.layers[8].set_weights(l8)

# setting weights and biases after converting to posit  in layer 11 #
l11 = []
ww11 = weights_posit_11  # weights
bb11 = bias_posit_11  # biases
l11.append(ww11)
l11.append(bb11)
ls11 = model.layers[11].set_weights(l11)

# setting weights and biases after converting to posit  in layer 13 #
l13 = []
ww13 = weights_posit_13  # weights
bb13 = bias_posit_13  # biases
l13.append(ww13)
l13.append(bb13)
ls13 = model.layers[13].set_weights(l13)

r = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=5)
