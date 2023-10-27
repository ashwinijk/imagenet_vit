# This program gives floating point weights fo a given number of convolutional layers #
import glob

import tensorflow as tf
import os
import time

from keras import regularizers
from keras.optimizers import Adam
from numpy import float32

os.system('clear')
os.system('cls')

print(tf.__version__)

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout, GlobalMaxPooling2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.models import Model
import softposit as sp

#load the cifar 10 data
#cifar10 = tf.keras.datasets.cifar10
#(x_train, y_train), (x_test, y_test) = cifar10.load_data()

#(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data();

#Split them into train & test
#x_train, x_test = x_train / 255.0, x_test / 255.0
#y_train, y_test = y_train.flatten(), y_test.flatten()

data1 = np.load('imagenet_float32_100.npy')
data2 = np.load('imagenet_float32_200.npy')

x_train = np.concatenate((data1, data2))
print("x_train.shape:", x_train.shape)
time.sleep(3)

y_train = np.load('imagenet_ytrain.npy')
print("y_train.shape", y_train.shape)
time.sleep(3)

x_val = np.load('imagenet_float32_val.npy')
print("x_val.shape:", x_val.shape)
time.sleep(3)

y_val = np.load('imagenet_ytrain_val.npy')
print("y_val.shape:", y_val.shape)
time.sleep(3)

#x_train = tf.cast(x_train, tf.bfloat16)
#print(' x_train datatype', x_train.dtype)
#time.sleep(3)

#x_val = tf.cast(x_val, tf.bfloat16)
#print(' x_val datatype', x_val.dtype)
#time.sleep(3)

# number of classes
K = len(set(y_train))
print("number of classes:", K)

i = Input(shape=x_train[0].shape)

x = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l=0.0001))(i)
x = BatchNormalization()(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l=0.0001))(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l=0.0001))(x)
x = BatchNormalization()(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l=0.0001))(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l=0.0001))(x)
x = BatchNormalization()(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l=0.0001))(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)

x = Flatten()(x)
x = Dropout(0.2)(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(K, activation='softmax')(x)

model = Model(i, x)

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# weights and biases before model training #

#for layer in model.layers:
#    print(layer.name)

weights_posit_1 = tf.cast(model.layers[1].get_weights()[0], tf.bfloat16)
bias_posit_1 = tf.cast(model.layers[1].get_weights()[1], tf.bfloat16)

weights_posit_3 = tf.cast(model.layers[3].get_weights()[0], tf.bfloat16)
bias_posit_3 = tf.cast(model.layers[3].get_weights()[1], tf.bfloat16)

weights_posit_6 = tf.cast(model.layers[6].get_weights()[0], tf.bfloat16)
bias_posit_6 = tf.cast(model.layers[6].get_weights()[1], tf.bfloat16)

weights_posit_8 = tf.cast(model.layers[8].get_weights()[0], tf.bfloat16)
bias_posit_8 = tf.cast(model.layers[8].get_weights()[1], tf.bfloat16)

weights_posit_11 = tf.cast(model.layers[11].get_weights()[0], tf.bfloat16)
bias_posit_11 = tf.cast(model.layers[11].get_weights()[1], tf.bfloat16)

weights_posit_13 = tf.cast(model.layers[13].get_weights()[0], tf.bfloat16)
bias_posit_13 = tf.cast(model.layers[13].get_weights()[1], tf.bfloat16)

weights_posit_18 = tf.cast(model.layers[18].get_weights()[0], tf.bfloat16)
bias_posit_18 = tf.cast(model.layers[18].get_weights()[1], tf.bfloat16)

weights_posit_20 = tf.cast(model.layers[20].get_weights()[0], tf.bfloat16)
bias_posit_20 = tf.cast(model.layers[20].get_weights()[1], tf.bfloat16)


#for i in range(1, 16):
#    if (i == 5 or i == 10 or i == 15):
#        print("exclude")
#    else:
#        print(model.layers[i].get_weights()[0].shape, 'and', model.layers[i].get_weights()[1].shape)
#        print(model.layers[i].get_weights()[0], 'and', model.layers[i].get_weights()[1])

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

l18 = []
ww18 = weights_posit_18  # weights
bb18 = bias_posit_18  # biases
l18.append(ww18)
l18.append(bb18)
ls18 = model.layers[18].set_weights(l18)

l20 = []
ww20 = weights_posit_20 # weights
bb20 = bias_posit_20  # biases
l20.append(ww20)
l20.append(bb20)
ls20 = model.layers[20].set_weights(l20)

#model.summary()

begin = time.time()

#chunk_size = 1000
#for i in range(0, len(x_train), chunk_size):
#    X_chunk = x_train[i:i + chunk_size]
#    y_chunk = y_train[i:i + chunk_size]
#    model.fit(X_chunk, y_chunk, validation_data=(x_val, y_val), epochs=2, batch_size=100, verbose=1)


r = model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size = 100, epochs=100)
# store end time


end = time.time()

#model.evaluate(x_test, y_test)

# total time taken
print(f"Total runtime of the program in seconds is {end - begin}")
