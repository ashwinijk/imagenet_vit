import glob
import time

import numpy as np
import matplotlib.pyplot as plt
from keras import regularizers
from keras.optimizers import Adam
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout, GlobalMaxPooling2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.models import Model
import softposit as sp
import tensorflow as tf

import tensorflow as tf
import numpy as np

# Suppose you have a NumPy array 'data' and corresponding labels 'labels'
#data = np.random.rand(1000, 64, 64, 3)
#labels = np.random.randint(0, 10, size=(1000,))

#data1 = np.load('imagenet_float32_100.npy',  mmap_mode='r')
#data2 = np.load('imagenet_float32_200.npy',  mmap_mode='r')

labels = np.load('imagenet_ytrain.npy',  mmap_mode='r')

# Create an empty memory-mapped array for the concatenated data
#concatenated_data = np.memmap('concatenated.npy', dtype=np.float32, mode='w+', shape=(100000, 64, 64, 3))
#concatenated_data[:50000] = data1
#concatenated_data[50000:100000] = data2

concatenated_data = np.memmap('concatenated.npy', dtype='float32', mode='r', shape=(100000, 64, 64, 3)) #mode = w+ to write
#np.save('concatenated.npy', concatenated_data)

val_data = np.load('imagenet_float32_val.npy')
val_labels = np.load('imagenet_ytrain_val.npy')

# Create a tf.data.Dataset from the NumPy arrays
dataset_new = tf.data.Dataset.from_tensor_slices((concatenated_data, labels))
dataset_val = tf.data.Dataset.from_tensor_slices((val_data, val_labels))

# Batch the dataset
batch_size = 1000
dataset_new = dataset_new.batch(batch_size)
dataset_val = dataset_val.batch(batch_size)
print('new', len(dataset_new))
# Iterate through the batches
#for batch_data, batch_labels in dataset_new:
    # Perform your operations on each batch here
    # 'batch_data' will contain a batch of data
    # 'batch_labels' will contain the corresponding batch of labels
#    print(batch_data.shape, batch_labels.shape)


i = Input(shape=(64,64,3))

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
x = Dense(200, activation='softmax')(x)

model = Model(i, x)

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


print('step4')
for batch_data, batch_labels in dataset_new:
    # Perform your operations on each batch here
    # 'batch_data' will contain a batch of data
    # 'batch_labels' will contain the corresponding batch of labels
    #print(batch_data.shape, batch_labels.shape)
    batch_data = tf.cast(batch_data, tf.bfloat16)
    model.fit(batch_data, batch_labels, validation_data = dataset_val, batch_size=100, epochs=10)

del concatenated_data
