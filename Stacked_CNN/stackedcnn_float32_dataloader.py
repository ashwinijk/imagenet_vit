import os
import shutil

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Model, regularizers
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.layers import Conv2D, MaxPool2D, Dropout, Dense, Input, concatenate, GlobalAveragePooling2D, \
    AveragePooling2D, Flatten, BatchNormalization, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, Adam
from keras.callbacks import LearningRateScheduler
import math
import softposit as sp

from tensorflow_addons.optimizers import AdamW


#image = cv2.imread("/home/ashwini/Documents/calligo_tasks/posit_vit_redo/redo_for_imagenet/val_restruct/n01443537/val_653.JPEG")
#cv2.imshow("Image", image)
#cv2.waitKey(0)

num_classes = 200
# Set the paths to the directories
val_dir = '/home/ashwini/Documents/calligo_tasks/posit_vit_redo/redo_for_imagenet/tiny-imagenet-200/val'
val_restruct_dir = '/home/ashwini/Documents/calligo_tasks/posit_vit_redo/redo_for_imagenet/val_restruct/'

# Load the annotations file
annotations_file = os.path.join(val_dir, 'val_annotations.txt')
annotations_df = pd.read_csv(annotations_file, sep='\t', header=None,
                             usecols=[0, 1], names=['filename', 'label'])

# Loop over the unique labels and create subdirectories
unique_labels = annotations_df['label'].unique()
for label in unique_labels:
    label_dir = os.path.join(val_restruct_dir, label)
    os.makedirs(label_dir, exist_ok=True)

# Move the images to their corresponding subdirectories
for index, row in annotations_df.iterrows():
    image_path = os.path.join(val_dir, 'images', row['filename'])
    label_dir = os.path.join(val_restruct_dir, row['label'])
    shutil.copy(image_path, label_dir)


def load_dataset(img_rows, img_cols, batch_size, augment=True):
    # Define data augmentation parameters for training data
    if augment:
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')
    else:
        train_datagen = ImageDataGenerator(rescale=1./255)

    test_datagen = ImageDataGenerator(rescale=1./255)

    # Load original training data
    train_generator = train_datagen.flow_from_directory(
        '/home/ashwini/Documents/calligo_tasks/posit_vit_redo/redo_for_imagenet/tiny-imagenet-200/train',
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        class_mode='categorical')

    # Load validation data
    validation_generator = test_datagen.flow_from_directory(
        '/home/ashwini/Documents/calligo_tasks/posit_vit_redo/redo_for_imagenet/val_restruct',
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        class_mode='categorical')

    return train_generator, validation_generator

batch_size = 100

train_generator, validation_generator = load_dataset(64, 64, batch_size, augment=False)
augmented_train_generator = load_dataset(64, 64, batch_size, augment=False)[0]

###################################################################################

i = Input(shape=(64, 64, 3))

x = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l=0.0001))(i)
x = BatchNormalization()(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l=0.0001))(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l=0.0001))(x)
x = BatchNormalization()(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l=0.0001))(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l=0.0001))(x)
x = BatchNormalization()(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l=0.0001))(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)

x = Flatten()(x)
x = Dropout(0.2)(x) #0.2
x = Dense(32, activation='relu')(x) #1024
x = Dropout(0.2)(x) #0.2
x = Dense(200, activation='softmax')(x)

model = Model(i, x)

adamw = AdamW(learning_rate=0.001, weight_decay=0.0001) #lr = 0.001, wd = 0.0001
sgd = SGD(lr=0.001, decay=0.0001, momentum=0.9, nesterov=True)

#model.compile(loss=['categorical_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy'], loss_weights=[1, 0.3, 0.3], optimizer=adamw, metrics=['accuracy'])
model.compile(
    optimizer=adamw,
    loss=keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=[
        keras.metrics.CategoricalAccuracy(name="accuracy")],
)
def step_decay(epoch):
    initial_lr = 0.001
    drop = 0.001
    epochs_drop = 8
    lr = initial_lr * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lr

lr_scheduler = LearningRateScheduler(step_decay)

model.fit_generator(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=50,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
   #callbacks=[lr_scheduler]
    )


#X_train, X_validation, y_train, y_validation = train_test_split(
#    X_train, y_train, test_size=0.5) #, random_state=42, shuffle=True)

#r = model.fit(X_train, y_train, validation_data=(X_validation, y_validation), epochs=100, batch_size=batch_size)

#scores = model.evaluate_generator(validation_generator, steps=validation_generator.samples // batch_size, verbose=1)
#print("Final Accuracy: %.2f%%" % (scores[4] * 100))

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    '/home/ashwini/Documents/calligo_tasks/posit_vit_redo/redo_for_imagenet/tiny-imagenet-200/test',
    target_size=(64, 64),
    batch_size=batch_size,
    class_mode='categorical')

scores = model.evaluate_generator(test_generator, steps=test_generator.samples // batch_size, verbose=1)
print("Final Accuracy: %.2f%%" % (scores[4] * 100))