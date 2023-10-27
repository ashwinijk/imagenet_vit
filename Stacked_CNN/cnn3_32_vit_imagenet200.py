import os
import sys
import time
from multiprocessing import Pool
import cv2
import pandas as pd
import time

# Setup
import numpy as np
import tensorflow as tf
from keras import Input, regularizers, Model
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dropout, Dense
from keras.optimizers import Adam
from tensorflow import keras
from keras import layers
import tensorflow_addons as tfa
from numpy import load

# function to read image-paths from image files(1,00,000 .jpeg) using pandas
def dataframe_load(path):
    df = pd.read_csv(path)
    path_list = df.values[:, 1]
    #print(path_list[:10])
    return path_list
def dataframe_load_test(path):
    df = pd.read_csv(path)
    path_list = df.values[:, 2]
    #print(path_list[:10])
    return path_list
# function to load image from path using opencv
def train_images(path):
    img = cv2.imread(path)
    return img


# main function to parallely load all images from paths
def main(train_path):
    # loading the paths to all images
    path_list = dataframe_load(train_path)
    # displaying number of images in imagenet
    print("Number of images to load :", len(path_list), '\n', path_list[:10])

    print("Logical Cores/Threads : ", os.cpu_count())
    with Pool(2) as pool:
        images = pool.map(train_images, path_list)
    print("Success")
    print("Number of Loaded Images :  ", len(images))
    return images

def main1(test_path):

    # loading the paths to all images
    path_list = dataframe_load_test(test_path)
    # displaying number of images in imagenet
    print("Number of images to load :", len(path_list), '\n', path_list[:10])

    print("Logical Cores/Threads : ", os.cpu_count())
    with Pool(2) as pool:
        images_test = pool.map(train_images, path_list)
    print("Success")
    print("Number of Loaded Images :  ", len(images_test))
    return images_test

if __name__ == "__main__":
    # timer

    ##LOADING DATA
    start_time = time.time()
    import numpy as np
    # setting a test path - image size (64,64,3)
    # train_path = '/home/adithya/tiny-imagenet-200/train_label.csv'
    train_path = '/home/ashwini/Documents/calligo_tasks/posit_vit_redo/redo_for_imagenet/train_label.csv'
    test_path='/home/ashwini/Documents/calligo_tasks/posit_vit_redo/redo_for_imagenet/valid_label.csv'
    # running main function
    X_train=np.array(main(train_path))
    X_test=np.array(main1(test_path))
    df = pd.read_csv(train_path)
    df1 = pd.read_csv(test_path)
    y_train=np.array(list(df.values[:,2]))
    y_test=np.array(list(df1.values[:,1]))

    print(X_train.shape," " , X_test.shape," ", y_train.shape," ",y_test.shape)

    learning_rate = 0.001
    weight_decay = 0.0001
    batch_size = 512  # 256
    num_epochs = 100  # 100
    image_size = 32  # 72 # We'll resize input images to this size
    patch_size = 6  # Size of the patches to be extract from the input images
    num_patches = (image_size // patch_size) ** 2
    projection_dim = 32  # 32 or 64
    num_heads = 4
    transformer_units = [projection_dim * 2, projection_dim]  # Size of the transformer layers
    transformer_layers = 8
    mlp_head_units = [64, 64]  # [2048, 1024] Size of the dense layers of the final classifier
    num_classes = 200
    input_shape = (64, 64, 3)

    ##TRAINING STARTS HERE
    #-----------------------------------------------------------------------------------

    data_augmentation = keras.Sequential(
        [
            layers.Normalization(),
            layers.Resizing(image_size, image_size),
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(factor=0.02),
            layers.RandomZoom(
                height_factor=0.2, width_factor=0.2
            ),
        ],
        name="data_augmentation",
    )
    # Compute the mean and the variance of the training data for normalization.
    data_augmentation.layers[0].adapt(X_train)


    # Implement multilayer perceptron (MLP)
    def mlp(x, hidden_units, dropout_rate):
        for units in hidden_units:
            x = layers.Dense(units, activation=tf.nn.gelu)(x)
            x = layers.Dropout(dropout_rate)(x)
        return x


    # Implement patch creation as a layer

    class Patches(layers.Layer):
        def __init__(self, patch_size):
            super().__init__()
            self.patch_size = patch_size

        def call(self, images):
            batch_size = tf.shape(images)[0]
            patches = tf.image.extract_patches(
                images=images,
                sizes=[1, self.patch_size, self.patch_size, 1],
                strides=[1, self.patch_size, self.patch_size, 1],
                rates=[1, 1, 1, 1],
                padding="VALID",
            )
            patch_dims = patches.shape[-1]
            patches = tf.reshape(patches, [batch_size, -1, patch_dims])
            return patches


    # Implement the patch encoding layer
    class PatchEncoder(layers.Layer):
        def __init__(self, num_patches, projection_dim):
            super().__init__()
            self.num_patches = num_patches
            self.projection = layers.Dense(units=projection_dim)
            self.position_embedding = layers.Embedding(
                input_dim=num_patches, output_dim=projection_dim
            )

        def call(self, patch):
            positions = tf.range(start=0, limit=self.num_patches, delta=1)
            encoded = self.projection(patch) + self.position_embedding(positions)
            return encoded

    inputs = layers.Input(shape=input_shape)
    # Augment data.
    augmented = data_augmentation(inputs)
    # Create patches.
    patches = Patches(patch_size)(augmented)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    x = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l=0.0001))(inputs)
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

    for _ in range(transformer_layers):
        # Create multiple layers of the Transformer block.
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(x)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim, dropout=0.1)(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, x])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim, dropout=0.1)(x, x)
    # x = layers.Add()([att, x])
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.2)(representation)
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.2)
    logits = layers.Dense(num_classes)(features)
    model = keras.Model(inputs=inputs, outputs=logits)

    # x = Flatten()(encoded_patches)
    # x = Dropout(0.2)(x)
    # x = mlp(x, hidden_units=mlp_head_units, dropout_rate=0.1)
    # x = Dense(100, activation='softmax')(x)
    # model = Model(inputs, x)

    optimizer = tfa.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            keras.metrics.SparseTopKCategoricalAccuracy(1, name="top-1-accuracy"),
        ],
    )

    # weights and biases before model training #

    # for layer in model.layers:
    #    print(layer.name)
    model.summary()

    begin = time.time()

    r = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size = 512)

    # store end time
    end = time.time()

    model.evaluate(X_test, y_test)

    # total time taken
    print(f"Total runtime of the program in seconds is {end - begin}")