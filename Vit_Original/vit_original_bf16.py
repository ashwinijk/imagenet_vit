# This program gives floating point weights fo a given number of convolutional layers #
import os
import pickle
import time

# Setup
import numpy as np
import tensorflow as tf
from keras import Input, regularizers, Model
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dropout, Dense
from keras.optimizers import Adam
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa

# Prepare the data
num_classes = 10
input_shape = (32, 32, 3)

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = tf.cast(x_train, tf.bfloat16)
x_test = tf.cast(x_test, tf.bfloat16)

#print('loading x_train')
#x_train_full = np.load('xtrain_compressed_8.npz')
#x_train_posit = x_train_full['arr_0']
#x_train = tf.strings.to_number(x_train_posit)

#print('loading x_test')
#x_test_full = np.load('xtest_compressed_8.npz')
#x_test_posit = x_test_full['arr_0']
#x_test = tf.strings.to_number(x_test_posit)

print(f"x_train shape: {x_train.shape} - y_train shape: {y_train.shape}")
print(f"x_test shape: {x_test.shape} - y_test shape: {y_test.shape}")

# Configure the hyperparameters
learning_rate = 0.001
weight_decay = 0.0001
batch_size = 500 #256
num_epochs = 100  #100
image_size = 32  #72 # We'll resize input images to this size
patch_size = 6  # Size of the patches to be extract from the input images
num_patches = (image_size // patch_size) ** 2
projection_dim = 32  #64
num_heads = 4
transformer_units = [projection_dim * 2, projection_dim]  # Size of the transformer layers
transformer_layers = 8 #8
mlp_head_units = [2048, 1024]  # Size of the dense layers of the final classifier

# use data augmentation
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
data_augmentation.layers[0].adapt(x_train)


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

# Build the ViT model

inputs = layers.Input(shape=input_shape)
    # Augment data.
augmented = data_augmentation(inputs)
    # Create patches.
patches = Patches(patch_size)(augmented)
    # Encode patches.
encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

for _ in range(transformer_layers):

# Create multiple layers of the Transformer block.
        # Layer normalization 1.
    x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
    attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
    x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
    x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
    x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
    encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
#representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
representation = layers.Flatten()(encoded_patches)
representation = layers.Dropout(0.2)(representation)
    # Add MLP.
features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.2)
    # Classify outputs.
logits = layers.Dense(num_classes)(features)
    # Create the Keras model.
model = keras.Model(inputs=inputs, outputs=logits)

#x = Flatten()(encoded_patches)
#x = Dropout(0.2)(x)
#x = mlp(x, hidden_units=mlp_head_units, dropout_rate=0.1)
#x = Dense(100, activation='softmax')(x)
#model = Model(inputs, x)

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

#for layer in model.layers:
#    print(layer.name)
model.summary()

weights_posit_78 = tf.cast(model.layers[78].get_weights()[0], tf.bfloat16)
bias_posit_78 = tf.cast(model.layers[78].get_weights()[1], tf.bfloat16)

weights_posit_80 = tf.cast(model.layers[80].get_weights()[0], tf.bfloat16)
bias_posit_80 = tf.cast(model.layers[80].get_weights()[1], tf.bfloat16)

weights_posit_82 = tf.cast(model.layers[82].get_weights()[0], tf.bfloat16)
bias_posit_82 = tf.cast(model.layers[82].get_weights()[1], tf.bfloat16)

print('shape', weights_posit_78.shape)
print('shape', weights_posit_80.shape)
print('shape', weights_posit_82.shape)

l78 = []
ww78 = weights_posit_78  # weights
bb78 = bias_posit_78  # biases
l78.append(ww78)
l78.append(bb78)
ls18 = model.layers[78].set_weights(l78)

# setting weights and biases after converting to posit  in layer 3 #
l80 = []
ww80 = weights_posit_80  # weights
bb80 = bias_posit_80  # biases
l80.append(ww80)
l80.append(bb80)
ls20 = model.layers[80].set_weights(l80)

# setting weights and biases after converting to posit  in layer 6 #
l82 = []
ww82 = weights_posit_82  # weights
bb82 = bias_posit_82  # biases
l82.append(ww82)
l82.append(bb82)
ls82 = model.layers[82].set_weights(l82)


begin = time.time()

r = model.fit(x_train, y_train, validation_data= (x_test, y_test) , epochs=num_epochs, batch_size = batch_size) #use_multiprocessing=True , workers= processes)

end = time.time()


# total time taken
print(f"Total runtime of the program in seconds is {end - begin}")

_, accuracy, top_1_accuracy = model.evaluate(x_test, y_test)
print(f"Test accuracy: {round(accuracy * 100, 2)}%")
print(f"Test top 5 accuracy: {round(top_1_accuracy * 100, 2)}%")