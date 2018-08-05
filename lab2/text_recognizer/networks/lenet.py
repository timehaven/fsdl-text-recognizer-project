from typing import Optional, Tuple

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Input, Lambda, MaxPooling2D
from tensorflow.keras.models import Sequential, Model

# import os
# import sys
# sys.path.append(os.path.abspath(os.curdir))
# from networks.rw_thievery import *

# 5x5

def model_theft_1(input_shape, output_shape, num_classes):
    
    _input_shape = (input_shape[0], input_shape[1], 1)
    #-----
    #
    # Theft #1:
    #
    # https://github.com/matthewrenze/lenet-on-mnist-with-keras-and-tensorflow-in-python
    # Create a sequential model
    model = Sequential()

    # Add the first convolution layer
    # model.add(Convolution2D(
    model.add(Conv2D(
        filters = 20,
        kernel_size = (5, 5),
        padding = "same",
        input_shape = input_shape,
        activation = "relu"))

    # Add a ReLU activation function
    #     model.add(Activation(
    #         activation = "relu"))

    # Add a pooling layer
    model.add(MaxPooling2D(
        pool_size = (2, 2),
        strides =  (2, 2)))

    # Add the second convolution layer
    #     model.add(Conv2D(
    #         filters = 50,
    #         kernel_size = (5, 5),
    #         padding = "same",
    #         activation = "relu"))

    #     # Add a second pooling layer
    #     model.add(MaxPooling2D(
    #         pool_size = (2, 2),
    #         strides = (2, 2)))

    # Flatten the network
    model.add(Flatten())

    # Add a fully-connected hidden layer
    model.add(Dense(500, activation = "relu"))


    # Add a fully-connected output layer
    model.add(Dense(num_classes, activation='softmax'))

    #-----
    #
    # Theft #1:
    
    return model


def lab2_sln(input_shape, output_shape, num_classes):
    model = Sequential()
    if len(input_shape) < 3:
        model.add(Lambda(lambda x: tf.expand_dims(x, -1), input_shape=input_shape))
        input_shape = (input_shape[0], input_shape[1], 1)
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    return model


def lenet(input_shape: Tuple[int, ...], output_shape: Tuple[int, ...]) -> Model:
    
    num_classes = output_shape[0]

    ##### Your code below (Lab 2)

    # model = model_theft_1(input_shape, output_shape, num_classes)
    model = lab2_sln(input_shape, output_shape, num_classes)

    ##### Your code above (Lab 2)

    return model

