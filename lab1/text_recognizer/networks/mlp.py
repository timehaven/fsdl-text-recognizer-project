import sys
import time

from typing import Tuple

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten

def mlp(input_shape: Tuple[int, ...],
        output_shape: Tuple[int, ...],
        layer_size: int=128,
        dropout_amount: float=0.2,
        num_layers: int=3) -> Model:
    """
    Simple multi-layer perceptron: just fully-connected layers with dropout between them, with softmax predictions.
    Creates num_layers layers.
    """
    num_classes = output_shape[0]
    
    ##### Your code below (Lab 1)

    # Diagnostics
    
    print(f"""
input_shape     {input_shape}
output_shape    {output_shape}
layer_size      {layer_size}
dropout_amount  {dropout_amount}
num_layers      {num_layers}
""")
    # input_shape     [28, 28]
    # output_shape    (80,)
    # layer_size      128
    # dropout_amount  0.2
    # num_layers      3          
    # sys.exit()
    
    if 0:
        print("Iteration 1:  simply add an extra layer.")
        time.sleep(3)
        # NOT SUBMITTED
        # Epoch 00007: early stopping
        # Training took 136.289008 s
        # GPU utilization: 38.75 +- 4.13
        # Test evaluation: 0.8264831546641679
        model = Sequential()
        model.add(Flatten(input_shape=input_shape))
        for _ in range(num_layers + 1):
            model.add(Dense(layer_size, activation='relu'))
            model.add(Dropout(dropout_amount))
        model.add(Dense(num_classes, activation='softmax'))
    
    elif 0:
        print("Iteration 2:  same number of layers, just 1.5x wider.")
        time.sleep(3)
        # SUBMITTED
        # Epoch 00006: early stopping
        # Training took 111.958745 s
        # GPU utilization: 37.14 +- 4.56
        # Test evaluation: 0.8363350326246743
        model = Sequential()
        model.add(Flatten(input_shape=input_shape))
        for _ in range(num_layers):
            model.add(Dense(int(1.5 * layer_size), activation='relu'))
            model.add(Dropout(dropout_amount))
        model.add(Dense(num_classes, activation='softmax'))
        
    elif 0:
        print("Iteration 3:  same number of layers, just 2x wider.")
        # SUBMITTED
        # Epoch 00006: early stopping
        # Training took 112.378707 s
        # GPU utilization: 42.47 +- 5.85
        # Test evaluation: 0.836481177411174
        time.sleep(3)
        model = Sequential()
        model.add(Flatten(input_shape=input_shape))
        for _ in range(num_layers):
            model.add(Dense(int(2 * layer_size), activation='relu'))
            model.add(Dropout(dropout_amount))
        model.add(Dense(num_classes, activation='softmax'))
    
    elif 0:
        print("Iteration 4:  same number of layers, layer_size=512.")
        # NOT SUBMITTED
        # Epoch 00005: early stopping
        # Training took 105.468419 s
        # GPU utilization: 55.13 +- 4.37
        # Test evaluation: 0.8360599365559691
        time.sleep(3)
        layer_size = 512
        model = Sequential()
        model.add(Flatten(input_shape=input_shape))
        for _ in range(num_layers):
            model.add(Dense(layer_size, activation='relu'))
            model.add(Dropout(dropout_amount))
        model.add(Dense(num_classes, activation='softmax'))
    
    elif 1:
        
        print("Iteration 5:  512, 256, 128.")
        time.sleep(3)
        layer_size = 512
        model = Sequential()
        model.add(Flatten(input_shape=input_shape))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(dropout_amount))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(dropout_amount))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(dropout_amount))
        model.add(Dense(num_classes, activation='softmax'))
        
    else:
        print("Init:  stolen from sln manual for efficiency.")
        # SUBMITTED
        time.sleep(3)
        model = Sequential()
        model.add(Flatten(input_shape=input_shape))
        for _ in range(num_layers):
            model.add(Dense(layer_size, activation='relu'))
            model.add(Dropout(dropout_amount))
        model.add(Dense(num_classes, activation='softmax'))

    ##### Your code above (Lab 1)

    return model

