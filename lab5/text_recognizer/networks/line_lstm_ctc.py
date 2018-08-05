from boltons.cacheutils import cachedproperty
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.python.client import device_lib
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D, Permute, RepeatVector, Reshape, TimeDistributed, Lambda, LSTM, GRU, CuDNNLSTM, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model as KerasModel

from text_recognizer.models.line_model import LineModel
from text_recognizer.networks.lenet import lenet
from text_recognizer.networks.misc import slide_window
from text_recognizer.networks.ctc import ctc_decode


def line_lstm_ctc(input_shape, output_shape, window_width=28, window_stride=14,
                  conv_dim=128, lstm_dim=128):
    image_height, image_width = input_shape
    output_length, num_classes = output_shape

    num_windows = int((image_width - window_width) / window_stride) + 1
    if num_windows < output_length:
        raise ValueError(f'Window width/stride need to generate at least {output_length} windows (currently {num_windows})')

    image_input = Input(shape=input_shape, name='image')
    y_true = Input(shape=(output_length,), name='y_true')
    input_length = Input(shape=(1,), name='input_length')
    label_length = Input(shape=(1,), name='label_length')

    gpu_present = len(device_lib.list_local_devices()) > 1
    lstm_fn = CuDNNLSTM if gpu_present else LSTM

    # Your code should use slide_window and extract image patches from image_input.
    # Pass a convolutional model over each image patch to generate a feature vector per window.
    # Pass these features through one or more LSTM layers.
    # Convert the lstm outputs to softmax outputs.
    # Note that lstms expect a input of shape (num_batch_size, num_timesteps, feature_length).

    ##### Your code below (Lab 3)
    image_reshaped = Reshape((image_height, image_width, 1))(image_input)
    # (image_height, image_width, 1)

    image_patches = Lambda(
        slide_window,
        arguments={'window_width': window_width, 'window_stride': window_stride}
    )(image_reshaped)
    # (num_windows, image_height, window_width, 1)

    if 0:
        # Make a LeNet and get rid of the last two layers (softmax and dropout)
        convnet = lenet((image_height, window_width, 1), (num_classes,))
        convnet = KerasModel(inputs=convnet.inputs, outputs=convnet.layers[-2].output)
        convnet_outputs = TimeDistributed(convnet)(image_patches)
        # (num_windows, 128)

        # lstm_output = Bidirectional(lstm_fn(128, return_sequences=True)(convnet))
        lstm_output0 = Bidirectional(lstm_fn(128, return_sequences=True))(convnet_outputs)
        lstm_output1 = Bidirectional(lstm_fn(128, return_sequences=True))(lstm_output0)
        lstm_output2 = Bidirectional(lstm_fn(128, return_sequences=True))(lstm_output1)
        lstm_output = Bidirectional(lstm_fn(128, return_sequences=True))(lstm_output2)
        # (num_windows, 128)

        #bidir = Bidirectional(lstm_output)
        #bidir = Bidirectional(lstm_output)

        softmax_output = Dense(num_classes, activation='softmax', name='softmax_output')(lstm_output)
        # softmax_output = Dense(num_classes, activation='softmax', name='softmax_output')(bidir)

        # softmax_output = Dense(num_classes, activation='softmax', name='softmax_output')(lstm_output)
        # (num_windows, num_classes)
        
    elif 0:
        # Make a LeNet and get rid of the last two layers (softmax and dropout)
        convnet = lenet((image_height, window_width, 1), (num_classes,))
        convnet = KerasModel(inputs=convnet.inputs, outputs=convnet.layers[-2].output)
        convnet_outputs = TimeDistributed(convnet)(image_patches)
        # (num_windows, 128)

        dropout_amount = .2
        
        # lstm_output = Bidirectional(lstm_fn(128, return_sequences=True)(convnet))
        lstm_output0 = Bidirectional(lstm_fn(128, return_sequences=True))(convnet_outputs)
        do0 = Dropout(dropout_amount)(lstm_output0)
        lstm_output1 = Bidirectional(lstm_fn(128, return_sequences=True))(do0)
        # do1 = Dropout(dropout_amount)(lstm_output1)
        lstm_output = Dropout(dropout_amount)(lstm_output1)
        # lstm_output = Bidirectional(lstm_fn(128, return_sequences=True))(do1)

        
        # lstm_output2 = Bidirectional(lstm_fn(128, return_sequences=True))(lstm_output1)
        # lstm_output = Bidirectional(lstm_fn(128, return_sequences=True))(lstm_output2)
        # (num_windows, 128)

        #bidir = Bidirectional(lstm_output)
        #bidir = Bidirectional(lstm_output)

        softmax_output = Dense(num_classes, activation='softmax', name='softmax_output')(lstm_output)
        # softmax_output = Dense(num_classes, activation='softmax', name='softmax_output')(bidir)

        # softmax_output = Dense(num_classes, activation='softmax', name='softmax_output')(lstm_output)
        # (num_windows, num_classes)
        
    elif 1:
        # restarting
        
        # Make a LeNet and get rid of the last two layers (softmax and dropout)
        convnet = lenet((image_height, window_width, 1), (num_classes,))
        convnet = KerasModel(inputs=convnet.inputs, outputs=convnet.layers[-2].output)
        convnet_outputs = TimeDistributed(convnet)(image_patches)
        # (num_windows, 128)

        dropout_amount = .2
        
        # lstm_output = Bidirectional(lstm_fn(128, return_sequences=True)(convnet))
        lstm_output0 = Bidirectional(lstm_fn(128, return_sequences=True))(convnet_outputs)
        do0 = Dropout(dropout_amount)(lstm_output0)
        lstm_output1 = Bidirectional(lstm_fn(128, return_sequences=True))(do0)
        do1 = Dropout(dropout_amount)(lstm_output1)
        lstm_output = Bidirectional(lstm_fn(128, return_sequences=True))(do1)

        
        # lstm_output2 = Bidirectional(lstm_fn(128, return_sequences=True))(lstm_output1)
        # lstm_output = Bidirectional(lstm_fn(128, return_sequences=True))(lstm_output2)
        # (num_windows, 128)

        #bidir = Bidirectional(lstm_output)
        #bidir = Bidirectional(lstm_output)

        softmax_output = Dense(num_classes, activation='softmax', name='softmax_output')(lstm_output)
        # softmax_output = Dense(num_classes, activation='softmax', name='softmax_output')(bidir)

        # softmax_output = Dense(num_classes, activation='softmax', name='softmax_output')(lstm_output)
        # (num_windows, num_classes)    elif 0:
        # SERGEY:
        # Slide a conf filter stack over image in horizontal direction.
        conv = Conv2D(conv_dim, (image_height, window_width), (1, window_stride),
                      activation='relu')(image_reshaped)
        # (1, num_windows, 128)
        # height of conv filter and height of image are same, so first dim is 1 of output
        # num_windows = (image_width - window_width) / window_stride + 1
        
        conv_squeezed = Lambda(lambda x: K.squeeze(x, 1))(conv)
        # (num_windows, 128)
    
        # lstm_output = Bidirectional(lstm_fn(128, return_sequences=True)(convnet))
        lstm_output0 = lstm_fn(lstm_dim, return_sequences=True)(conv_squeezed)
        lstm_output = lstm_fn(lstm_dim, return_sequences=True)(lstm_output0)
        softmax_output = Dense(num_classes, activation='softmax', name='softmax_output')(lstm_output)
        
    ##### Your code above (Lab 3)

    input_length_processed = Lambda(
        lambda x, num_windows=None: x * num_windows,
        arguments={'num_windows': num_windows}
    )(input_length)

    ctc_loss_output = Lambda(
        lambda x: K.ctc_batch_cost(x[0], x[1], x[2], x[3]),
        name='ctc_loss'
    )([y_true, softmax_output, input_length_processed, label_length])

    ctc_decoded_output = Lambda(
        lambda x: ctc_decode(x[0], x[1], output_length),
        name='ctc_decoded'
    )([softmax_output, input_length_processed])

    model = KerasModel(
        inputs=[image_input, y_true, input_length, label_length],
        outputs=[ctc_loss_output, ctc_decoded_output]
    )
    return model

