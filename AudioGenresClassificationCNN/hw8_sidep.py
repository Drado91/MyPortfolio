import numpy as np
import os
from os.path import isfile
import keras
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Bidirectional, LSTM, Dropout, Activation, GRU
from keras.layers import Conv2D, concatenate, MaxPooling2D, Flatten, Embedding, Lambda
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras import backend as K
from keras.utils import np_utils
from tensorflow.keras.optimizers import RMSprop
from keras import regularizers
import librosa
import librosa.display
import matplotlib.pyplot as plt
import keract


dict_genres = {'Electronic': 0, 'Experimental': 1, 'Folk': 2, 'Hip-Hop': 3,
               'Instrumental': 4,'International': 5, 'Pop': 6, 'Rock': 7}
reverse_map = {v: k for k, v in dict_genres.items()}
print(reverse_map)
X_train = np.load('shuffled_train/arr_0.npy')
y_train = np.load('shuffled_train/arr_1.npy')
print(X_train.shape, y_train.shape)

X_valid = np.load('shuffled_valid/arr_0.npy')
y_valid = np.load('shuffled_valid/arr_1.npy')
print(X_valid.shape, y_valid.shape)
"""
one_arbitrary_sample = 514
spectogram = X_train[one_arbitrary_sample]
genre = np.argmax(y_train[one_arbitrary_sample])
print(reverse_map[genre])     # Reverse Map: Number to Label
plt.figure(figsize=(10, 5))
librosa.display.specshow(spectogram.T, y_axis='mel', x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Test Melspectogram')
plt.tight_layout()
plt.show()
"""

# The number of possible predicted classes
num_classes = 8
# The number of features in a single data input (frequency components)
n_features = X_train.shape[2]
# The length of an input sample (in seconds)
n_time = X_train.shape[1]


def conv_recurrent_model_build(model_input):
    print('Building model...')
    layer = model_input

    ### Convolutional blocks
    conv_1 = Conv2D(filters=16, kernel_size=(3, 1), strides=1,
                    padding='valid', activation='relu', name='conv_1')(layer)
    pool_1 = MaxPooling2D((2, 2))(conv_1)

    # Your Code Here
    conv_2 = Conv2D(filters=32, kernel_size=(3, 1), strides=1,
                    padding='valid', activation='relu', name='conv_2')(pool_1)
    pool_2 = MaxPooling2D((2, 2))(conv_2)
    # conv_2 = ...
    # pool_2 = ...

    # conv_3 = ...
    # pool_3 = ...
    conv_3 = Conv2D(filters=64, kernel_size=(3, 1), strides=1,
                    padding='valid', activation='relu', name='conv_3')(pool_2)
    pool_3 = MaxPooling2D((2, 2))(conv_3)

    # conv_4 = ...
    # pool_4 = ...
    conv_4 = Conv2D(filters=64, kernel_size=(3, 1), strides=1,
                    padding='valid', activation='relu', name='conv_4')(pool_3)
    pool_4 = MaxPooling2D((4, 4))(conv_4)

    # conv_5 = ...
    # pool_5 = ...
    conv_5 = Conv2D(filters=64, kernel_size=(3, 1), strides=1,
                    padding='valid', activation='relu', name='conv_5')(pool_4)
    pool_5 = MaxPooling2D((2, 2))(conv_5)

    flatten1 = Flatten()(pool_5)

    ### Recurrent Block
    lstm_count = 64

    # Pooling layer
    pool_lstm1 = MaxPooling2D((4, 2), name='pool_lstm')(layer)

    # Embedding layer
    squeezed = Lambda(lambda x: K.squeeze(x, axis=-1))(pool_lstm1)

    # Bidirectional GRU
    lstm = Bidirectional(GRU(lstm_count))(squeezed)  # default merge mode is concat

    # Concat Output
    concat = concatenate([flatten1, lstm], axis=-1, name='concat')

    ## Softmax Output
    output = Dense(num_classes, activation='softmax', name='preds')(concat)

    model_output = output
    model = Model(model_input, model_output)

    opt = RMSprop(learning_rate=0.0005)  # Optimizer
    model.compile(
        loss='categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy']
    )

    return model
n_frequency = 128
n_frames = 640
input_shape = (n_frames, n_frequency, 1)
model_input = Input(input_shape, name='input')
model = conv_recurrent_model_build(model_input)

def train_model(x_train, y_train, x_val, y_val):
    n_frequency = 128
    n_frames = 640

    x_train = np.expand_dims(x_train, axis=-1)
    x_val = np.expand_dims(x_val, axis=-1)

    input_shape = (n_frames, n_frequency, 1)
    model_input = Input(input_shape, name='input')

    model = conv_recurrent_model_build(model_input)  ### Step 1

    checkpoint_callback = ModelCheckpoint('./models/parallel/weights.best.h5', monitor='val_accuracy', verbose=1,
                                          save_best_only=True, mode='max')

    reducelr_callback = ReduceLROnPlateau(
        monitor='val_accuracy', factor=0.5, patience=10, min_delta=0.01,
        verbose=1
    )

    callbacks_list = [checkpoint_callback, reducelr_callback]

    # Fit the model and get training history.
    print('Training...')
    history = model.fit(x_train, y_train, batch_size=64, epochs=2,
                        validation_data=(x_val, y_val), verbose=1, callbacks=callbacks_list)  ### Step 2

    '''
    test_accuracy = model.eval(x_test, y_test)   ### Step 3
    '''

    return model, history
def show_summary_stats(history):
    # List all data in history
    print(history.history.keys())

    # Summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # Summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
model, history = train_model(X_train, y_train, X_valid, y_valid)
#show_summary_stats(history)
from sklearn.metrics import classification_report

y_true = np.argmax(y_valid, axis = 1)
X_valid = np.expand_dims(X_valid, axis = -1)
y_pred = model.predict(X_valid)
y_pred = np.argmax(y_pred, axis=1)
labels = [0,1,2,3,4,5,6,7]
target_names = dict_genres.keys()

print(y_true.shape, y_pred.shape)
print(classification_report(y_true, y_pred, target_names=target_names))

one_arbitrary_sample = 1
sample = X_valid[one_arbitrary_sample:one_arbitrary_sample+1]
genre = np.argmax(y_valid[one_arbitrary_sample])
print(genre)
activations = keract.get_activations(model, sample, layer_names=None, nodes_to_evaluate=None, output_format='simple', nested=False, auto_compile=True)
keract.display_activations(activations=activations,cmap='YlGnBu')
keract.display_heatmaps(activations=activations)

temp_spectrogram=activations['input'][0][:][:][0]
plt.figure(figsize=(10, 5))
librosa.display.specshow(temp_spectrogram.T, y_axis='mel', x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Test Melspectogram')
plt.tight_layout()
plt.show()
#keract.display_activations(activations, input_image=sample, save=False, data_format='channels_last', fig_size=(20, 10))