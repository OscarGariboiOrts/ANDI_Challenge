from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Conv1D
from keras.layers import Dropout
from keras.layers import Bidirectional
import tensorflow as tf

from keras.preprocessing import sequence as seqq
from keras.layers import Input

from keras.models import Model

from keras.optimizers import Adam
from keras.utils import to_categorical

from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

import numpy as np
import sys
import argparse

def main(args):
    """
        Program's main function.

        Script to train model for classification of trajectories in 1D under the scope of the ANDI Challenge.

        Params:
         - train_data: path to the dataset used to train the model.
         - truth_file: path to the truth file.
         - path_to_save_model: path where the models are to be saved.

        Output:
         - models saved to the location previously indicated.

    """

    tf.random.set_seed(1)

    train_file = args.train_data
    truth_file = args.truth_file
    path_to_model = args.path_to_save_model

    with open(train_file, "r") as f:
        task_raw = [np.array(x.split(";"), dtype="float64") for x in f.readlines()]
        task_raw = np.array([x[1:] for x in task_raw]) 
        task = task_raw[:,]
    task = np.array(task)

    with open(truth_file, "r") as f:
        ref = [x.split(";") for x in f.readlines()]
        ref = [float(x[1].replace("\n", "")) for x in ref]
        ref_cat = to_categorical(ref)
        ref_cat = ref_cat[:,]

    max_length = 1000
    task_HomLen = seqq.pad_sequences(task, maxlen=max_length, dtype="float64")
    task_HomLenRes = task_HomLen.reshape((-1, 1000,1))    

    n_train = 3800000
    n_trainval = 4000000
    trainX, testX = task_HomLenRes[:n_train, :], task_HomLenRes[n_train:n_trainval, :]
    trainY, testY = ref_cat[:n_train], ref_cat[n_train:n_trainval]

    # define model
    optimizer = Adam(lr=0.001, decay=1e-6, clipnorm=1.0)

    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=8, activation='relu', input_shape=(1000,1)))
    model.add(Conv1D(filters=64, kernel_size=8, activation='relu'))
    model.add(Bidirectional(LSTM(32,return_sequences=True)))
    model.add(Dropout(0.1))
    model.add(Bidirectional(LSTM(32,return_sequences=False)))    
    model.add(Dense(5, activation='softmax'))

    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
    mc = ModelCheckpoint(path_to_model + 'task2_dim1_-{epoch:03d}-{val_accuracy:.4f}.h5', monitor='val_accuracy', mode='max', verbose=0, save_best_only=True)

    print('------------------------------------------------------------------------')
    print(f'Training ...')
    
    history = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=200, verbose=1, callbacks=[es, mc])

    print("Program finished.")
    sys.stdout.flush()

    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", help='Path to the training text file', default=".")
    parser.add_argument("--truth_file", help="Path to the truth txt file", default=".")
    parser.add_argument("--path_to_save_model", help="Path to save the best model", default=".")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    main(args)


