import os
os.environ['KERAS_BACKEND'] = 'theano'
import numpy
import numpy as np
from keras.layers import Dense, Dropout, LSTM, Activation, Reshape
from keras.models import Sequential, load_model
from keras.callbacks import ModelCheckpoint
from os import listdir
from os.path import isfile, join
from parse_music import MidiFilesContainer, MidiFile


seq_len = 50

EPOCH = 1
MODEL_FILE_NAME = f'models/Model_Epoch{EPOCH}.h5'
mypath = 'music/'
files = [f for f in listdir(mypath) if isfile(join(mypath, f))]

if __name__ == '__main__':
    container = MidiFilesContainer()

    for file in files:
        midi = MidiFile(mypath + file)
        midi.read_file()
        container.append(midi)

    data = container.prepare_to_model(seq_len)
    print(data)
    model = Sequential()
    model.add(Reshape((data.model_inp.shape[1] * data.model_inp.shape[2], data.model_inp.shape[3])))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(256, return_sequences=True))
    model.add(LSTM(256))
    model.add(Dense(2 * data.count_notes * seq_len))
    model.add(Reshape((seq_len, 2, data.count_notes)))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    print('start learning')
    model.fit(data.model_inp, data.model_out, epochs=EPOCH, callbacks=[ModelCheckpoint(MODEL_FILE_NAME)])






