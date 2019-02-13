from __future__ import division
import numpy as np
from numpy import newaxis
from keras.models import Model
from keras.layers import Input, Dense
from sklearn.model_selection import KFold
import keras.backend as K
#import theano.tensor
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.layers import Dropout
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, Merge, Flatten, Concatenate, Reshape, LSTM
from keras.layers.core import Lambda
from keras.callbacks import ModelCheckpoint, History
from keras import optimizers
import keras

class NeuralNetwork(object):
    """This class creates a neural network model according to the _init_ parameters"""
    # Loss is either categorical_crossentropy (labels should have size [batchSize, classes])
    # or sparse_categorical_crossentropy (labels have size [batchSize, 1] with elements in [0, l - 1])
    def __init__(self, n, m, r, dic_size, l, epochs=5, batchSize=128, checkpoint_period=1, activation='sigmoid', monitor='val_loss', loss='binary_crossentropy'):
        self._n = n  # number of samples
        self._m = m  # number of image features
        self._r = r  # number of word features
        self._dic_size = dic_size # size of the dictionary
        self._l = l  # number of possible labels
        self._epochs = epochs
        self._batchSize = batchSize
        self._wordEmbeddingSize = 1024
        if (self._wordEmbeddingSize % self._r != 0):
            print("Dimension mismatch: number of words in a question has to divide embedding size")
        self._checkpoint_period = checkpoint_period
        self._activation = activation
        self._monitor = monitor
        self._loss = loss

        image_input = Input(shape=(self._m,),name="image_input")
        word_input = Input(shape=(self._r,),name="word_input")
        word_embedding = Embedding(self._dic_size, int(self._wordEmbeddingSize / self._r), input_length=self._r, name='word_embedding')(word_input)
        flatten_embedding = Flatten(name="flatten_embedding")(word_embedding)
        merge = keras.layers.concatenate([image_input, flatten_embedding])
        hidden_layer1 = Dense(4096, activation='relu')(merge)
        hidden_layer2 = Dense(4096, activation='relu')(hidden_layer1)
        hidden_layer3 = Dense(4096, activation='relu')(hidden_layer2)
        output_layer = Dense(self._l, activation=self._activation, name='output_layer')(hidden_layer3)

        self._model = Model(inputs=[image_input, word_input], outputs=[output_layer])
        print(self._model.summary())
        self._model.compile(loss=self._loss, optimizer='adam', metrics=['accuracy']) #TODO what loss?

    def get_model_summary(self):
        return self._model.summary()

    def fit(self, images, questions, labels):
        # Split the data
        # self._matrix = matrix
        self._labels = labels
        checkpointer = ModelCheckpoint(filepath="weights/weights-{epoch:02d}-{val_loss:.4f}.hdf5", verbose=1, save_best_only=True, monitor=self._monitor, period=self._checkpoint_period)
        history = History()
        
        print(images.shape, questions.shape, self._labels.shape)            
        self._model.fit(x=[images, questions],y=self._labels, batch_size=self._batchSize, epochs=self._epochs, validation_split=0.3, callbacks=[history, checkpointer])
        
    def predict(self, images, questions, weights):
        self._model.load_weights(weights)
        prediction = self._model.predict(x=[images, questions], batch_size=self._batchSize)
        return prediction

    def predict_current_state(self, images, questions):
        prediction = self._model.predict(x=[images, questions], batch_size=self._batchSize)
        return prediction
