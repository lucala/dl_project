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
from keras.callbacks import ModelCheckpoint, History, ReduceLROnPlateau
from keras import optimizers
import keras

class NeuralNetwork(object):
    """This class creates a neural network model according to the _init_ parameters"""
    # Loss is either categorical_crossentropy (labels should have size [batchSize, classes])
    # or sparse_categorical_crossentropy (labels have size [batchSize, 1] with elements in [0, l - 1])
    def __init__(self, n, dic_size, l, epochs=5, batchSize=128, loss='sparse_categorical_crossentropy', checkpoint_period=1):
        self._n = n  # number of samples
        self._dic_size = dic_size # size of the dictionary
        self._l = l  # number of possible labels
        self._epochs = epochs
        self._batchSize = batchSize
        self._wordEmbeddingSize = 2048
        if (self._wordEmbeddingSize % self._n != 0):
            print("Dimension mismatch: number of words in a question has to divide embedding size")
        self._loss = loss
        self._checkpoint_period = checkpoint_period

        word_input = Input(shape=(self._n,),name="word_input")
        word_embedding = Embedding(self._dic_size, int(self._wordEmbeddingSize / self._n), input_length=self._n, name='word_embedding')(word_input)
        flatten_embedding = Flatten(name="flatten_embedding")(word_embedding)
        hidden_layer1 = Dense(1024, activation='relu')(flatten_embedding)
        hidden_layer2 = Dense(512, activation='relu')(hidden_layer1)
        hidden_layer3 = Dense(256, activation='relu')(hidden_layer2)
        hidden_layer4 = Dense(128, activation='relu')(hidden_layer3)
        hidden_layer5 = Dense(64, activation='relu')(hidden_layer4)
        hidden_layer6 = Dense(32, activation='relu')(hidden_layer5)
        hidden_layer7 = Dense(8, activation='relu')(hidden_layer6)
        output_layer = Dense(self._l, activation='softmax', name='output_layer')(hidden_layer7)

        self._model = Model(inputs=[word_input], outputs=[output_layer])
        print(self._model.summary())
        self._model.compile(loss=self._loss, optimizer='adam', metrics=['accuracy'])

    def get_model_summary(self):
        return self._model.summary()

    def fit(self, questions, labels):
        # Split the data
        # self._matrix = matrix
        self._labels = labels
        checkpointer = ModelCheckpoint(filepath="weights/weights-{epoch:02d}-{val_loss:.4f}.hdf5", verbose=1, save_best_only=True, monitor='val_acc', period=self._checkpoint_period)
        history = History()
        reduceLR = ReduceLROnPlateau(patience=6, monitor='val_acc', factor=0.8)
        
        print(questions.shape, self._labels.shape)            
        self._model.fit(x=[questions],y=self._labels, batch_size=self._batchSize, epochs=self._epochs, validation_split=0.3, callbacks=[history, checkpointer])#, reduceLR])
        
    def predict(self, questions, weights):
        self._model.load_weights(weights)
        prediction = self._model.predict(x=[questions], batch_size=self._batchSize)
        return np.argmax(prediction, axis=1)

    def predict_current_state(self, questions):
        prediction = self._model.predict(x=[questions], batch_size=self._batchSize)
        return np.argmax(prediction, axis=1)
