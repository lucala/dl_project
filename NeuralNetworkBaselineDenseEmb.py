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
from keras import regularizers
import keras

class NeuralNetwork(object):
    """This class creates a neural network model according to the _init_ parameters"""
    # Loss is either categorical_crossentropy (labels should have size [batchSize, classes])
    # or sparse_categorical_crossentropy (labels have size [batchSize, 1] with elements in [0, l - 1])
    def __init__(self, n, m, r, l, objects = 80, epochs=5, batchSize=1024, loss='sparse_categorical_crossentropy', activation = 'softmax', lr = 0.0001):
        self._n = n  # number of samples
        self._m = m  # number of image features
        self._r = r  # number of word features
        self._l = l  # number of possible labels
        self._epochs = epochs
        self._batchSize = batchSize
        self._wordEmbeddingSize = 1024
        self._loss = loss

        image_input = Input(shape=(self._m,),name="image_input")
        object_input = Input(shape=(objects,),name="object_input")
        word_input = Input(shape=(self._r,),name="word_input")
        word_embedding = Dense(self._wordEmbeddingSize, input_shape=(self._r,),name='word_embedding')(word_input)
        merge = keras.layers.concatenate([image_input, object_input, word_embedding])
        hidden_layer1 = Dense(4096, activation='relu')(merge)
        drop1 = Dropout(0.5)(hidden_layer1)
        """
        hidden_layer2 = Dense(4096, activation='relu', kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l2(0.01))(drop1)
        drop2 = Dropout(0.5)(hidden_layer2)
        hidden_layer3 = Dense(4096, activation='relu')(drop1)
        """
        output_layer = Dense(self._l, activation=activation, name='output_layer')(drop1)
        adam = optimizers.adam(lr=lr)
        self._model = Model(inputs=[image_input, object_input, word_input], outputs=[output_layer])
        self._model.compile(loss=self._loss, optimizer=adam, metrics=['accuracy']) 

    def get_model_summary(self):
        return self._model.summary()

    def fit(self, image_features, objects, questions, labels, checkpoint_period=10, monitor='val_acc', validation_split=0.3):
        # Split the data
        self._labels = labels
        checkpointer = ModelCheckpoint(filepath="weights/weights-{epoch:02d}.hdf5", verbose=1, period=checkpoint_period, save_best_only=False, monitor=monitor)
        history = History()
        
        # print(self._matrix[:,0:self._m].shape, self._matrix[:,self._m:].shape, self._labels.shape)
        # Define input dictionary
        dic = {"image_input": image_features, "object_input": objects, "word_input": questions}            
        self._model.fit(x=dic,y=self._labels, batch_size=self._batchSize, epochs=self._epochs, validation_split=validation_split, callbacks=[history, checkpointer])
        
    def predict(self, image_features, objects, questions, weights):
        self._model.load_weights(weights)
        dic = {"image_input": image_features, "object_input": objects, "word_input": questions}
        prediction = self._model.predict(x=dic, batch_size=self._batchSize)
        return prediction

    def predict_current_state(self, image_features, objects, questions):
        dic = {"image_input": image_features, "object_input": objects, "word_input": questions}
        prediction = self._model.predict(x=dic, batch_size=self._batchSize)
        return prediction

    def load_weights(self, weights):
        self._model.load_weights(weights)
