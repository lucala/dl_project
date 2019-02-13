from __future__ import division
import numpy as np
from numpy import newaxis
from keras.models import Model
from keras.layers import Input, Dense
from sklearn.model_selection import KFold
import keras.backend as K
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.layers import Dropout
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, Flatten, Reshape, Concatenate, LSTM, Dropout, GRU, Multiply, Add, RepeatVector, Dot
from keras.layers.core import Lambda
from keras.callbacks import ModelCheckpoint, History, ReduceLROnPlateau
from keras import optimizers
import keras

class NeuralNetwork(object):
    """This class creates a neural network model according to the _init_ parameters"""
    # Loss is either categorical_crossentropy (labels should have size [batchSize, classes])
    # or sparse_categorical_crossentropy (labels have size [batchSize, 1] with elements in [0, l - 1])
    def __init__(self, n, m, r, l, objects=240, k=16, epochs=5, batchSize=128, loss='sparse_categorical_crossentropy',
                 activation='softmax', checkpoint_period=1, monitor='val_loss', lr=0.001):
        self._n = n  # number of samples
        self._m = m  # number of image features
        self._r = r  # number of word features
        self._l = l  # number of possible labels
        self._k = k#number of image locations
        self._epochs = epochs
        self._batchSize = batchSize
        self._loss = loss
        self._checkpoint_period = checkpoint_period
        self._monitor = monitor

        image_input = Input(shape=(self._k,self._m,),name="image_input")
        object_input = Input(shape=(objects,),name='object_input')
        word_input = Input(shape=(self._r,),name="word_input")
        word_embedding = Dense(1024, input_shape=(self._r,),name='word_embedding')(word_input)
        
        repeat_question = RepeatVector(self._k)(word_embedding)
        concat_layer = Concatenate()([repeat_question, image_input])
        gated_tanh_1_1 = Dense(512, activation='tanh')(concat_layer)
        gated_tanh_1_2 = Dense(512, activation='sigmoid')(concat_layer)
        gated_tanh_1_3 = Multiply()([gated_tanh_1_1, gated_tanh_1_2])
        softmax_1 = Dense(1, activation='softmax')(gated_tanh_1_3)
        # reshape_1 = Flatten()([relu_1])#Reshape((self._n, self._k))(relu_1)
        # softmax_1 = Dense(self._k, activation='softmax')(reshape_1)

        weighted_sum = Flatten()(Dot(axes=1)([image_input, softmax_1]))
        # gated_tanh_2_1 = Dense(512, activation='tanh')(weighted_sum)
        # gated_tanh_2_2 = Dense(512, activation='sigmoid')(weighted_sum)
        # gated_tanh_2_3 = Multiply()([gated_tanh_2_1, gated_tanh_2_2])
        
        merge = Concatenate()([word_embedding, object_input, weighted_sum])
        
        hidden_layer1 = Dense(4096, activation='relu')(merge)
        drop1 = Dropout(0.5)(hidden_layer1)
        
        output_layer = Dense(self._l, activation=activation, name='output_layer')(drop1)

        self._model = Model(inputs=[image_input, object_input, word_input], outputs=[output_layer])

        adam = optimizers.adam(lr=lr)
        self._model.compile(loss=self._loss, optimizer=adam, metrics=['accuracy'])

    def get_model_summary(self):
        return self._model.summary()

    def fit(self, images, objects, questions, labels, validation_split=0.3):
        # Split the data
        # self._matrix = matrix
        self._labels = labels
        checkpointer = ModelCheckpoint(filepath="weights/weights-{epoch:02d}.hdf5", verbose=1, save_best_only=False,
                                       monitor=self._monitor,period=self._checkpoint_period)
        history = History()
        reduceLR = ReduceLROnPlateau(patience=5, monitor='loss', factor=0.9)
        
        print(images.shape, questions.shape, self._labels.shape)            
        self._model.fit(x=[images, objects, questions],y=self._labels, batch_size=self._batchSize, epochs=self._epochs, validation_split=validation_split,
                        callbacks=[history, checkpointer, reduceLR])
        
    def predict(self, images, objects, questions, weights):
        self.load_weights(weights)
        prediction = self._model.predict(x=[images, objects, questions], batch_size=self._batchSize)
        return prediction

    def predict_current_state(self, images, objects, questions):
        prediction = self._model.predict(x=[images, objects, questions], batch_size=self._batchSize)
        return prediction

    def load_weights(self, weights):
        self._model.load_weights(weights)

