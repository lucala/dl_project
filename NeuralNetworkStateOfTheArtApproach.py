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
    def __init__(self, n, m, r, dic_size, glove_weights, l, k=16, epochs=5, batchSize=128, loss='sparse_categorical_crossentropy', checkpoint_period=1):
        self._n = n  # number of samples
        self._m = m  # number of image features
        self._r = r  # number of word features
        self._dic_size = dic_size # size of the dictionary
        self._l = l  # number of possible labels
        self._k = k#number of image locations
        self._epochs = epochs
        self._batchSize = batchSize
        self._wordEmbeddingSize = 14
        #if (self._wordEmbeddingSize % self._r != 0):
        #    print("Dimension mismatch: number of words in a question has to divide embedding size")
        self._loss = loss
        self._checkpoint_period = checkpoint_period

        image_input = Input(shape=(self._k,self._m,),name="image_input")
        word_input = Input(shape=(self._r,),name="word_input")
        word_embedding = Embedding(self._dic_size, self._wordEmbeddingSize, input_length=self._r, weights=[glove_weights], trainable=False, name='word_embedding')(word_input)
        #word_embedding = Dense(self._wordEmbeddingSize, input_shape=(self._r,),name='word_embedding')(word_input)
        gru_1 = GRU(512)(word_embedding)
        
        repeat_question = RepeatVector(self._k)(gru_1)
        concat_layer = Concatenate()([repeat_question, image_input])
        gated_tanh_1_1 = Dense(512, activation='tanh')(concat_layer)
        gated_tanh_1_2 = Dense(512, activation='sigmoid')(concat_layer)
        gated_tanh_1_3 = Multiply()([gated_tanh_1_1, gated_tanh_1_2])
        relu_1 = Dense(1, activation='relu')(gated_tanh_1_3)
        reshape_1 = Flatten()([relu_1])#Reshape((self._n, self._k))(relu_1)
        softmax_1 = Dense(self._k, activation='softmax')(reshape_1)

        weighted_sum = Dot(axes=1)([image_input, softmax_1]) #need to check this, if dimensions are correct, should be self._m
        gated_tanh_2_1 = Dense(512, activation='tanh')(weighted_sum)
        gated_tanh_2_2 = Dense(512, activation='sigmoid')(weighted_sum)
        gated_tanh_2_3 = Multiply()([gated_tanh_2_1, gated_tanh_2_2])
        
        gated_tanh_3_1 = Dense(512, activation='tanh')(gru_1)
        gated_tanh_3_2 = Dense(512, activation='sigmoid')(gru_1)
        gated_tanh_3_3 = Multiply()([gated_tanh_3_1, gated_tanh_3_2])
        
        multiply_1 = Multiply()([gated_tanh_3_3, gated_tanh_2_3])
        
        #text_based
        gated_tanh_4_1 = Dense(self._r, activation='tanh')(multiply_1)
        gated_tanh_4_2 = Dense(self._r, activation='sigmoid')(multiply_1)
        gated_tanh_4_3 = Multiply()([gated_tanh_4_1, gated_tanh_4_2])
        relu_2 = Dense(int(self._l), activation='relu')(gated_tanh_4_3)
        
        #image_based
        gated_tanh_5_1 = Dense(self._m, activation='tanh')(multiply_1)
        gated_tanh_5_2 = Dense(self._m, activation='sigmoid')(multiply_1)
        gated_tanh_5_3 = Multiply()([gated_tanh_5_1, gated_tanh_5_2])
        relu_3 = Dense(int(self._l), activation='relu')(gated_tanh_5_3)
        
        add_1 = Add()([relu_2, relu_3])
        
        output_layer = Dense(int(self._l), activation='sigmoid', name='output_layer')(add_1)

        self._model = Model(inputs=[image_input, word_input], outputs=[output_layer])
        print(self._model.summary())
        #adam = optimizers.adam(lr=0.0005)
        self._model.compile(loss=self._loss, optimizer='adadelta', metrics=['accuracy'])

    def get_model_summary(self):
        return self._model.summary()

    def fit(self, images, questions, labels):
        # Split the data
        # self._matrix = matrix
        self._labels = labels
        checkpointer = ModelCheckpoint(filepath="weights/weights-{epoch:02d}-{val_loss:.4f}.hdf5", verbose=1, save_best_only=True, monitor='val_acc', period=self._checkpoint_period)
        history = History()
        #reduceLR = ReduceLROnPlateau(patience=5, monitor='val_loss', factor=0.8)
        
        print(images.shape, questions.shape, self._labels.shape)            
        self._model.fit(x=[images, questions],y=self._labels, batch_size=self._batchSize, epochs=self._epochs, validation_split=0.3, callbacks=[history, checkpointer])#, reduceLR])
        
    def predict(self, images, questions, weights):
        self._model.load_weights(weights)
        prediction = self._model.predict(x=[images, questions], batch_size=self._batchSize)
        return prediction

    def predict_current_state(self, images, questions):
        prediction = self._model.predict(x=[images, questions], batch_size=self._batchSize)
        return prediction
