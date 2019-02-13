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
from keras.layers import Dense, Activation, Embedding, Merge, Flatten, Concatenate, Reshape, LSTM, Conv2D, MaxPooling2D
from keras.layers.core import Lambda
from keras.callbacks import ModelCheckpoint, History, EarlyStopping, ReduceLROnPlateau
from keras import optimizers
import keras
from scipy.misc import imread, imresize
# from keras.utils.training_utils import multi_gpu_model

class NeuralNetwork(object):
    """This class creates a neural network model according to the _init_ parameters"""
    # Loss is either categorical_crossentropy (labels should have size [batchSize, classes])
    # or sparse_categorical_crossentropy (labels have size [batchSize, 1] with elements in [0, l - 1])
    def __init__(self, n, m, r, dic_size, l, epochs=5, batchSize=32, bulkMultiplier=64, loss='categorical_crossentropy'):
        self._n = n  # number of samples
        self._m = m  # number of image features
        self._r = r  # number of word features
        self._dic_size = dic_size # size of the dictionary
        self._l = l  # number of possible labels
        self._epochs = epochs
        self._batchSize = batchSize
        self._wordEmbeddingSize = 1024
        self._bulk = bulkMultiplier
        if (self._wordEmbeddingSize % self._r != 0):
            print("Dimension mismatch: number of words in a question has to divide embedding size")
        self._loss = loss


        vision_model = Sequential()
        vision_model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(224, 224, 3)))
        vision_model.add(Conv2D(64, (3, 3), activation='relu'))
        vision_model.add(MaxPooling2D(pool_size=2, strides=2))
        vision_model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        vision_model.add(Conv2D(128, (3, 3), activation='relu'))
        vision_model.add(MaxPooling2D(pool_size=2, strides=2))
        vision_model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        vision_model.add(Conv2D(256, (3, 3), activation='relu'))
        vision_model.add(Conv2D(256, (3, 3), activation='relu'))
        vision_model.add(MaxPooling2D(pool_size=2, strides=2))
        vision_model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        vision_model.add(Conv2D(256, (3, 3), activation='relu'))
        vision_model.add(Conv2D(256, (3, 3), activation='relu'))
        vision_model.add(MaxPooling2D(pool_size=2, strides=2))
        vision_model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        vision_model.add(Conv2D(256, (3, 3), activation='relu'))
        vision_model.add(Conv2D(256, (3, 3), activation='relu'))
        vision_model.add(MaxPooling2D(pool_size=2, strides=2))
        vision_model.add(Flatten())


        image_input = Input(shape=(224, 224, 3),name="image_input")
        encoded_image = vision_model(image_input)
        word_input = Input(shape=(self._r,),name="word_input")
        word_embedding = Embedding(self._dic_size, int(self._wordEmbeddingSize / self._r), input_length=self._r, name='word_embedding')(word_input)
        flatten_embedding = Flatten(name="flatten_embedding")(word_embedding)

        merge = keras.layers.concatenate([encoded_image, flatten_embedding])
        hidden_layer1 = Dense(4096, activation='relu')(merge) #TODO nr of layers and units still needs to be extracted from baseline!
        #hidden_layer2 = Dense(4096, activation='relu')(hidden_layer1)
        #hidden_layer3 = Dense(4096, activation='relu')(hidden_layer2)
        output_layer = Dense(self._l, activation='softmax', name='output_layer')(hidden_layer1)

        self._model = Model(inputs=[image_input, word_input], outputs=[output_layer])
        #self._model = multi_gpu_model(self._model, gpus=2)
        print(self._model.summary())
        self._model.compile(loss=self._loss, optimizer='adam', metrics=['accuracy']) #TODO what loss?

    def get_model_summary(self):
        return self._model.summary()

    def fit(self, images, questions, labels, mean_dataset):
        # Split the data
        #self._matrix = matrix
        self._labels = labels
        checkpointer = ModelCheckpoint(filepath="weights/weights-{epoch:02d}-{val_loss:.4f}.hdf5", verbose=1, save_best_only=True)
        history = History()
        earlyStopping = EarlyStopping(patience=1)
        reduceLR = ReduceLROnPlateau()

        #print(self._matrix[:,0:self._m].shape, self._matrix[:,self._m:].shape, self._labels.shape)
        # Define input dictionary
        #dic = {"image_input": images, "word_input": questions}
        print(images.shape)
        if images.shape[1] > 1:
            print("input are images")
            self._model.fit(x=[images, questions],y=self._labels, batch_size=self._batchSize, epochs=self._epochs, validation_split=0.3, callbacks=[history, earlyStopping, reduceLR])#checkpointer,

        else:
            print("input is image stream")
            bulk = self._bulk*self._batchSize
            lenImgs = len(images)
            for it in range(0,lenImgs,bulk):
                if it+bulk < lenImgs:
                    image_features = np.empty((bulk,224,224,3))
                    imagesBulk = images[it:it+bulk]
                    question_features = questions[it:it+bulk]
                    label_features = labels[it:it+bulk]
                    for bulkIterator in range(bulk):
                        tmp = imresize(imread(imagesBulk[bulkIterator][0], mode='RGB'), (224, 224)).astype(np.float32) / 255 - mean_dataset
                        image_features[bulkIterator] = tmp
                else: #fill up rest images
                    imagesBulk = images[it:]
                    image_features = np.empty((imagesBulk.shape[0],224,224,3))
                    question_features = questions[it:]
                    label_features = labels[it:]
                    for bulkIterator in range(len(imagesBulk)):
                        tmp = imresize(imread(imagesBulk[bulkIterator][0], mode='RGB'), (224, 224)).astype(np.float32) / 255 - mean_dataset
                        image_features[bulkIterator] = tmp

                #image_features = np.asarray(image_features)

                self._model.fit(x=[image_features, question_features],y=label_features, batch_size=self._batchSize, epochs=self._epochs, validation_split=0.3, callbacks=[history, earlyStopping, reduceLR])#, checkpointer])

        self._model.save_weights('weights/cnn_fully_trained.hdf5')

    def predict(self, images, questions, weights, mean_dataset):
        self._model.load_weights(weights)

        print("input is image stream")
        bulk = self._bulk*self._batchSize
        lenImgs = len(images)
        for it in range(0,lenImgs,bulk):
            image_features = []
            if it+bulk < lenImgs:
                imagesBulk = images[it:it+bulk]
                question_features = questions[it:it+bulk]
                for bulkIterator in range(bulk):
                    tmp = imresize(imread(imagesBulk[bulkIterator][0], mode='RGB'), (224, 224)).astype(np.float32) / 255 - mean_dataset
                    image_features.append(tmp)
            else: #fill up rest images
                imagesBulk = images[it:]
                question_features = questions[it:]
                for bulkIterator in range(len(imagesBulk)):
                    tmp = imresize(imread(imagesBulk[bulkIterator][0], mode='RGB'), (224, 224)).astype(np.float32) / 255 - mean_dataset
                    image_features.append(tmp)

            image_features = np.asarray(image_features)

        prediction = self._model.predict(x=[image_features, question_features], batch_size=self._batchSize)
        return prediction
