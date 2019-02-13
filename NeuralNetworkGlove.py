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
from keras.layers import Dense, Activation, Embedding, Merge, Flatten, Concatenate, Reshape, LSTM, GRU
from keras.layers.core import Lambda
from keras.callbacks import ModelCheckpoint, History
from keras import optimizers
import keras

class NeuralNetwork(object):
    """This class creates a neural network model according to the _init_ parameters"""
    # Loss is either categorical_crossentropy (labels should have size [batchSize, classes])
    # or sparse_categorical_crossentropy (labels have size [batchSize, 1] with elements in [0, l - 1])
    def __init__(self, n, m, r, l, dic_size, w, objects = 80, epochs=5, batchSize=1024, loss='sparse_categorical_crossentropy', activation = 'softmax', lr = 0.0001):
        self._n = n  # number of samples
        self._m = m  # number of image features
        self._r = r  # number of word features
        self._l = l  # number of possible labels
        self._epochs = epochs
        self._batchSize = batchSize
        self._loss = loss
        self._dic_size = dic_size

        image_input = Input(shape=(self._m,),name="image_input")
        object_input = Input(shape=(objects,),name="object_input")
        word_input = Input(shape=(self._r,),name="word_input")
        word_embedding = Embedding(self._dic_size, 300, input_length=self._r, weights=[w], trainable=True, name='word_embedding')(word_input)
        gru = GRU(1024)(word_embedding)
        merge = keras.layers.concatenate([image_input, gru])
        #hidden_layer1 = Dense(64, activation='relu')(merge) #TODO nr of layers and units still needs to be extracted from baseline!
        #hidden_layer2 = Dense(64, activation='relu')(hidden_layer1)
        #hidden_layer3 = Dense(64, activation='relu')(hidden_layer2)
        output_layer = Dense(self._l, activation=activation, name='output_layer')(merge)
        adam = optimizers.adam(lr=lr)
        self._model = Model(inputs=[image_input, object_input, word_input], outputs=[output_layer])
        self._model.compile(loss=self._loss, optimizer=adam, metrics=['accuracy']) #TODO what loss?

    def get_model_summary(self):
        return self._model.summary()

    def fit(self, image_features, objects, questions, labels, checkpoint_period=10, monitor='val_acc'):
        # Split the data
        self._labels = labels
        checkpointer = ModelCheckpoint(filepath="weights/weights-{epoch:02d}-{val_loss:.4f}.hdf5", verbose=1, period=checkpoint_period, save_best_only=True, monitor=monitor)
        history = History()
        
        # print(self._matrix[:,0:self._m].shape, self._matrix[:,self._m:].shape, self._labels.shape)
        # Define input dictionary
        dic = {"image_input": image_features, "object_input": objects, "word_input": questions}            
        self._model.fit(x=dic,y=self._labels, batch_size=self._batchSize, epochs=self._epochs, validation_split=0.3, callbacks=[history, checkpointer])
        
    def predict(self, image_features, objects, questions, weights):
        self._model.load_weights(weights)
        dic = {"image_input": image_features, "object_input": objects, "word_input": questions}
        prediction = self._model.predict(x=dic, batch_size=self._batchSize)
        return prediction

    def predict_current_state(self, image_features, objects, questions):
        dic = {"image_input": image_features, "object_input": objects, "word_input": questions}
        prediction = self._model.predict(x=dic, batch_size=self._batchSize)
        return prediction

# Load data
from PrepareOriginalData import PrepareData
from ExtractObjects import ExtractObjects
from EvaluateModel import ProduceResult
import numpy as np

# Some constants
taskType = 'all'
data_amount = 1
epochs = 50

# Load training set
p = PrepareData(path_images='data_vqa_feat', # Path to image features 
                subset='trainval', # Desired subset: either train2014 or val2014
                taskType='OpenEnded', # 'OpenEnded', 'MultipleChoice', 'all'
                cut_data=data_amount, # Percentage of data to use, 1 = All values, above 1=#samples for debugging
                output_path='data', # Path where we want to output temporary data
                pad_length=14, # Number of words in a question (zero padded)
                question_threshold=0, answer_threshold=3, # Keep only most common words
                questions_sparse=True, answers_sparse=True)
image_features, questions, answers, annotations = p.load_data()
print("Image features", image_features.shape)
print("Question features", questions.shape)
print("Answers", answers.shape)
print("Dictionary size", p.dic_size)
p.dumpDictionary()

# Load object features
# Consider three thresholds
"""
eo = ExtractObjects(cut_data=data_amount, output_fileName='objects_trainval.txt', subset='train2014', threshold=25)
object_matrix1 = eo.onehotvector(annotations)
eo = ExtractObjects(cut_data=data_amount, output_fileName='objects_trainval.txt', subset='train2014', threshold=50)
object_matrix2 = eo.onehotvector(annotations)
eo = ExtractObjects(cut_data=data_amount, output_fileName='objects_trainval.txt', subset='train2014', threshold=75)
object_matrix3 = eo.onehotvector(annotations)

object_matrix = np.concatenate([object_matrix1, object_matrix2, object_matrix3], axis=1)
# np.save('data/object_matrix_trainval.npy', object_matrix)
"""
object_matrix = np.load('data/object_matrix_trainval.npy')
print(object_matrix.shape)

# Load Glove embedding
# load the whole embedding into memory
embeddings_index = dict()
f = open('data/glove.6B.300d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))

# Create an embedding matrix
embedding_matrix = np.zeros((np.max(questions) + 1, 300)) # Ignore thresholded words
for word in p._question_dict.keys():
    # If word was thresholded ignore it
    if p._question_dict[word] != 0:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[p._question_dict[word]] = embedding_vector
print(embedding_matrix.shape)

# Define neural network
# neuralnet = NeuralNetwork(image_features.shape[0],1024,questions.shape[1],p.dic_size,np.max(answers)+1, objects = 240, checkpoint_period=10, lr=0.0001,
#                          epochs = epochs, batchSize=512, activation='softmax', loss='sparse_categorical_crossentropy', monitor='val_acc')
neuralnet = NeuralNetwork(image_features.shape[0], 1024, questions.shape[1], np.max(answers)+1, np.max(questions) + 1, embedding_matrix, 
                          objects= 240, epochs = epochs, lr=0.001, batchSize=512)
neuralnet.get_model_summary()

# Fine tune
# neuralnet.load_weights('weights/weights-69-4.0170.hdf5')

# Train network
# neuralnet.fit(image_features, object_matrix, questions, answers)

# Load validation set and evaluate prediction on it
image_features = questions = answers = annotations = []
pt= PrepareData(path_images='data_vqa_feat', # Path to image features 
                subset='val2014', # Desired subset: either train2014 or val2014
                taskType=taskType, # 'OpenEnded', 'MultipleChoice', 'all'
                cut_data=data_amount, # Percentage of data to use, 1 = All values, above 1 = 10 samples for debugging
                output_path='data')
pt.loadDictionary('data/dictionary.pkl') # Use same dictionary as in training
image_features, questions, answers, annotations = pt.load_data()
print("Image features", image_features.shape)
print("Question features", questions.shape)
print("Answers", answers.shape)
print("Dictionary size", pt.dic_size)

# Extract objects
# Consider three thresholds

eo = ExtractObjects(cut_data=data_amount, output_fileName='objects_val.txt', subset='val2014', threshold=25)
object_matrix1 = eo.onehotvector(annotations)
eo = ExtractObjects(cut_data=data_amount, output_fileName='objects_val.txt', subset='val2014', threshold=50)
object_matrix2 = eo.onehotvector(annotations)
eo = ExtractObjects(cut_data=data_amount, output_fileName='objects_val.txt', subset='val2014', threshold=75)
object_matrix3 = eo.onehotvector(annotations)

object_matrix = np.concatenate([object_matrix1, object_matrix2, object_matrix3], axis=1)

# Test prediction on validation set
pred = neuralnet.predict(image_features, object_matrix, questions, 'weights/weights-19-4.9949.hdf5')
# pred = neuralnet.predict_current_state(image_features, object_matrix, questions)
print(pred.shape)

# Evaluate model
print("=== Results on validation set ===")
image_features = questions = answers = annotations = []
model_evaluator = ProduceResult(pt._int_to_answer, pt._answer_to_int, dataSubType='val2014')
answers = model_evaluator.produce_results(pred, pt._original_questions)
model_evaluator.evaluate(taskType=taskType)

# Predict on test set
print("=== Predicting on test set ===")
pred = []
image_features, questions = pt.load_test_set(set_name='test-dev2015')
print("Image features", image_features.shape)
print("Question features", questions.shape)

# Extract objects
# Consider three thresholds
"""
eo = ExtractObjects(cut_data=data_amount, output_fileName='objects_test.txt', subset='test-dev2015', threshold=25)
object_matrix1 = eo.onehotvector(pt._original_questions)
eo = ExtractObjects(cut_data=data_amount, output_fileName='objects_test.txt', subset='test-dev2015', threshold=50)
object_matrix2 = eo.onehotvector(pt._original_questions)
eo = ExtractObjects(cut_data=data_amount, output_fileName='objects_test.txt', subset='test-dev2015', threshold=75)
object_matrix3 = eo.onehotvector(pt._original_questions)

object_matrix = np.concatenate([object_matrix1, object_matrix2, object_matrix3], axis=1)
np.save('data/object_matrix_test.npy', object_matrix)
print(object_matrix.shape)
"""
object_matrix = np.load('data/object_matrix_test.npy')

# Test prediction on validation set
# pred = neuralnet.predict(image_features, object_matrix, questions, 'weights/weights-05-4.9527.hdf5')
pred = neuralnet.predict_current_state(image_features, object_matrix, questions)
print(pred.shape)

# Evaluate model
image_features = questions =  []
model_evaluator = ProduceResult(pt._int_to_answer, pt._answer_to_int, dataSubType='test-dev2015')
answers = model_evaluator.produce_results(pred, pt._original_questions)
