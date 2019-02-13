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
from keras.layers import Dense, Activation, Embedding, Merge, Flatten, Concatenate, Reshape, LSTM, Dropout
from keras.layers.core import Lambda
from keras.callbacks import ModelCheckpoint, History, ReduceLROnPlateau
from keras import optimizers
import keras

class NeuralNetwork(object):
    """This class creates a neural network model according to the _init_ parameters"""
    # Loss is either categorical_crossentropy (labels should have size [batchSize, classes])
    # or sparse_categorical_crossentropy (labels have size [batchSize, 1] with elements in [0, l - 1])
    def __init__(self, n, m, r, dic_size, l, objects= 80, epochs=5, batchSize=128, loss='sparse_categorical_crossentropy', checkpoint_period=1, lr=0.0001):
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
        self._loss = loss
        self._checkpoint_period = checkpoint_period

        image_input = Input(shape=(self._m,),name="image_input")
        object_input = Input(shape=(objects,),name="object_input")
        word_input = Input(shape=(self._r,),name="word_input")
        word_embedding = Embedding(self._dic_size, int(self._wordEmbeddingSize / self._r), input_length=self._r, name='word_embedding')(word_input)
        flatten_embedding = Flatten(name="flatten_embedding")(word_embedding)
        merge = keras.layers.concatenate([image_input, flatten_embedding])
        hidden_layer0 = Dense(2048, activation='relu')(merge)
        dropout_1 = Dropout(0.5)(hidden_layer0)
        hidden_layer4 = Dense(1024, activation='relu')(dropout_1)
        dropout_2 = Dropout(0.5)(hidden_layer4)
        hidden_layer5 = Dense(512, activation='relu')(dropout_2)
        dropout_3 = Dropout(0.5)(hidden_layer5)
        hidden_layer6 = Dense(256, activation='relu')(dropout_3)
        hidden_layer7 = Dense(128, activation='relu')(hidden_layer6)
        #hidden_layer8 = Dense(4, activation='relu')(hidden_layer7)
        output_layer = Dense(self._l, activation='sigmoid', name='output_layer')(hidden_layer7)#use sigmoid for binary, softmax for c.entr.

        self._model = Model(inputs=[image_input, object_input, word_input], outputs=[output_layer])
        print(self._model.summary())
        adam = optimizers.adam(lr=lr)
        self._model.compile(loss=self._loss, optimizer=adam, metrics=['accuracy'])

    def get_model_summary(self):
        return self._model.summary()

    def fit(self, images, objects, questions, labels):
        # Split the data
        # self._matrix = matrix
        self._labels = labels
        checkpointer = ModelCheckpoint(filepath="weights/weights-{epoch:02d}-{val_loss:.4f}.hdf5", verbose=1, save_best_only=True, monitor='val_acc', period=self._checkpoint_period)
        history = History()
        reduceLR = ReduceLROnPlateau(patience=4, monitor='val_loss', factor=0.95)
        
        print(images.shape, questions.shape, self._labels.shape)            
        self._model.fit(x=[images, objects, questions],y=self._labels, batch_size=self._batchSize, epochs=self._epochs, validation_split=0.3, callbacks=[history, checkpointer, reduceLR])
        
    def predict(self, images, objects, questions, weights):
        self._model.load_weights(weights)
        prediction = self._model.predict(x=[images, objects, questions], batch_size=self._batchSize)
        return prediction

    def predict_current_state(self, images, objects, questions):
        prediction = self._model.predict(x=[images, objects, questions], batch_size=self._batchSize)
        return prediction


from PrepareDataSoftLabels import PrepareData # Got 78.24 without any hack
import numpy as np

# Some constants
taskType = 'all'
data_amount = 1
epochs = 50

# Train on only one question type
question_type = 'yes/no'

# Load training set
p = PrepareData(path_images='data_vqa_feat', # Path to image features 
                subset='train2014', # Desired subset: either train2014 or val2014
                taskType='OpenEnded', # 'OpenEnded', 'MultipleChoice', 'all'
                cut_data=data_amount, # Percentage of data to use, 1 = All values, above 1=#samples for debugging
                output_path='data', # Path where we want to output temporary data
                pad_length=32, # Number of words in a question (zero padded)
                question_threshold=0, answer_threshold=10, # Keep only most common words
                questions_sparse=True, answer_type=question_type)
image_features, questions, answers, annotations = p.load_data()
print("Image features", image_features.shape)
print("Question features", questions.shape)
print("Answers", answers.shape)
print("Dictionary size", p.dic_size)
print("Number of possible classes", np.max(answers) + 1)

# Transform to probabilities
print(answers)

# Save dictionary
p.dumpDictionary('dictionary_yes_no')

# Extract object features
from ExtractObjects import ExtractObjects
"""
# Consider three thresholds
eo = ExtractObjects(cut_data=data_amount, output_fileName='objects_train.txt', subset='train2014', threshold=25)
object_matrix1 = eo.onehotvector(annotations)
eo = ExtractObjects(cut_data=data_amount, output_fileName='objects_train.txt', subset='train2014', threshold=50)
object_matrix2 = eo.onehotvector(annotations)
eo = ExtractObjects(cut_data=data_amount, output_fileName='objects_train.txt', subset='train2014', threshold=75)
object_matrix3 = eo.onehotvector(annotations)

object_matrix = np.concatenate([object_matrix1, object_matrix2, object_matrix3], axis = 1)
print(object_matrix.shape)
np.save('data/object_matrix_train_yesno.npy', object_matrix)
"""
object_matrix = np.load('data/object_matrix_train_yesno.npy')

# Use this when using sparse representation np.max(answers)+1
neuralnet = NeuralNetwork(image_features.shape[0],1024,questions.shape[1],p.dic_size, 2,
                          epochs = epochs, batchSize=512, loss='binary_crossentropy', lr=0.001, objects=240)

# Train network
# neuralnet.fit(image_features, object_matrix, questions, answers)

#see what it predicts on training data, on which it trained!
# pred = neuralnet.predict_current_state(image_features, object_matrix, questions)
pred = neuralnet.predict(image_features, object_matrix, questions, 'weights/weights-11-0.5826.hdf5')
# pred = np.round(pred)
print(pred)

# One hot encode
"""
predO = np.zeros((pred.shape[0], 2))
for i in range(pred.shape[0]):
    predO[i, int(pred[i, 0])] = 1
print(predO)
"""
from EvaluateModel import ProduceResult
model_evaluator = ProduceResult(p._int_to_answer, p._answer_to_int, dataSubType='train2014')
answers = model_evaluator.produce_results(pred, p._original_questions)
model_evaluator.evaluate(taskType='OpenEnded')

image_features = questions = answers = annotations = []
question_type = 'yes/no'
# Load validation set and evaluate prediction on it
pt= PrepareData(path_images='data_vqa_feat', # Path to image features 
                        subset='val2014', # Desired subset: either train2014 or val2014
                        taskType=taskType, # 'OpenEnded', 'MultipleChoice', 'all'
                        cut_data=data_amount, # Percentage of data to use, 1 = All values, above 1 = 10 samples for debugging
                        output_path='data', # Path where we want to output temporary data
                        pad_length=32, # Number of words in a question (zero padded)
                        question_threshold=0, answer_threshold=10, # Keep only most common words
                        questions_sparse=True, answer_type=question_type,
                        precomputed_dic=p._question_dict)
pt.loadDictionary('data/dictionary_yes_no.pkl') # Use same dictionary as in training
image_features, questions, answers, annotations = pt.load_data()
print("Image features", image_features.shape)
print("Question features", questions.shape)
print("Dictionary size", pt.dic_size)

# Consider three thresholds
eo = ExtractObjects(cut_data=data_amount, output_fileName='objects_val.txt', subset='val2014', threshold=25)
object_matrix1 = eo.onehotvector(annotations)
eo = ExtractObjects(cut_data=data_amount, output_fileName='objects_val.txt', subset='val2014', threshold=50)
object_matrix2 = eo.onehotvector(annotations)
eo = ExtractObjects(cut_data=data_amount, output_fileName='objects_val.txt', subset='val2014', threshold=75)
object_matrix3 = eo.onehotvector(annotations)

object_matrix = np.concatenate([object_matrix1, object_matrix2, object_matrix3], axis = 1)

# Predict
pred = neuralnet.predict_current_state(image_features, object_matrix, questions)
#pred = numpy.argmax(pred, axis=1)
#pred = np.round(pred)
print(pred)
print(answers)

# One hot encode
"""
predO = np.zeros((pred.shape[0], 2))
for i in range(pred.shape[0]):
    predO[i, int(pred[i, 0])] = 1
print(predO)
"""
from EvaluateModel import ProduceResult
model_evaluator = ProduceResult(p._int_to_answer, p._answer_to_int, dataSubType='val2014')
answers = model_evaluator.produce_results(pred, pt._original_questions)
model_evaluator.evaluate(taskType=taskType)