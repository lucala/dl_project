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
    def __init__(self, n, m, r, dic_size, objects = 80, epochs=5, batchSize=128, checkpoint_period=1, monitor='val_loss', loss='mean_squared_error', lr=0.001):
        self._n = n  # number of samples
        self._m = m  # number of image features
        self._r = r  # number of word features
        self._dic_size = dic_size # size of the dictionary
        self._objects = objects # Number of object features
        self._epochs = epochs
        self._batchSize = batchSize
        self._wordEmbeddingSize = 1024
        if (self._wordEmbeddingSize % self._r != 0):
            print("Dimension mismatch: number of words in a question has to divide embedding size")
        self._checkpoint_period = checkpoint_period
        self._monitor = monitor
        self._loss = loss

        image_input = Input(shape=(self._m,),name="image_input")
        object_input = Input(shape=(self._objects,),name="objects_input")
        word_input = Input(shape=(self._r,),name="word_input")
        word_embedding = Embedding(self._dic_size, int(self._wordEmbeddingSize / self._r), input_length=self._r, name='word_embedding')(word_input)
        flatten_embedding = Flatten(name="flatten_embedding")(word_embedding)
        merge = keras.layers.concatenate([image_input, object_input, flatten_embedding])
        # hidden_layer0 = Dense(2048, activation='relu')(merge)
        hidden_layer1 = Dense(1024, activation='relu')(merge)
        hidden_layer2 = Dense(512, activation='relu')(hidden_layer1)
        # hidden_layer3 = Dense(256, activation='relu')(hidden_layer2)
        output_layer = Dense(1, activation='relu', name='output_layer')(hidden_layer2)

        self._model = Model(inputs=[image_input, object_input, word_input], outputs=[output_layer])
        print(self._model.summary())
        optimizer = optimizers.Adam(lr=lr)
        self._model.compile(loss=self._loss, optimizer=optimizer, metrics=['accuracy']) #TODO what loss?

    def get_model_summary(self):
        return self._model.summary()

    def fit(self, images, object_matrix, questions, labels):
        # Split the data
        # self._matrix = matrix
        self._labels = labels
        checkpointer = ModelCheckpoint(filepath="weights/weights-{epoch:02d}-{val_loss:.4f}.hdf5", verbose=1, save_best_only=True, monitor=self._monitor, period=self._checkpoint_period)
        history = History()
        
        print(images.shape, object_matrix.shape, questions.shape, self._labels.shape)            
        self._model.fit(x=[images, object_matrix, questions],y=self._labels, batch_size=self._batchSize, epochs=self._epochs, validation_split=0.3, callbacks=[history, checkpointer])
        
    def predict(self, images, object_matrix, questions, weights):
        self._model.load_weights(weights)
        prediction = self._model.predict(x=[images, object_matrix, questions], batch_size=self._batchSize)
        return prediction

    def predict_current_state(self, images, object_matrix, questions):
        prediction = self._model.predict(x=[images, object_matrix, questions], batch_size=self._batchSize)
        return prediction

"""
from PrepareOriginalData import PrepareData
from EvaluateModel import ProduceResult
import numpy as np

# Some constants
taskType = 'all'
data_amount = 1
epochs = 200

# Load training set
p = PrepareData(path_images='data_vqa_feat', # Path to image features 
                subset='train2014', # Desired subset: either train2014 or val2014
                taskType=taskType, # 'OpenEnded', 'MultipleChoice', 'all'
                cut_data=data_amount, # Percentage of data to use, 1 = All values, above 1=#samples for debugging
                output_path='data', # Path where we want to output temporary data
                pad_length=32, # Number of words in a question (zero padded)
                question_threshold=0, answer_threshold=0, # Keep only most common words
                answers_sparse=True, questions_sparse=True)
_, questions, _, annotations = p.load_data()
print("Question features", questions.shape)
print("Dictionary size", p.dic_size)

# Save dictionary
p.dumpDictionary('dictionary_all_types')

# Get labels
from VQA.PythonHelperTools.loadData import get_type
y = np.array([2 if get_type(ann) == 'number' else 1 if get_type(ann) == 'yes/no' else 0 for ann in annotations])
print(y)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()
classifier.fit(questions, y)

# Train on only one question type
question_type = 'number'
questions = annotations = []
# Load training set
p = PrepareData(path_images='data_vqa_feat', # Path to image features 
                subset='train2014', # Desired subset: either train2014 or val2014
                taskType=taskType, # 'OpenEnded', 'MultipleChoice', 'all'
                cut_data=data_amount, # Percentage of data to use, 1 = All values, above 1=#samples for debugging
                output_path='data', # Path where we want to output temporary data
                pad_length=32, # Number of words in a question (zero padded)
                question_threshold=0, answer_threshold=0, # Keep only most common words
                answers_sparse=True, questions_sparse=True, answer_type=question_type,
                precomputed_dic=p._question_dict)
image_features, questions, answers, annotations = p.load_data()
print("Image features", image_features.shape)
print("Question features", questions.shape)
print("Answers", answers.shape)
print("Dictionary size", p.dic_size)
print("Number of possible classes", np.max(answers) + 1)

# Save dictionary
p.dumpDictionary('dictionary_yes_no')

# Transform labels to integers
max_int = 15
def extract_number(ann):
    numbers = [int(s) for s in ann['multiple_choice_answer'].split() if s.isdigit()]
    if len(numbers) > 0:
        return numbers[0]
    else:
        return max_int
y = np.array([min(max_int, extract_number(ann)) for ann in annotations])
print(y)
"""
"""
# Extract object features
from ExtractObjects import ExtractObjects

# Consider three thresholds
eo = ExtractObjects(cut_data=data_amount, output_fileName='objects_train.txt', subset='train2014', threshold=25)
object_matrix1 = eo.onehotvector(annotations)
eo = ExtractObjects(cut_data=data_amount, output_fileName='objects_train.txt', subset='train2014', threshold=50)
object_matrix2 = eo.onehotvector(annotations)
eo = ExtractObjects(cut_data=data_amount, output_fileName='objects_train.txt', subset='train2014', threshold=75)
object_matrix3 = eo.onehotvector(annotations)

object_matrix = np.concatenate([object_matrix1, object_matrix2, object_matrix3], axis = 1)
print(object_matrix.shape)
np.save('data/object_matrix_train_numbers.npy', object_matrix)
"""
"""
object_matrix = np.load('data/object_matrix_train_numbers.npy')

# Use this when using sparse representation
neuralnet = NeuralNetwork(image_features.shape[0],1024,questions.shape[1],p.dic_size, epochs = epochs, batchSize=64,
                          objects=240, lr=0.0001, checkpoint_period=20)

# Train network
neuralnet.fit(image_features, object_matrix, questions, y)

# Test performance on training set
pred = neuralnet.predict_current_state(image_features, object_matrix, questions)
# pred = neuralnet.predict(image_features, object_matrix, questions, "weights/weights-38-0.4695.hdf5")
pred = pred.astype(int)
np.sum(pred < 0)
pred[pred > max_int] = max_int
pred = pred[:, 0]
model_evaluator = ProduceResult(p._int_to_answer, p._answer_to_int, dataSubType='train2014')
answers = model_evaluator.produce_results_numbers(pred, p._original_questions, threshold=max_int, max_word='many')
model_evaluator.evaluate(taskType=taskType)

# Load validation set and evaluate prediction on it
image_features = questions = answers = annotations = []
pt= PrepareData(path_images='data_vqa_feat', # Path to image features 
                subset='val2014', # Desired subset: either train2014 or val2014
                taskType=taskType, # 'OpenEnded', 'MultipleChoice', 'all'
                cut_data=data_amount, # Percentage of data to use, 1 = All values, above 1 = 10 samples for debugging
                output_path='data', # Path where we want to output temporary data
                pad_length=32)
pt.loadDictionary('data/dictionary_all_types.pkl') # Use same dictionary as in training
image_features, questions, _, annotations = pt.load_data()
print("Image features", image_features.shape)
print("Question features", questions.shape)
print("Dictionary size", pt.dic_size)

# Check prediction accuracy of answer-type classifier
y = np.array([2 if get_type(ann) == 'number' else 1 if get_type(ann) == 'yes/no' else 0 for ann in annotations])
# Predict
pred = classifier.predict(questions)

from sklearn.metrics import accuracy_score
# TODO: can probably still improve this accuracy
print('Answer type classification accuracy:', accuracy_score(pred, y))

# Filter questions accordingly to their predicted type
question_type_idx = 2 if question_type == 'number' else 1 if question_type == 'yes/no' else 0
image_features = image_features[pred == question_type_idx, :]
questions = questions[pred == question_type_idx, :]
annotations = np.array(annotations)[pred == question_type_idx]
original_questions = np.array(pt._original_questions)[pred == question_type_idx]
print(image_features.shape)
print(questions.shape)
print(original_questions.shape)
print(annotations.shape)

# Extract object features
from ExtractObjects import ExtractObjects

# Consider three thresholds
eo = ExtractObjects(cut_data=data_amount, output_fileName='objects_val.txt', subset='val2014', threshold=25)
object_matrix1 = eo.onehotvector(annotations)
eo = ExtractObjects(cut_data=data_amount, output_fileName='objects_val.txt', subset='val2014', threshold=50)
object_matrix2 = eo.onehotvector(annotations)
eo = ExtractObjects(cut_data=data_amount, output_fileName='objects_val.txt', subset='val2014', threshold=75)
object_matrix3 = eo.onehotvector(annotations)

object_matrix = np.concatenate([object_matrix1, object_matrix2, object_matrix3], axis = 1)
print(object_matrix.shape)
# np.save('data/object_matrix_val.npy', object_matrix)

# Test prediction on validation set
# pred = neuralnet.predict(image_features, object_matrix, questions, 'weights/weights-96-0.1448.hdf5')
pred = neuralnet.predict_current_state(image_features, object_matrix, questions)
print(pred.shape)

# Transform predictions to numbers
pred = pred.astype(int)
np.sum(pred < 0)
pred[pred > max_int] = max_int
pred = pred[:, 0]

model_evaluator = ProduceResult(p._int_to_answer, p._answer_to_int, dataSubType='val2014')
answers = model_evaluator.produce_results_numbers(pred, original_questions, threshold=max_int, max_word='many')
model_evaluator.evaluate(taskType=taskType)
"""