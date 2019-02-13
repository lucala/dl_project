# Load data
from ExtractObjects import ExtractObjects
from NeuralNetworkDenseAttentions import NeuralNetwork
from EvaluateModel import ProduceResult
import numpy as np
import pickle
from sklearn.preprocessing import normalize

# Some constants
taskType = 'all'
data_amount = 1
epochs = 300
soft_labels = True
train_on = 'trainval'

if soft_labels:
    from PrepareDataSoftLabels import PrepareData
else:
    from PrepareOriginalData import PrepareData

# Load training set
p = PrepareData(path_images='data_vqa_feat', # Path to image features 
                subset=train_on, # Desired subset: either train2014 or val2014
                taskType='OpenEnded', # 'OpenEnded', 'MultipleChoice', 'all'
                cut_data=data_amount, # Percentage of data to use, 1 = All values, above 1=#samples for debugging
                output_path='data', # Path where we want to output temporary data
                pad_length=32, # Number of words in a question (zero padded)
                question_threshold=6, answer_threshold=3, # Keep only most common words
                questions_sparse=False,
                image_extractor='RawImages')
image_features, questions, answers, annotations = p.load_data()
print("Image features", image_features.shape)
print("Question features", questions.shape)
print("Answers", answers.shape)
print("Dictionary size", p.dic_size)
p.dumpDictionary()

# Load object features
# Consider three thresholds
"""
eo = ExtractObjects(cut_data=data_amount, output_fileName='objects_train.txt', subset='train2014', threshold=25)
object_matrix1 = eo.onehotvector(annotations)
eo = ExtractObjects(cut_data=data_amount, output_fileName='objects_train.txt', subset='train2014', threshold=50)
object_matrix2 = eo.onehotvector(annotations)
eo = ExtractObjects(cut_data=data_amount, output_fileName='objects_train.txt', subset='train2014', threshold=75)
object_matrix3 = eo.onehotvector(annotations)

object_matrix = np.concatenate([object_matrix1, object_matrix2, object_matrix3], axis=1)
print(object_matrix.shape)
# np.save('data/object_matrix_trainval.npy', object_matrix)
"""

object_matrix = np.load('data/object_matrix_trainval.npy')

# Load image features
attentions = pickle.load(open('data/attentions_train.pkl', 'rb'))

if train_on == 'trainval':
    with open('data/attentions_val.pkl', 'rb') as f:
        attentions_tmp = pickle.load(f)
        for key in attentions_tmp.keys():
            attentions[key] = attentions_tmp[key]

# Order them according to questions
image_features = np.zeros((len(p._original_questions), 4, 4, 2048))
for i in range(len(p._original_questions)):
    image_features[i] = attentions[p._original_questions[i]['image_id']]
attentions = [] # In order to save memory

# Reshape
image_features = image_features.reshape((image_features.shape[0], 16, 2048))
print(image_features.shape)

# Normalise features
for i in range(16):
    image_features[:, i, :] = normalize(image_features[:, i, :], copy=False)

# Define neural network
if soft_labels:
    neuralnet = NeuralNetwork(image_features.shape[0], 2048, questions.shape[1], answers.shape[1], objects= 240, epochs = epochs, batchSize=1024,
                              activation='sigmoid', loss='binary_crossentropy', checkpoint_period=20, lr=0.0001)
else:
    neuralnet = NeuralNetwork(image_features.shape[0], 2048, questions.shape[1], np.max(answers) + 1, objects= 240, epochs = epochs, batchSize=512,
                              activation='softmax', loss='sparse_categorical_crossentropy', checkpoint_period=15, lr=0.0001)
print(neuralnet.get_model_summary())

# Fine tune
# neuralnet.load_weights('weights/weights-199.hdf5')

# Train network
neuralnet.fit(image_features, object_matrix, questions, answers, validation_split=0)

# Load validation set and evaluate prediction on it
image_features = questions = answers = annotations = object_matrix = []
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
"""
eo = ExtractObjects(cut_data=data_amount, output_fileName='objects_val.txt', subset='val2014', threshold=25)
object_matrix1 = eo.onehotvector(annotations)
eo = ExtractObjects(cut_data=data_amount, output_fileName='objects_val.txt', subset='val2014', threshold=50)
object_matrix2 = eo.onehotvector(annotations)
eo = ExtractObjects(cut_data=data_amount, output_fileName='objects_val.txt', subset='val2014', threshold=75)
object_matrix3 = eo.onehotvector(annotations)

object_matrix = np.concatenate([object_matrix1, object_matrix2, object_matrix3], axis=1)
object_matrix1 = object_matrix2 = object_matrix3 = []

# Load image features
image_features = np.zeros((len(pt._original_questions), 4, 4, 2048))
with open('data/attentions_val.pkl', 'rb') as f:
    attentions = pickle.load(f)
    for i in range(len(pt._original_questions)):
        image_features[i] = attentions[pt._original_questions[i]['image_id']]

# Reshape
image_features = image_features.reshape((image_features.shape[0], 16, 2048))
print(image_features.shape)

# Normalise features
for i in range(16):
    image_features[:, i, :] = normalize(image_features[:, i, :], copy=False)

# Test prediction on validation set
pred = neuralnet.predict(image_features, object_matrix, questions, 'weights_sa/weights-04.hdf5')
# pred = neuralnet.predict_current_state(image_features, object_matrix, questions)
print(pred.shape)

# Evaluate model
print("=== Results on validation set ===")
image_features = questions = answers = annotations = object_matrix = []
model_evaluator = ProduceResult(pt._int_to_answer, pt._answer_to_int, dataSubType='val2014')
answers = model_evaluator.produce_results(pred, pt._original_questions)
model_evaluator.evaluate(taskType=taskType)
"""

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

# Load extracted features
image_features = np.zeros((len(pt._original_questions), 4, 4, 2048))
with open('data/attentions_test.pkl', 'rb') as f:
    attentions = pickle.load(f)
    for i in range(len(pt._original_questions)):
        image_features[i] = attentions[pt._original_questions[i]['image_id']]

# Reshape
image_features = image_features.reshape((image_features.shape[0], 16, 2048))
print(image_features.shape)

# Normalise features
for i in range(16):
    image_features[:, i, :] = normalize(image_features[:, i, :], copy=False)

# Test prediction on test set
pred = neuralnet.predict(image_features, object_matrix, questions, 'weights/weights-299.hdf5')
pred = neuralnet.predict_current_state(image_features, object_matrix, questions)
print(pred.shape)

# Evaluate model
model_evaluator = ProduceResult(pt._int_to_answer, pt._answer_to_int, dataSubType='test-dev2015', modelName='sandro_300')
model_evaluator.produce_results(pred, pt._original_questions)

# Test prediction on test set
pred = neuralnet.predict(image_features, object_matrix, questions, 'weights/weights-274.hdf5')
# pred = neuralnet.predict_current_state(image_features, object_matrix, questions)
print(pred.shape)

# Evaluate model
model_evaluator = ProduceResult(pt._int_to_answer, pt._answer_to_int, dataSubType='test-dev2015', modelName='sandro275')
model_evaluator.produce_results(pred, pt._original_questions)

# Test prediction on test set
pred = neuralnet.predict(image_features, object_matrix, questions, 'weights/weights-249.hdf5')
# pred = neuralnet.predict_current_state(image_features, object_matrix, questions)
print(pred.shape)

# Evaluate model
model_evaluator = ProduceResult(pt._int_to_answer, pt._answer_to_int, dataSubType='test-dev2015', modelName='sandro250')
model_evaluator.produce_results(pred, pt._original_questions)

# Test prediction on test set
pred = neuralnet.predict(image_features, object_matrix, questions, 'weights/weights-224.hdf5')
# pred = neuralnet.predict_current_state(image_features, object_matrix, questions)
print(pred.shape)

# Evaluate model
model_evaluator = ProduceResult(pt._int_to_answer, pt._answer_to_int, dataSubType='test-dev2015', modelName='sandro225')
model_evaluator.produce_results(pred, pt._original_questions)

# Test prediction on test set
pred = neuralnet.predict(image_features, object_matrix, questions, 'weights/weights-199.hdf5')
# pred = neuralnet.predict_current_state(image_features, object_matrix, questions)
print(pred.shape)

# Evaluate model
model_evaluator = ProduceResult(pt._int_to_answer, pt._answer_to_int, dataSubType='test-dev2015', modelName='sandro200')
model_evaluator.produce_results(pred, pt._original_questions)

# Test prediction on test set
pred = neuralnet.predict(image_features, object_matrix, questions, 'weights_sa/weights-174.hdf5')
# pred = neuralnet.predict_current_state(image_features, object_matrix, questions)
print(pred.shape)

# Evaluate model
model_evaluator = ProduceResult(pt._int_to_answer, pt._answer_to_int, dataSubType='test-dev2015', modelName='sandro175')
model_evaluator.produce_results(pred, pt._original_questions)
