# Load data
# from PrepareOriginalData import PrepareData
from PrepareDataSoftLabels import PrepareData
from ExtractObjects import ExtractObjects
from NeuralNetworkBaselineDenseEmb import NeuralNetwork
from EvaluateModel import ProduceResult
import numpy as np

# Some constants
taskType = 'all'
data_amount = 1
epochs = 80

# Load training set
p = PrepareData(path_images='data_vqa_feat', # Path to image features 
                subset='trainval', # Desired subset: either train2014 or val2014
                taskType='OpenEnded', # 'OpenEnded', 'MultipleChoice', 'all'
                cut_data=data_amount, # Percentage of data to use, 1 = All values, above 1=#samples for debugging
                output_path='data', # Path where we want to output temporary data
                pad_length=32, # Number of words in a question (zero padded)
                question_threshold=6, answer_threshold=3, # Keep only most common words
                questions_sparse=False)
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
print(object_matrix.shape)
np.save('data/object_matrix_trainval.npy', object_matrix)
"""
object_matrix = np.load('data/object_matrix_trainval.npy')

# Define neural network
# neuralnet = NeuralNetwork(image_features.shape[0],1024,questions.shape[1],p.dic_size,np.max(answers)+1, objects = 240, checkpoint_period=10, lr=0.0001,
#                          epochs = epochs, batchSize=512, activation='softmax', loss='sparse_categorical_crossentropy', monitor='val_acc')
neuralnet = NeuralNetwork(image_features.shape[0], 1024, questions.shape[1], answers.shape[1], objects= 240, epochs = epochs, lr=0.0001, batchSize=1024,
                          activation='sigmoid', loss='binary_crossentropy')
print(neuralnet.get_model_summary())

# Fine tune
neuralnet.load_weights('weights/weights-119.hdf5')

# Train network
neuralnet.fit(image_features, object_matrix, questions, answers, monitor='train_loss', validation_split=0, checkpoint_period=20)

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
"""
eo = ExtractObjects(cut_data=data_amount, output_fileName='objects_val.txt', subset='val2014', threshold=25)
object_matrix1 = eo.onehotvector(annotations)
eo = ExtractObjects(cut_data=data_amount, output_fileName='objects_val.txt', subset='val2014', threshold=50)
object_matrix2 = eo.onehotvector(annotations)
eo = ExtractObjects(cut_data=data_amount, output_fileName='objects_val.txt', subset='val2014', threshold=75)
object_matrix3 = eo.onehotvector(annotations)

object_matrix = np.concatenate([object_matrix1, object_matrix2, object_matrix3], axis=1)

# Test prediction on validation set
# pred = neuralnet.predict(image_features, object_matrix, questions, 'weights/weights-59.hdf5')
pred = neuralnet.predict_current_state(image_features, object_matrix, questions)
print(pred.shape)

# Evaluate model
print("=== Results on validation set ===")
image_features = questions = answers = annotations = []
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

# Test prediction on validation set
# pred = neuralnet.predict(image_features, object_matrix, questions, 'weights/weights-99.hdf5')
pred = neuralnet.predict_current_state(image_features, object_matrix, questions)
print(pred.shape)

# Evaluate model
model_evaluator = ProduceResult(pt._int_to_answer, pt._answer_to_int, dataSubType='test-dev2015', modelName='baseline200')
model_evaluator.produce_results(pred, pt._original_questions)

# Test prediction on validation set
pred = neuralnet.predict(image_features, object_matrix, questions, 'weights/weights-59.hdf5')
# pred = neuralnet.predict_current_state(image_features, object_matrix, questions)
print(pred.shape)

# Evaluate model
model_evaluator = ProduceResult(pt._int_to_answer, pt._answer_to_int, dataSubType='test-dev2015', modelName='baseline180')
model_evaluator.produce_results(pred, pt._original_questions)

# Test prediction on validation set
pred = neuralnet.predict(image_features, object_matrix, questions, 'weights/weights-39.hdf5')
# pred = neuralnet.predict_current_state(image_features, object_matrix, questions)
print(pred.shape)

# Evaluate model
model_evaluator = ProduceResult(pt._int_to_answer, pt._answer_to_int, dataSubType='test-dev2015', modelName='baseline160')
model_evaluator.produce_results(pred, pt._original_questions)

# Test prediction on validation set
pred = neuralnet.predict(image_features, object_matrix, questions, 'weights/weights-19.hdf5')
# pred = neuralnet.predict_current_state(image_features, object_matrix, questions)
print(pred.shape)

# Evaluate model
model_evaluator = ProduceResult(pt._int_to_answer, pt._answer_to_int, dataSubType='test-dev2015', modelName='baseline140')
model_evaluator.produce_results(pred, pt._original_questions)
