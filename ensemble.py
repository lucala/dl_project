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
soft_labels = True

if soft_labels:
    from PrepareDataSoftLabels import PrepareData
else:
    from PrepareOriginalData import PrepareData

# Load training set
# Note: this operation is needed only to generate a dictionary, but one could also load a saved one instead
p = PrepareData(path_images='data_vqa_feat', # Path to image features 
                subset='trainval', # Desired subset: either train2014, val2014 or trainval
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

# Define neural network
if soft_labels:
    neuralnet = NeuralNetwork(image_features.shape[0], 2048, questions.shape[1], answers.shape[1], objects=240, activation='sigmoid')
else:
    neuralnet = NeuralNetwork(image_features.shape[0], 2048, questions.shape[1], np.max(answers) + 1, objects= 240, activation='softmax')

# Predict on test set
print("=== Predicting on test set ===")
pred = []
image_features, questions = p.load_test_set(set_name='test-dev2015')
print("Image features", image_features.shape)
print("Question features", questions.shape)

# Extract objects
# Consider three thresholds
eo = ExtractObjects(cut_data=data_amount, output_fileName='objects_test.txt', subset='test-dev2015', threshold=25)
object_matrix1 = eo.onehotvector(p._original_questions)
eo = ExtractObjects(cut_data=data_amount, output_fileName='objects_test.txt', subset='test-dev2015', threshold=50)
object_matrix2 = eo.onehotvector(p._original_questions)
eo = ExtractObjects(cut_data=data_amount, output_fileName='objects_test.txt', subset='test-dev2015', threshold=75)
object_matrix3 = eo.onehotvector(p._original_questions)

# Concatenate different thresholds
object_matrix = np.concatenate([object_matrix1, object_matrix2, object_matrix3], axis=1)
print(object_matrix.shape)

# Load extracted features
image_attentions = np.zeros((len(p._original_questions), 4, 4, 2048))
with open('data/attentions_test.pkl', 'rb') as f:
    attentions = pickle.load(f)
    for i in range(len(p._original_questions)):
        image_attentions[i] = attentions[p._original_questions[i]['image_id']]

# Reshape
image_attentions = image_attentions.reshape((image_features.shape[0], 16, 2048))
print(image_attentions.shape)

# Normalise features
for i in range(16):
    image_attentions[:, i, :] = normalize(image_attentions[:, i, :], copy=False)

# Predict and ensemble
pred = neuralnet.predict(image_attentions, object_matrix, questions, 'weights/weights_att_soft.hdf5')
pred = pred + neuralnet.predict(image_features, object_matrix, questions, 'weights/weights_baseline_soft.hdf5')
pred = 0.5 * pred
print(pred.shape)

# Produce results
model_evaluator = ProduceResult(p._int_to_answer, p._answer_to_int, dataSubType='test-dev2015', modelName='ensemble')
model_evaluator.produce_results(pred, p._original_questions)
