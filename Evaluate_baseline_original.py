from PrepareOriginalData import PrepareData
from NeuralNetworkBaseline import NeuralNetwork
import numpy as np
from EvaluateModel import ProduceResult


# Some constants
taskType = 'all'
data_amount = 1
epochs = 50

# Load training set
p = PrepareData(path_images='data_vqa_feat', # Path to image features 
                        subset='train2014', # Desired subset: either train2014 or val2014
                        taskType=taskType, # 'OpenEnded', 'MultipleChoice', 'all'
                        cut_data=data_amount, # Percentage of data to use, 1 = All values, above 1=#samples for debugging
                        output_path='data', # Path where we want to output temporary data
                        pad_length=32, # Number of words in a question (zero padded)
                        question_threshold=0, answer_threshold=0, # Keep only most common words
                        answers_sparse=True, questions_sparse=True)
_, _, answers, _ = p.load_data()
p.dumpDictionary()
print("Number of possible classes", np.max(answers) + 1)

weights = 'weights/weights-379-3.1370.hdf5'

# Load validation set and evaluate prediction on it
pt= PrepareData(path_images='data_vqa_feat', # Path to image features 
                        subset='val2014', # Desired subset: either train2014 or val2014
                        taskType=taskType, # 'OpenEnded', 'MultipleChoice', 'all'
                        cut_data=data_amount, # Percentage of data to use, 1 = All values, above 1 = 10 samples for debugging
                        output_path='data')
pt.loadDictionary('data/dictionary.pkl') # Use same dictionary as in training
image_features, questions, _, annotations = pt.load_data()
print("Image features", image_features.shape)
print("Question features", questions.shape)
print("Dictionary size", pt.dic_size)

# Define model
neuralnet = NeuralNetwork(image_features.shape[0],1024,questions.shape[1],pt.dic_size,np.max(answers)+1, epochs = epochs)

# Test prediction on validation set
pred = neuralnet.predict(image_features, questions, weights)

model_evaluator = ProduceResult(pt._int_to_answer, pt._answer_to_int, dataSubType='val2014')
model_evaluator.produce_results(pred, pt._original_questions)
model_evaluator.evaluate(taskType=taskType)
