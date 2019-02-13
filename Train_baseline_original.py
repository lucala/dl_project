from PrepareOriginalData import PrepareData
from NeuralNetworkBaseline import NeuralNetwork
import numpy as np
from EvaluateModel import ProduceResult

# Some constants
taskType = 'all'
data_amount = 1
epochs = 400

# Load training set
p = PrepareData(path_images='data_vqa_feat', # Path to image features 
                        subset='train2014', # Desired subset: either train2014 or val2014
                        taskType=taskType, # 'OpenEnded', 'MultipleChoice', 'all'
                        cut_data=data_amount, # Percentage of data to use, 1 = All values, above 1=#samples for debugging
                        output_path='data', # Path where we want to output temporary data
                        pad_length=32, # Number of words in a question (zero padded)
                        question_threshold=0, answer_threshold=0, # Keep only most common words
                        answers_sparse=True, questions_sparse=True)
image_features, questions, answers, annotations = p.load_data()
print("Image features", image_features.shape)
print("Question features", questions.shape)
print("Answers", answers.shape)
print("Dictionary size", p.dic_size)
print("Number of possible classes", np.max(answers) + 1)

# Save dictionary
p.dumpDictionary()

# Use this when using sparse representation
neuralnet = NeuralNetwork(image_features.shape[0],1024,questions.shape[1],p.dic_size,np.max(answers)+1,
                                  epochs = epochs, batchSize=1024, checkpoint_period=10)

# Train network
neuralnet.fit(image_features, questions, answers)

# Test prediction on training set
print("=== Results on training set ===")
# Predict
pred = neuralnet.predict_current_state(image_features, questions)

# Evaluate
model_evaluator = ProduceResult(p._int_to_answer, p._answer_to_int, dataSubType='train2014')
model_evaluator.produce_results(pred, p._original_questions)
model_evaluator.evaluate(taskType=taskType)

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

# Test prediction on validation set
pred = neuralnet.predict_current_state(image_features, questions)
print("=== Results on validation set ===")
model_evaluator = ProduceResult(pt._int_to_answer, pt._answer_to_int, dataSubType='val2014')
answers = model_evaluator.produce_results(pred, pt._original_questions)
model_evaluator.evaluate(taskType=taskType)
