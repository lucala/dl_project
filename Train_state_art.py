import numpy as np
import pickle
from NeuralNetworkStateOfTheArtApproachSandro import NeuralNetwork
from EvaluateModel import ProduceResult

# Some constants
taskType = 'all'
data_amount = 1
epochs = 30
glove_version = 300
soft_labels = False

# Train on only one question type
question_type = 'all'

#use 6 and 3 for questions and answers!

if soft_labels:
    from PrepareDataSoftLabels import PrepareData
else:
    from PrepareOriginalData import PrepareData

# Load training set
p = PrepareData(path_images='data_vqa_feat', # Path to image features 
                subset='trainval', # Desired subset: either train2014 or val2014
                taskType='OpenEnded', # 'OpenEnded', 'MultipleChoice', 'all'
                cut_data=data_amount, # Percentage of data to use, 1 = All values, above 1=#samples for debugging
                output_path='data', # Path where we want to output temporary data
                pad_length=14, # Number of words in a question (zero padded)
                question_threshold=6, answer_threshold=3, # Keep only most common words #yes/no use 10
                questions_sparse=True, answer_type=question_type,
                image_extractor='RawImages')
image_features, questions, answers, annotations = p.load_data()
print("Image features", image_features.shape)
print("Question features", questions.shape)
print("Answers", answers.shape)
print("Dictionary size", p.dic_size)
print("Number of possible classes", answers.shape[1] if soft_labels else np.max(answers) + 1)

# Save dictionary
p.dumpDictionary('dictionary_other')

# Load Glove embedding
# load the whole embedding into memory
embeddings_index = dict()
f = open('data/glove.6B.' + str(glove_version) + 'd.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))

# Create an embedding matrix
embedding_matrix = np.zeros((np.max(questions) + 1, glove_version)) # Ignore thresholded words
for word in p._question_dict.keys():
    # If word was thresholded ignore it
    if p._question_dict[word] != 0:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[p._question_dict[word]] = embedding_vector
print(embedding_matrix.shape)

# Merge features of two dataset into one
"""
attentions = pickle.load(open('data/attentions_train.pkl', 'rb'))
with open('data/attentions_val.pkl', 'rb') as f:
    attentions_tmp = pickle.load(f)
    for key in attentions_tmp.keys():
        attentions[key] = attentions_tmp[key]

# Load extracted features
image_features = np.zeros((len(p._original_questions), 4, 4, 2048))
for i in range(len(p._original_questions)):
    image_features[i] = attentions[p._original_questions[i]['image_id']]
attentions = [] # In order to save memory

# Reshape
image_features = image_features.reshape((image_features.shape[0], 16, 2048))
print(image_features.shape)
"""

if soft_labels:
    # TODO: make it work
    neuralnet = NeuralNetwork(image_features.shape[0],2048,questions.shape[1],np.max(questions)+1,
                              embedding_matrix, answers.shape[1], epochs = epochs, batchSize=384,
                              loss='binary_crossentropy', activation='sigmoid', monitor='val_loss',
                              checkpoint_period=1, wordEmbeddingSize=glove_version)
else:
    neuralnet = NeuralNetwork(image_features.shape[0],2048,questions.shape[1],np.max(questions)+1,
                              embedding_matrix, np.max(answers)+1, epochs = epochs, batchSize=384,
                              loss='sparse_categorical_crossentropy', activation='softmax', monitor='train_loss',
                              checkpoint_period=5, wordEmbeddingSize=glove_version)

# Train network
# neuralnet.fit(image_features, questions, answers, validation_split=0)

"""
#see what it predicts on training data, on which it trained!
pred = neuralnet.predict_current_state(image_features, questions)
print(pred)
print(answers)

# Load validation set and evaluate prediction on it
image_features = questions = answers = annotations = pred = [] # Free memory
pt= PrepareData(path_images='data_vqa_feat', # Path to image features 
                subset='val2014', # Desired subset: either train2014 or val2014
                taskType='all', # 'OpenEnded', 'MultipleChoice', 'all'
                cut_data=data_amount, # Percentage of data to use, 1 = All values, above 1 = 10 samples for debugging
                output_path='data')
pt.loadDictionary('data/dictionary_other.pkl') # Use same dictionary as in training
image_features, questions, answers, annotations = pt.load_data()
print("Image features", image_features.shape)
print("Question features", questions.shape)
print("Dictionary size", pt.dic_size)

# Load extracted features
image_features = np.zeros((len(pt._original_questions), 4, 4, 2048))
with open('data/attentions_val.pkl', 'rb') as f:
    attentions = pickle.load(f)
    for i in range(len(pt._original_questions)):
        image_features[i] = attentions[pt._original_questions[i]['image_id']]

# Reshape
image_features = image_features.reshape((image_features.shape[0], 16, 2048))
print(image_features.shape)

# Predict
pred = neuralnet.predict_current_state(image_features, questions)
print(pred)

print("=== Results on validation set ===")
model_evaluator = ProduceResult(p._int_to_answer, p._answer_to_int, dataSubType='val2014')
model_evaluator.produce_results(pred, pt._original_questions)
model_evaluator.evaluate(taskType=taskType)
"""

# Load test set
image_features = questions = answers = annotations = [] # Free memory
features = {}
image_features, questions = p.load_test_set(set_name='test-dev2015')
print("Image features", image_features.shape)
print("Question features", questions.shape)
print("Dictionary size", p.dic_size)

# Load extracted features
image_features = np.zeros((len(p._original_questions), 4, 4, 2048))
with open('data/attentions_test.pkl', 'rb') as f:
    attentions = pickle.load(f)
    for i in range(len(p._original_questions)):
        image_features[i] = attentions[p._original_questions[i]['image_id']]

# Reshape
image_features = image_features.reshape((image_features.shape[0], 16, 2048))
print(image_features.shape)

# Predict
pred = neuralnet.predict(image_features, questions, 'weights/weights-14.hdf5')

# Produce results
model_evaluator = ProduceResult(p._int_to_answer, p._answer_to_int, dataSubType='test-dev2015', modelName='sa14')
model_evaluator.produce_results(pred, p._original_questions)
"""
# Predict
pred = neuralnet.predict(image_features, questions, 'weights/weights-24.hdf5')

# Produce results
model_evaluator = ProduceResult(p._int_to_answer, p._answer_to_int, dataSubType='test-dev2015', modelName='sa24')
model_evaluator.produce_results(pred, p._original_questions)
"""
# Predict
pred = neuralnet.predict(image_features, questions, 'weights/weights-29.hdf5')

# Produce results
model_evaluator = ProduceResult(p._int_to_answer, p._answer_to_int, dataSubType='test-dev2015', modelName='sa29')
model_evaluator.produce_results(pred, p._original_questions)
