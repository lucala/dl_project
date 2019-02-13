from PrepareData import PrepareData
import numpy as np
from NeuralNetworkBaselineDenseEmb import NeuralNetworkBaseline
#from NeuralNetworkBaselineGPU import NeuralNetworkBaselineGPU

p = PrepareData(path_images='data_vqa_feat', # Path to image features 
                path_questions='vqa_data_share', # Path to questions and answers
                subset='val', # Desired subset: either train or val, test is not supported yet
                cut_data=1, # Percentage of data to use, 1 = All values, above 1 = 10 samples for debugging
                output_path='data', # Path where we want to output temporary data
                pad_length=32, # Number of words in a question (zero padded)
                question_threshold=6, answer_threshold=3, # Keep only most common words
                answers_sparse=False, questions_sparse=False)
X, y, dic_size = p.load_data()
print(X.shape) # First 1024 columns are image features, the rest are question features
print(y.shape)
print(dic_size)
print(np.max(y))

print(X.shape[0],1024,X.shape[1]-1024,dic_size,np.max(y) + 1)
# Use this when using one-hot encoding
neuralnet = NeuralNetworkBaseline(X.shape[0],1024,X.shape[1]-1024,y.shape[1], epochs = 100)
# Use this when using sparse representation
# neuralnet = NeuralNetworkBaseline(X.shape[0],1024,X.shape[1]-1024,dic_size,np.max(y)+1, epochs = 100, loss='sparse_categorical_crossentropy')
#neuralnet = NeuralNetworkBaselineGPU(X.shape[0],1024,X.shape[1]-1024,y.shape[1],nr_gpus=2)

neuralnet.fit(X, y)

neuralnet.get_model_summary()

# Test prediction
# pred = neuralnet.predict(X[0:10, :], 'weights-01-5.4370.hdf5')
# pred_class = np.argmax(pred, axis=1)

# Map it back to answers
# answers = [p._int_to_answer[c] for c in pred_class]
# print(answers) # Note: T.o.P is an thresholded uncommon answer (TODO: we should consider the second best in this case)
