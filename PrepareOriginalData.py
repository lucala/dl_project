import numpy as np
import os
from PreprocessingQuestions import PreprocessingQuestions, PreprocessingAnswers
from VQA.PythonHelperTools.loadData import loadData
from inception_net import InceptionNet
# from resnet import ResNet
import pickle

class PrepareData():

    def __init__(self, subset='val2014', cut_data=1, output_path='data', pad_length=32, taskType='all',
                 question_threshold=0, answer_threshold=0, answers_sparse=True, questions_sparse=True,
                 image_extractor='Baseline', path_images='data_vqa_feat', originalDataDir='VQA',
                 precomputed_dic=None, answer_type='all'):
        self.subset = subset
        self.cut_data = cut_data
        self.output_path = output_path
        self.pad_length = pad_length
        self.question_threshold = question_threshold
        self.answer_threshold = answer_threshold
        self.answers_sparse = answers_sparse
        self.questions_sparse = questions_sparse
        self.taskType = taskType
        self.path_images = path_images # Only important if image_extractor = 'Baseline'
        self.originalDataDir = originalDataDir
        self.answer_type = answer_type

        # Set dictionaries to None
        self._question_dict = precomputed_dic
        self._answer_to_int = None
        self._int_to_answer = None

        # Define network to extract image features
        # TODO: handle other cases
        if image_extractor == 'InceptionNet':
            self.cnn = InceptionNet()
        elif image_extractor == 'Baseline':
            self.cnn = 'Baseline'
        elif image_extractor == 'GoogleNet':
            self.cnn = GoogleNet()
        elif image_extractor == 'RawImages':
            self.cnn = 'RawImages'


    # Return the features extracted from GoogleNet and the list of corresponding images
    def load_image_features(self, associated_questions):

        # Get file paths
        data = []
        imags = []
        if self.subset == 'trainval' or self.subset == 'val2014':
            data.append('coco_val2014_googlenetFCdense_feat.dat.npy')
            imags.append('coco_val2014_googlenetFCdense_imglist.dat.txt')
        if self.subset == 'trainval' or self.subset == 'train2014':
            data.append('coco_train2014_googlenetFCdense_feat.dat.npy')
            imags.append('coco_train2014_googlenetFCdense_imglist.dat.txt')
        else:
            data.append('coco_test2015_googlenetFCdense_feat.dat.npy')
            imags.append('coco_test2015_googlenetFCdense_imglist.dat.txt')

        data = [self.path_images + '/' + d for d in data]
        imags = [self.path_images + '/' + im for im in imags]

        # Load data
        image_matrix = np.concatenate([np.load(d) for d in data])
        image_list = []
        for im in imags:
            with open(im) as f:
                content = f.readlines()
            image_list.append(np.array([x.strip() for x in content]))
        image_list = np.concatenate(image_list)

        # Create hash map image number -> position
        mapping = {}
        for i in range(image_list.size):
            mapping[int(image_list[i].split("_")[-1])] = i

        # Retrieve correct features
        image_features = [image_matrix[mapping[int(q["image_id"])], :] for q in associated_questions]

        return image_features

    # Load question features
    def load_question_features(self, questions, answers):

        # Encode using correct format
        # questions = [q.encode('utf-8') for q in questions]
        # answers = [q.encode('utf-8') for q in answers]
        questions = [q.replace(u"\u2018", "'").replace(u"\u2019", "'") for q in questions]
        answers = [a.replace(u"\u2018", "'").replace(u"\u2019", "'") for a in answers]

        # Save questions and answers to file
        np.savetxt(self.output_path + '/questions_permuted.txt', questions, fmt="%s")
        np.savetxt(self.output_path + '/answers_permuted.txt', answers, fmt="%s")

        # Transform questions using bag of words
        questionPreprocesser = PreprocessingQuestions(filePath=self.output_path + '/questions_permuted.txt',
                                                      pad_length=self.pad_length,
                                                      threshold=self.question_threshold,
                                                      dictionary=self._question_dict)
        questionPreprocesser.preprocess()
        if self.questions_sparse:
            questions = questionPreprocesser.to_indices()
        else:
            questions = questionPreprocesser.onehotvectormatrix()
        self._question_dict = questionPreprocesser._word_dict
        
        # Transform answers using bag of words
        answerPreprocesser = PreprocessingAnswers(filePath=self.output_path + '/answers_permuted.txt',
                                                    pad_length=1, # Needed if sparse_categorical_cross_entropy loss is used
                                                    threshold=self.answer_threshold)
        answerPreprocesser.preprocess()

        # Do not overwrite dictionary
        if self._answer_to_int == None:
            self._answer_to_int = answerPreprocesser._word_dict # Save dictionary
            self._int_to_answer = answerPreprocesser._rev_dict # Save inverse dictionary

        # Case distincion
        if self.answers_sparse:
            answers = answerPreprocesser.to_indices()
        else:
            answers = answerPreprocesser.onehotvectormatrix()
        return [questions, answers]

    # Do majority voting on answers
    def majority_voting(self, answers):
        
        dic = {}
        for a in answers:
            if a["answer"] in dic.keys():
                dic[a["answer"]] = dic[a["answer"]] + 1
            else:
                dic[a["answer"]] = 1

        # Get the answer with maximum votes
        max_answer = answers[0]["answer"]
        max_votes = dic[max_answer]
        for k in dic.keys():
            if dic[k] > max_votes:
                max_votes = dic[k]
                max_answer = k
        return max_answer

    # Load test set
    def load_test_set(self, set_name='test2015'):
        self.subset = 'test2015'
        images = question_dicts = []
        if self.taskType == 'all' or self.taskType == 'OpenEnded':
            _, images_tmp, question_dicts_tmp = loadData(cut_data=self.cut_data, taskType='OpenEnded', dataSubType=set_name,
                dataDir=self.originalDataDir, answer_type=self.answer_type)
            images = images + images_tmp
            question_dicts = question_dicts + question_dicts_tmp
        if self.taskType == 'all' or self.taskType == 'MultipleChoice':
            _, images_tmp, question_dicts_tmp = loadData(cut_data=self.cut_data, taskType='MultipleChoice', dataSubType=set_name,
                dataDir=self.originalDataDir, answer_type=self.answer_type)
            images = images + images_tmp
            question_dicts = question_dicts + question_dicts_tmp

        # Extract questions
        self._original_questions = question_dicts
        questions = []
        for i in range(len(question_dicts)):

            # Extract question and answer
            questions.append(question_dicts[i]["question"])

        # Load question features using precomputed dictionary
        questions = [q.replace(u"\u2018", "'").replace(u"\u2019", "'") for q in questions]
        np.savetxt(self.output_path + '/questions_permuted.txt', questions, fmt="%s")
        questionPreprocesser = PreprocessingQuestions(filePath=self.output_path + '/questions_permuted.txt',
                                                      pad_length=self.pad_length,
                                                      threshold=self.question_threshold,
                                                      dictionary=self._question_dict)
        questionPreprocesser.preprocess()
        if self.questions_sparse:
            questions = questionPreprocesser.to_indices()
        else:
            questions = questionPreprocesser.onehotvectormatrix()

        # Get dictionary size
        self.dic_size = len(self._question_dict) + 1

        # Extract image features
        image_features = []
        if self.cnn == 'Baseline':

            # Use pre-extracted features
            image_features = self.load_image_features(question_dicts)
        elif self.cnn == 'RawImages':
            image_features = images
        else:

            # Extract features using a CNN
            for img in images:
                image_features.append(self.cnn.extract_image_features(img))

        image_features = np.vstack(image_features)

        return [image_features, questions]


    # Load data
    def load_data(self):

        # Load dataset
        annotations = images = question_dicts = []
        if self.subset == 'trainval' or self.subset == 'train2014':
            if self.taskType == 'all' or self.taskType == 'OpenEnded':
                annotations_tmp, images_tmp, question_dicts_tmp = loadData(cut_data=self.cut_data, taskType='OpenEnded', dataSubType='train2014',
                    dataDir=self.originalDataDir, answer_type=self.answer_type)
                annotations = annotations + annotations_tmp
                images = images + images_tmp
                question_dicts = question_dicts + question_dicts_tmp
            if self.taskType == 'all' or self.taskType == 'MultipleChoice':
                annotations_tmp, images_tmp, question_dicts_tmp = loadData(cut_data=self.cut_data, taskType='MultipleChoice', dataSubType='train2014',
                    dataDir=self.originalDataDir, answer_type=self.answer_type)
                annotations = annotations + annotations_tmp
                images = images + images_tmp
                question_dicts = question_dicts + question_dicts_tmp
        if self.subset == 'trainval' or self.subset == 'val2014':
            if self.taskType == 'all' or self.taskType == 'OpenEnded':
                annotations_tmp, images_tmp, question_dicts_tmp = loadData(cut_data=self.cut_data, taskType='OpenEnded', dataSubType='val2014',
                    dataDir=self.originalDataDir, answer_type=self.answer_type)
                annotations = annotations + annotations_tmp
                images = images + images_tmp
                question_dicts = question_dicts + question_dicts_tmp
            if self.taskType == 'all' or self.taskType == 'MultipleChoice':
                annotations_tmp, images_tmp, question_dicts_tmp = loadData(cut_data=self.cut_data, taskType='MultipleChoice', dataSubType='val2014',
                    dataDir=self.originalDataDir, answer_type=self.answer_type)
                annotations = annotations + annotations_tmp
                images = images + images_tmp
                question_dicts = question_dicts + question_dicts_tmp
    
        # Extract questions and associated answers
        self._original_questions = question_dicts
        questions = []
        answers = []
        for i in range(len(question_dicts)):

            # Sanity check
            assert(annotations[i]["question_id"] == question_dicts[i]["question_id"])

            # Extract question and answer
            questions.append(question_dicts[i]["question"])
            answers.append(self.majority_voting(annotations[i]["answers"]))

        # Load question and answer features
        questions, answers = self.load_question_features(questions, answers)

        # Get dictionary size
        self.dic_size = len(self._question_dict) + 1

        # Extract image features
        image_features = []
        if self.cnn == 'Baseline':

            # Use pre-extracted features
            image_features = self.load_image_features(annotations)
        elif self.cnn == 'RawImages':
            image_features = images
        else:

            # Extract features using a CNN
            for img in images:
                image_features.append(self.cnn.extract_image_features(img))

        image_features = np.vstack(image_features)

        return [image_features, questions, answers, annotations]

    def dumpDictionary(self, fileName='dictionary'):
        with open(self.output_path + '/' + fileName + '.pkl', 'wb') as f:
            pickle.dump(self, f)

    def loadDictionary(self, filePath):
        with open(filePath, 'rb') as f:
            obj = pickle.load(f)

            # Set all parameters
            self.pad_length = obj.pad_length
            self.question_threshold = obj.question_threshold
            self.answer_threshold = obj.answer_threshold
            self.questions_sparse = obj.questions_sparse
            self.answers_sparse = obj.answers_sparse
            self.cnn = obj.cnn
            self._question_dict = obj._question_dict
            self._answer_to_int = obj._answer_to_int
            self._int_to_answer = obj._int_to_answer
            self.answer_type = obj.answer_type

"""
# Load training set
p = PrepareData(path_images='data_vqa_feat', # Path to image features 
                subset='train2014', # Desired subset: either train2014 or val2014
                taskType='OpenEnded', # 'OpenEnded', 'MultipleChoice', 'all'
                cut_data=3000, # Percentage of data to use, 1 = All values, above 1=#samples for debugging
                output_path='data', # Path where we want to output temporary data
                pad_length=14, # Number of words in a question (zero padded)
                question_threshold=6, answer_threshold=3, # Keep only most common words
                answers_sparse=True, questions_sparse=True, image_extractor='RawImages')
image_features, questions, answers, annotations = p.load_data()
print("Image features", image_features.shape)
print("Question features", questions.shape)
print("Dictionary size", p.dic_size)
print("Answers", answers.shape)
print("Number of possible classes", np.max(answers))
print("Annotations", len(annotations))

# Extract features using ResNet
resnet = InceptionNet()
images = []
for i in range(len(image_features)):
    if i % 3 == 0:

        # Sanity check
        assert(image_features[i] == image_features[i + 1] == image_features[i + 2])

        # Extract features
        # features[p._original_questions[i]["image_id"]] = resnet.extract_image_features(image_features[i][0])

        images.append(image_features[i][0])

# Extract features
out = resnet.extract_all(images, nr_batches=4).astype(np.float32)
print(out.shape)

# Map to dictionary
features = {}
for i in range(len(image_features)):
    if i % 3 == 0:

        # Sanity check
        assert(image_features[i] == image_features[i + 1] == image_features[i + 2])

        # Extract features
        features[p._original_questions[i]["image_id"]] = out[int(i / 3), :, :, :]

pickle.dump(features, open('data/attentions_val_1000.pkl', 'wb'))

"""
"""
# Load test set
image_features = questions = answers = annotations = []
features = {}
image_features, questions = p.load_test_set(set_name='test2015')
print("Image features", image_features.shape)
print("Question features", questions.shape)
print("Dictionary size", p.dic_size)

# Extract features using ResNet
resnet = InceptionNet()
images = []
for i in range(len(image_features[0:3000])):
    if i % 3 == 0:

        # Sanity check
        assert(image_features[i] == image_features[i + 1] == image_features[i + 2])

        # Extract features
        # features[p._original_questions[i]["image_id"]] = resnet.extract_image_features(image_features[i][0])

        images.append(image_features[i][0])

# Extract features
out = resnet.extract_all(images, nr_batches=64).astype(np.float32)
images = []
print(out.shape)

# Map to dictionary
features = {}
for i in range(len(image_features)):
    if i % 3 == 0:

        # Sanity check
        assert(image_features[i] == image_features[i + 1] == image_features[i + 2])

        # Extract features
        features[p._original_questions[i]["image_id"]] = out[int(i / 3), :, :, :]

pickle.dump(features, open('data/attentions_test_1000.pkl', 'wb'))
"""

