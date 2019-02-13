from scipy.misc import imread, imresize
import numpy as np
import os
from PreprocessingQuestions import PreprocessingQuestions, PreprocessingAnswers
from VQA.PythonHelperTools.loadData import loadData
from inception_net import InceptionNet
#from net import GoogleNet
from googlenet import GoogleNet
import json

# Prepare data for training/testing/validating
# Possibilities for subset: 'val', 'train', 'test'
# TODO: test is not supported for now
# cut_data: cut the data to the first ten samples, just for debugging
class PrepareData():

    def __init__(self, path_images='data_vqa_feat', path_questions='vqa_data_share', subset='val', cut_data=1, output_path='data', pad_length=32,
                 question_threshold=0, answer_threshold=0, answers_sparse=False, questions_sparse=True):
        self.path_images = path_images
        self.path_questions = path_questions
        self.subset = subset
        self.cut_data = cut_data
        self.output_path = output_path
        self.pad_length = pad_length
        self.question_threshold = question_threshold
        self.answer_threshold = answer_threshold
        self.answers_sparse = answers_sparse
        self.questions_sparse = questions_sparse


    # Return the features extracted from GoogleNet and the list of corresponding images
    def load_image_features(self):

        # Get file paths
        if self.subset == 'val':
            data = 'coco_val2014_googlenetFCdense_feat.dat.npy'
            imags = 'coco_val2014_googlenetFCdense_imglist.dat.txt'
        elif self.subset == 'test':
            data = 'coco_test2015_googlenetFCdense_feat.dat.npy'
            imags = 'coco_test2015_googlenetFCdense_imglist.dat.txt'
        else:
            data = 'coco_train2014_googlenetFCdense_feat.dat.npy'
            imags = 'coco_train2014_googlenetFCdense_imglist.dat.txt'

        data = self.path_images + '/' + data
        imags = self.path_images + '/' + imags

        # Load data
        image_matrix = np.load(data)
        with open(imags) as f:
            content = f.readlines()
        image_list = np.array([x.strip() for x in content])

        # If set is trainval then load everything
        if self.subset == 'trainval':

            # Load val set
            image_mat_val = np.load(self.path_images + '/coco_val2014_googlenetFCdense_feat.dat.npy')
            with open(self.path_images + '/coco_val2014_googlenetFCdense_imglist.dat.txt') as f:
                content = f.readlines()
            image_list_val = np.array([x.strip() for x in content])

            # Concatenate them
            image_matrix = np.concatenate((image_matrix, image_mat_val), axis=0)
            image_list = np.concatenate((image_list, image_list_val), axis=0)

        if self.cut_data < 1:
            self.cut_amount = int(len(image_list)*self.cut_data)
            image_matrix = image_matrix[0:self.cut_amount, :]
            image_list = image_list[0:self.cut_amount]
        if self.cut_data > 1:
            image_matrix = image_matrix[0:10, :]
            image_list = image_list[0:10]

        return [image_matrix, image_list]

    # Filter and preprocess questions
    def preprocess_questions(self):

        # Get file paths
        if self.subset == 'test':
            print("TODO: not supported for the moment")
        else:
            questions = 'coco_trainval2014_question.txt'
            answers = 'coco_trainval2014_allanswer.txt'
            imags = 'coco_trainval2014_imglist.txt'
            types = 'coco_trainval2014_answer_type.txt'
        questions = self.path_questions + '/' + questions
        answers = self.path_questions + '/' + answers
        imags = self.path_questions + '/' + imags
        answer_types = self.path_questions + '/' + types

        # Load indices mapping questions to images
        with open(imags) as f:
            content = f.readlines()
        img_indices = np.array([x.strip() for x in content]) 

        # Load questions
        with open(questions) as f:
            content = f.readlines()
        questions = np.array([x.strip() for x in content])

        # Load answers
        with open(answers) as f:
            content = f.readlines()
        answers = np.array([x.strip() for x in content])

        # Load answer types
        with open(answer_types) as f:
            content = f.readlines()
        answer_types = np.array([x.strip() for x in content])

        # Preprocess answers
        # TODO: think of a better way for our model
        # Maybe more answers are possible for the same question
        answers = self.majority_voting(answers)

        # Filter questions
        indices = []
        for i in range(img_indices.size):
            if self.subset == 'trainval' or self.subset in img_indices[i]:
                indices.append(i)
        indices = np.array(indices)
        img_indices = img_indices[indices]
        questions = questions[indices]
        answers = answers[indices]
        answer_types = answer_types[indices]
      
        return [img_indices, questions, answers, answer_types]


    # Load question features
    def load_question_features(self):

        # Preprocess questions
        img_indices, questions, answers, answer_types = self.preprocess_questions()

        # Sort questions (image features are already sorted)
        perm = np.argsort(img_indices)
        img_indices = img_indices[perm]
        questions = questions[perm]
        answers = answers[perm]
        answer_types = answer_types[perm]

        if self.cut_data < 1:
            img_indices = img_indices[0:self.cut_amount*3]
            questions = questions[0:self.cut_amount*3]
            answers = answers[0:self.cut_amount*3]
            answer_types = answer_types[0:self.cut_amount*3]
        if self.cut_data > 1:
            img_indices = img_indices[0:30]
            questions = questions[0:30]
            answers = answers[0:30]
            answer_types = answer_types[0:30]

        # Save questions and answers to file
        np.savetxt(self.output_path + '/questions_permuted.txt', questions, fmt="%s")
        np.savetxt(self.output_path + '/answers_permuted.txt', answers, fmt="%s")

        # Transform questions using bag of words
        questionPreprocesser = PreprocessingQuestions(filePath=self.output_path + '/questions_permuted.txt',
                                                      pad_length=self.pad_length,
                                                      threshold=self.question_threshold)
        self._question_dict = questionPreprocesser._word_dict
        self._original_questions = questions
        questionPreprocesser.preprocess()
        if self.questions_sparse:
            questions = questionPreprocesser.to_indices()
        else:
            questions = questionPreprocesser.onehotvectormatrix()
        
        # Transform answers using bag of words
        answerPreprocesser = PreprocessingAnswers(filePath=self.output_path + '/answers_permuted.txt',
                                                    pad_length=1, # Needed if sparse_categorical_cross_entropy loss is used
                                                    threshold=self.answer_threshold)
        answerPreprocesser.preprocess()
        self._answer_to_int = answerPreprocesser._word_dict # Save dictionary
        self._int_to_answer = answerPreprocesser._rev_dict # Save inverse dictionary
        if self.answers_sparse:
            answers = answerPreprocesser.to_indices()
        else:
            answers = answerPreprocesser.onehotvectormatrix()
        return [img_indices, questions, answers, answer_types]


    # Do majority voting on answers
    def majority_voting(self, answers):
        
        new_answers = []
        for a in answers:

            # Split list of possible answers into a list
            possibilities = a.split(",")
            
            # Create dictionary of possible answers
            dic = {}
            for p in possibilities:
                if p in dic.keys():
                    dic[p] = dic[p] + 1
                else:
                    dic[p] = 1

            # Get the answer with maximum votes
            max_answer = possibilities[0]
            max_votes = dic[max_answer]
            for k in dic.keys():
                if dic[k] > max_votes:
                    max_votes = dic[k]
                    max_answer = k
            new_answers.append(max_answer)
        return np.array(new_answers)


    # Load data
    def load_data(self):

        # Load image features
        image_matrix, image_list = self.load_image_features()

        # Load preprocessed questions
        img_indices, questions, answers, answer_types = self.load_question_features()
        
        # Get the dictionary size for the questions (+1 for the "empty 0" word)
        # Note that the result only makes sense if questions_sparse = True
        dic_size = np.max(questions) + 1

        # Repeat the image features in order to match with the questions
        image_list = np.repeat(image_list, 3)
        image_matrix = np.repeat(image_matrix, 3, axis=0)

        # Concatenate image and questions features to form the input
        X = np.hstack([image_matrix, questions])
        return [X, answers, dic_size, answer_types]
