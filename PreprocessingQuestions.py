from __future__ import division
import numpy as np
from numpy import newaxis
from collections import Counter
import re
from keras.preprocessing.sequence import pad_sequences


class PreprocessingQuestions(object):
    """This class transforms questions into a one hot vector"""

    def __init__(self, filePath, pad_length=32, threshold=0, padding='post', dictionary=None):
        self._filePath = filePath
        self._pad_length = pad_length
        self._threshold = threshold
        self._padding = padding
        self._word_dict = dictionary # Assign predefined dictionary (if None will be overwritten)

    def preprocess(self):
        question_file = open(self._filePath, "r") #, encoding="utf-8-sig")
        questions = question_file.read().lower()
        question_file.close()
        pattern = '[^a-z0-9\n \']+'
        formatted_questions = re.sub(pattern, '', questions)
        self._questions = formatted_questions

        # If the dictionary was given skip this step
        if self._word_dict != None:
            self._dict_size = len(self._word_dict) + 1
            return

        # Get cound
        self._count = {}
        for q in formatted_questions.split():
            if q in self._count.keys():
                self._count[q] = self._count[q] + 1
            else:
                self._count[q] = 1

        unique_words = sorted(set(formatted_questions.split()))

        # Filter out words below the threshold
        common_words = list(filter(lambda x : self._count[x] >= self._threshold, unique_words))
        uncommon_words = list(filter(lambda x : self._count[x] < self._threshold, unique_words))

        # Build dictionary
        word_dict = {}
        rev_dict = {}
        for idx, key in enumerate(common_words):
            word_dict[key] = idx + 1 # 0 will be assigned to uncommon words
            rev_dict[idx + 1] = key
        for key in uncommon_words:
            word_dict[key] = 0
        rev_dict[0] = "T.o.P!" # Thresholded or padding

        # word_dict = {key: idx for idx, key in enumerate(unique_words)}
        # rev_dict = {idx: key for idx, key in enumerate(unique_words)} # Compute inverse dictionary
        self._word_dict = word_dict
        self._rev_dict = rev_dict
        self._dict_size = len(word_dict) + 1
        # TODO: remove stopwords
        #wordcount = Counter(formatted_questions.split())
        #for item in wordcount.items():
            #print("{}\t{}".format(*item))

    def onehotvectormatrix(self):
        questions = self._questions.split("\n")
        questions.pop() # Remove last element from the list (empty line)
        dict_size = self._dict_size #len(self._word_dict)
        onehotmatrix = np.zeros((len(questions),dict_size))
        counter = 0;
        for question in questions:
            onehotvector = np.zeros(dict_size)
            for word in question.split():
                if word in self._word_dict.keys():
                    onehotvector[self._word_dict[word]] = 1
                else:
                    onehotvector[0] = 1
            onehotmatrix[counter] = onehotvector
            counter += 1
        return onehotmatrix

    def to_indices(self):
        questions = self._questions.split("\n")
        questions.pop() # Remove last element from the list
        nr_words = len(self._word_dict)

        # Encode every word to an integer using the precomputed dictionary
        encoded_questions = [[self._word_dict[word] if word in self._word_dict.keys() else 0 for word in question.split()] for question in questions]
        encoded_questions = pad_sequences(encoded_questions, maxlen=self._pad_length, padding=self._padding) # Pad the sequences with zeros
        return encoded_questions


class PreprocessingAnswers(object):
    """This class transforms questions into a one hot vector"""

    def __init__(self, filePath, pad_length=1, threshold=0, padding='post'):
        self._filePath = filePath
        self._pad_length = pad_length
        self._threshold = threshold
        self._padding = padding

    def preprocess(self):
        with open(self._filePath) as f:
            questions = f.readlines()
        self._questions = [q.strip() for q in questions]

        # Get cound
        self._count = {}
        for q in self._questions:
            if q in self._count.keys():
                self._count[q] = self._count[q] + 1
            else:
                self._count[q] = 1

        unique_words = sorted(set(self._questions))

        # Filter out words below the threshold
        common_words = list(filter(lambda x : self._count[x] >= self._threshold, unique_words))
        uncommon_words = list(filter(lambda x : self._count[x] < self._threshold, unique_words))

        # Build dictionary
        word_dict = {}
        rev_dict = {}
        for idx, key in enumerate(common_words):
            word_dict[key] = idx + 1 # 0 will be assigned to uncommon words
            rev_dict[idx + 1] = key
        for key in uncommon_words:
            word_dict[key] = 0
        rev_dict[0] = "T.o.P!" # Thresholded or padding

        # word_dict = {key: idx for idx, key in enumerate(unique_words)}
        # rev_dict = {idx: key for idx, key in enumerate(unique_words)} # Compute inverse dictionary
        self._word_dict = word_dict
        self._rev_dict = rev_dict
        self._dict_size = len(common_words) + 1

    def onehotvectormatrix(self):
        questions = self._questions
        dict_size = self._dict_size #len(self._word_dict)
        onehotmatrix = np.zeros((len(questions),dict_size))
        counter = 0;
        for question in questions:
            onehotvector = np.zeros(dict_size)
            onehotvector[self._word_dict[question]] = 1
            onehotmatrix[counter] = onehotvector
            counter += 1
        return onehotmatrix

    def to_indices(self):
        questions = self._questions
        nr_words = len(self._word_dict)

        # Encode every word to an integer using the precomputed dictionary
        encoded_questions = [[self._word_dict[question]] for question in questions]
        encoded_questions = pad_sequences(encoded_questions, maxlen=self._pad_length, padding=self._padding) # Pad the sequences with zeros
        return encoded_questions
