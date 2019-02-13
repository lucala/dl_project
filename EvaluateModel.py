import numpy as np
from sklearn.metrics import accuracy_score
import json
from VQA.PythonEvaluationTools.evalResults import evaluate_results
from VQA.PythonHelperTools.loadData import is_number_word

class ProduceResult():


    # taskType: 'OpenEnded' or 'MultipleChoice' for v1.0
    # dataSubType: 'train2014' or 'val2014'
    def __init__(self, dictionary, rev_dictionary, output_path='VQA/Results', dataSubType='val2014', modelName='baseline', answer_count=None, original_questions=[]):
        self.dictionary = dictionary
        self.output_path = output_path
        self.dataSubType = dataSubType
        self.modelName = modelName
        self.rev_dictionary = rev_dictionary
        self.answer_count = answer_count
        self.original_questions = original_questions

    # Produce results for numbers
    def produce_results_numbers(self, prediction, questions, threshold=30, max_word='many'):

        # Generate answersX
        answers_open_ended = []
        answers_multiple_choice = []
        answers = []
        for i in range(prediction.shape[0]):
            answer = {}

            # Check if question is multiple choice
            if 'multiple_choices' in questions[i].keys():

                # Get possible choices
                choices = questions[i]['multiple_choices']
                word = max_word

                # Transform choices to numbers
                choice_nr = []
                for choice in choices:
                    numbers = [int(s) for s in choice.split() if s.isdigit()]
                    if len(numbers) == 0 and is_number_word(choice):
                        choice_nr.append(threshold)
                        word = choice
                    elif len(numbers) == 0:
                        choice_nr.append(threshold + 2)
                    else:
                        choice_nr.append(int(numbers[0]))
 
                # Get most similar answer
                if prediction[i] > threshold:
                    res = word
                else:
                    min_diff = threshold + 1
                    res = "1"
                    for j in range(len(choice_nr)):
                        dist = np.abs(prediction[i] - choice_nr[j])
                        if dist < min_diff:
                            min_diff = dist
                            res = choices[j]
                
                answer["answer"] = str(res)
                answer["question_id"] = questions[i]["question_id"]
                answers_multiple_choice.append(answer)
                answers.append(res)
            else:

                # Get best answer
                answer["answer"] = str(int(np.round(prediction[i]))) if prediction[i] <= threshold else max_word
                answer["question_id"] = questions[i]["question_id"]
                answers_open_ended.append(answer)
                answers.append(answer["answer"])

        # Save results to file
        with open(self.output_path + '/MultipleChoice_mscoco_' + self.dataSubType + '_' + self.modelName + '_results.json', 'w') as f:
            json.dump(answers_multiple_choice, f)

        with open(self.output_path + '/OpenEnded_mscoco_' + self.dataSubType + '_' + self.modelName + '_results.json', 'w') as f:
            json.dump(answers_open_ended, f)

        return answers

    # Produce the results given the answers
    def produce_results(self, prediction, questions):

        # Generate answersX
        answers_open_ended = []
        answers_multiple_choice = []
        answers = []
        for i in range(prediction.shape[0]):
            answer = {}

            # Set thresholded probability to 0
            if self.dictionary[0] == 'T.o.P!':
                prediction[i, 0] = 0

            # Check if question is multiple choice
            if 'multiple_choices' in questions[i].keys():

                # Get possible indices
                choice_idx = []
                for c in questions[i]['multiple_choices']:
                    if c in self.rev_dictionary.keys():
                        choice_idx.append(self.rev_dictionary[c])
                if len(choice_idx) == 0:
                    print("No candidate answer found")
                    choice_idx.append(0)
                choice_idx = np.array(choice_idx)

                # Prefer most common answers
                if self.answer_count != None:
                    choice_count = np.array([self.answer_count[self.dictionary[idx]] for idx in choice_idx])
                    perm = choice_count.argsort()
                    choice_idx = choice_idx[perm] 

                # Sort
                current_pred = prediction[i, choice_idx]
                pred_indices = current_pred.argsort()[::-1]
                choice_idx = choice_idx[pred_indices]

                # Get top answer
                answer["answer"] = self.dictionary[choice_idx[0]]
                answer["question_id"] = questions[i]["question_id"]
                answers_multiple_choice.append(answer)

                # Build top three answers (just for debugging)
                top_answers = ''
                for j in range(min(3, len(choice_idx))):
                    
                    answer_i = self.dictionary[choice_idx[j]]
                    confidence = prediction[i, choice_idx[j]]
                    top_answers += (answer_i + ': ' + str(confidence) + '; ')
                answers.append(top_answers)
            else:

                # Sort
                current_pred = prediction[i, :]
                pred_indices = current_pred.argsort()[::-1]

                # Get best answer
                answer["answer"] = self.dictionary[np.argmax(prediction[i])]
                answer["question_id"] = questions[i]["question_id"]
                answers_open_ended.append(answer)
            
                # Build top three answers (just for debugging)
                top_answers = ''
                for j in range(min(3, len(pred_indices))):
                    
                    answer_i = self.dictionary[pred_indices[j]]
                    confidence = prediction[i, pred_indices[j]]
                    top_answers += (answer_i + ': ' + str(confidence) + '; ')
                answers.append(top_answers)

        # Save results to file
        with open(self.output_path + '/MultipleChoice_mscoco_' + self.dataSubType + '_' + self.modelName + '_results.json', 'w') as f:
            json.dump(answers_multiple_choice, f)

        with open(self.output_path + '/OpenEnded_mscoco_' + self.dataSubType + '_' + self.modelName + '_results.json', 'w') as f:
            json.dump(answers_open_ended, f)

        return answers

    # Evaluate produced results
    def evaluate(self, taskType='all'):
        if taskType == 'all' or taskType == 'OpenEnded':
            evaluate_results(taskType='OpenEnded', dataSubType=self.dataSubType, resultType=self.modelName)
        if taskType == 'all' or taskType == 'MultipleChoice':
            evaluate_results(taskType='MultipleChoice', dataSubType=self.dataSubType, resultType=self.modelName)

    def evaluation_metric(self, y_true, y_pred):
        self.produce_results(y_pred, self.original_questions)
        return evaluate_results(taskType='OpenEnded', dataSubType=self.dataSubType, resultType=self.modelName, verbose=False)