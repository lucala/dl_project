# coding: utf-8

import sys
dataDir = 'VQA'
from VQA.PythonHelperTools.vqaTools.vqa import VQA
from VQA.PythonEvaluationTools.vqaEvaluation.vqaEval import VQAEval
import json
import random
import os

# set up file names and paths
versionType ='' # this should be '' when using VQA v2.0 dataset
dataType    ='mscoco'  # 'mscoco' only for v1.0. 'mscoco' for real and 'abstract_v002' for abstract for v1.0. 
fileTypes   = ['results', 'accuracy', 'evalQA', 'evalQuesType', 'evalAnsType'] 

def evaluate_results(taskType='OpenEnded', dataSubType='train2014', resultType='baseline', verbose=True):

	# Build paths
	annFile     ='%s/Annotations/%s%s_%s_annotations.json'%(dataDir, versionType, dataType, dataSubType)
	quesFile    ='%s/Questions/%s%s_%s_%s_questions.json'%(dataDir, versionType, taskType, dataType, dataSubType)
	imgDir      ='%s/Images/%s/%s/' %(dataDir, dataType, dataSubType)

	[resFile, accuracyFile, evalQAFile, evalQuesTypeFile, evalAnsTypeFile] = ['%s/Results/%s%s_%s_%s_%s_%s.json'%(dataDir, versionType, taskType, dataType, dataSubType, \
	resultType, fileType) for fileType in fileTypes]  

	# create vqa object and vqaRes object
	vqa = VQA(annFile, quesFile)
	vqaRes = vqa.loadRes(resFile, quesFile)

	# create vqaEval object by taking vqa and vqaRes
	vqaEval = VQAEval(vqa, vqaRes, n=2)   #n is precision of accuracy (number of places after decimal), default is 2

	# evaluate results
	"""
	If you have a list of question ids on which you would like to evaluate your results, pass it as a list to below function
	By default it uses all the question ids in annotation file
	"""
	# Get list of question ids
	questionIds = [key for key in vqaRes.qa]
	vqaEval.evaluate(questionIds) 

	# print accuracies
	if verbose:
		print("\n")
		print("Overall Accuracy is: %.02f\n" %(vqaEval.accuracy['overall']))
		print("Per Question Type Accuracy is the following:")
		for quesType in vqaEval.accuracy['perQuestionType']:
			print("%s : %.02f" %(quesType, vqaEval.accuracy['perQuestionType'][quesType]))
		print("\n")
		print("Per Answer Type Accuracy is the following:")
		for ansType in vqaEval.accuracy['perAnswerType']:
			print("%s : %.02f" %(ansType, vqaEval.accuracy['perAnswerType'][ansType]))
		print("\n")

	# save evaluation results to ./Results folder
	json.dump(vqaEval.accuracy,     open(accuracyFile,     'w'))
	json.dump(vqaEval.evalQA,       open(evalQAFile,       'w'))
	json.dump(vqaEval.evalQuesType, open(evalQuesTypeFile, 'w'))
	json.dump(vqaEval.evalAnsType,  open(evalAnsTypeFile,  'w'))
	return vqaEval.accuracy['overall']