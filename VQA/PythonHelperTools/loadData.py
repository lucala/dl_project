# coding: utf-8

from VQA.PythonHelperTools.vqaTools.vqa import VQA
import os
import numpy as np
import json


versionType ='' # this should be '' when using VQA v2.0 dataset
dataType    ='mscoco'  # 'mscoco' only for v1.0. 'mscoco' for real and 'abstract_v002' for abstract for v1.0.

# taskType: 'OpenEnded' only for v2.0. 'OpenEnded' or 'MultipleChoice' for v1.0
# dataSubType: 'val2014' or 'train2014'
def loadData(cut_data=1, taskType='OpenEnded', dataSubType='val2014', dataDir='VQA', answer_type='all'):

	# Define paths
	annFile     ='%s/Annotations/%s%s_%s_annotations.json'%(dataDir, versionType, dataType, dataSubType)
	quesFile    ='%s/Questions/%s%s_%s_%s_questions.json'%(dataDir, versionType, taskType, dataType, dataSubType)
	imgDir 		= '%s/Images/%s/%s/' %(dataDir, dataType, dataSubType)

	if dataSubType == 'test2015' or dataSubType == 'test-dev2015':
		questions = json.load(open(quesFile, 'r'))['questions']
		images = []
		for a in questions:
			imgId = a['image_id']
			imgDir 		= '%s/Images/%s/%s/' %(dataDir, dataType, 'test2015')
			imgFilename = imgDir + 'COCO_test2015_'+ str(imgId).zfill(12) + '.jpg'
			images.append(imgFilename)
		return [[], images, questions]

	# initialize VQA api for QA annotations
	vqa=VQA(annFile, quesFile)

	# Load all the possible questions
	annIds = vqa.getQuesIds()
	anns = vqa.loadQA(annIds)
	questions = vqa.questions["questions"]

	if cut_data < 1:
		cut_amount = int(len(anns) * cut_data)
		anns = anns[0:cut_amount]
		questions = questions[0:cut_amount]
	elif cut_data > 1:
		anns = anns[0:cut_data]
		questions = questions[0:cut_data]

	# Create different matrices
	images = []
	for a in anns:
		imgId = a['image_id']
		imgFilename = imgDir + 'COCO_' + dataSubType + '_'+ str(imgId).zfill(12) + '.jpg'
		images.append(imgFilename)

	# Filter based on answer type
	ann_filtered = []
	questions_filtered = []
	images_filtered = []
	for i in range(len(anns)):
		if answer_type == 'all' or get_type(anns[i]) == answer_type:
			ann_filtered.append(anns[i])
			questions_filtered.append(questions[i])
			images_filtered.append(images[i])

	return [ann_filtered, images_filtered, questions_filtered]

def get_type(ann):

	if ann['answer_type'] == 'yes/no':
		return 'yes/no'
	# elif ann['answer_type']=='number' and (len([int(s) for s in ann['multiple_choice_answer'].split() if s.isdigit()]) > 0 or is_number_word(ann['multiple_choice_answer'])):
	elif ann['answer_type']=='number' and ann['multiple_choice_answer'].isdigit():
		return 'number'
	else:
		return 'other'

def is_number_word(word):
	return word == 'many' or word == 'lot' or word == 'lots' or word == 'hundreds' or word == 'dozens' or word == 'plenty'