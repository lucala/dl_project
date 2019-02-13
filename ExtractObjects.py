import numpy as np
from VQA.PythonHelperTools.loadData import loadData
from subprocess import call

class ExtractObjects():

    def __init__(self, output_folder='data', taskType='OpenEnded', cut_data=1, subset='train2014', originalDataDir='VQA',
                 input_fileName='images.txt', output_fileName='results.txt', class_path='darknet/data/coco.names', threshold=25):
        self.output_folder = output_folder
        self.taskType = taskType
        self.cut_data = cut_data
        self.subset = subset
        self.originalDataDir = originalDataDir
        self.input_fileName = input_fileName
        self.output_fileName = output_fileName
        self.class_path = class_path
        self.threshold = threshold

    def generate_input_file(self):

        # Get images
        # Load dataset
        images = []
        if self.taskType == 'all' or self.taskType == 'OpenEnded':
            _, images_tmp, _ = loadData(cut_data=self.cut_data, taskType='OpenEnded', dataSubType=self.subset, dataDir=self.originalDataDir)
            images = images + images_tmp
        if self.taskType == 'all' or self.taskType == 'MultipleChoice':
            _, images_tmp, _ = loadData(cut_data=self.cut_data, taskType='MultipleChoice', dataSubType=self.subset, dataDir=self.originalDataDir)
            images = images + images_tmp

        images = ['../' + i for i in images] # Adjust path
        images.append('^C')
        images = np.unique(np.array(images))
        np.savetxt(self.output_folder + '/' + self.input_fileName, images, fmt='%s')
        self.images = images[0:-1]

    def produce_output_file(self):

        # Generate script
        script = [
            'cd darknet',
            './darknet detect cfg/yolo.cfg yolo.weights '+'< ../'+self.output_folder+'/'+self.input_fileName+' > ../'+ self.output_folder+'/'+self.output_fileName,
            'cd ..']
        np.savetxt('extract_object_features.sh', np.array(script), fmt='%s')

        call(["chmod", "+x", "extract_object_features.sh"])
        call(["sh", "extract_object_features.sh"])
        call(["rm", "extract_object_features.sh"])

    def parse_result_file(self):

        # Load file
        with open(self.output_folder + '/' + self.output_fileName, "r") as f:
            lines = f.readlines()
        lines = [q.strip() for q in lines]

        # Create list of outputs per image
        image_output = []
        objects = []
        image_ids = []
        while len(lines) > 0:
            current = lines.pop(0)
            if 'Enter Image Path:' in current:
                image_output.append(objects)
                objects = []

                # Get image id
                img_id = current.split(':')[1].split('_')[-1].split('.')[0]
                if img_id != '':
                    image_ids.append(int(img_id))
            else:
                # Threshold low accuracies
                accuracy = current.split(':')[-1]
                accuracy = int(accuracy[:-1])
                if accuracy >= self.threshold:
                    objects.append(current.split(':')[-2])
        image_output.pop(0)
        
        # Transform to a dictionary of dictionaries
        object_count = {}
        for i in range(len(image_output)):
            dic = {}
            for obj in image_output[i]:
                if obj in dic.keys():
                    dic[obj] = dic[obj] + 1
                else:
                    dic[obj] = 1
            object_count[image_ids[i]] = dic
        return object_count

    def create_dictionary(self):

        # Load file
        with open(self.class_path) as f:
            lines = f.readlines()
        lines = [q.strip() for q in lines]
        
        # Create dictionary
        self.dictionary = {}
        for i in range(len(lines)):
            self.dictionary[lines[i]] = i

    def onehotvector(self, annotations, object_count=None):

        # Get object count
        if object_count == None:
            object_count = self.parse_result_file()

        # Generate dictionary
        self.create_dictionary()

        # Create onehot matrix
        onehot = np.zeros((len(annotations), len(self.dictionary)))
        for i in range(len(annotations)):
            for obj in object_count[int(annotations[i]['image_id'])].keys():
                onehot[i, self.dictionary[obj]] = object_count[int(annotations[i]['image_id'])][obj]
        return onehot

"""
eo = ExtractObjects(cut_data=1, output_fileName='objects_test.txt', threshold=25, subset='test2015')
eo.generate_input_file()
eo.produce_output_file()
# dic = eo.parse_result_file()
# onehot = eo.onehotvector([{"image_id": 78077}, {"image_id": 487025}])
# print(onehot)
"""
