from scipy.misc import imread, imresize
import numpy as np
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
import tensorflow as tf
# import matplotlib.pyplot as plt

class ResNet():

    def __init__(self, output=False):

        # Define pretrained model
        self.model = InceptionResNetV2(include_top=output, # Whether to include top fully connected layers as well
                                       weights='imagenet')


    def extract_image_features(self, filepath):

        # Load and preprocess image
        img = imresize(imread(filepath, mode='RGB'), (299, 299)).astype(np.float32)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)

        # Predict
        out = self.model.predict(img)

        # Reduce dimensionality by bilinear interpolation
        red = tf.image.resize_images(out, [4, 4])
        return tf.Session().run(red) # TODO: doesn't work


"""
# Test on the imagenet dataset
# Load labels
labels = np.loadtxt('data/imagenet_labels.txt', str, delimiter=':')

# Read image and preprocess
img = imresize(imread('images/stenshuvud.png', mode='RGB'), (299, 299)).astype(np.float32)
img = np.expand_dims(img, axis=0)
img = preprocess_input(img)

# Predict
out = ResNet(output=False, pooling=None).model.predict(img) # note: the model has three outputs
print(out.shape)

# Sort ouput
top_inds = out[0].argsort()[::-1]

# Print best three labels
for i in range(3):
    print(labels[top_inds[i]])
"""
