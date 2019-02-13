from scipy.misc import imread, imresize
import numpy as np
from keras.applications.inception_v3 import InceptionV3, preprocess_input
import tensorflow as tf
# import matplotlib.pyplot as plt

def bilinear_interpolation(X, c=2):
    out = np.zeros((X.shape[0], int(X.shape[1] / c), int(X.shape[2] / c), X.shape[3]))
    for i in range(X.shape[0]):
        for j in range(0, X.shape[1], c):
            for k in range(0, X.shape[2], c):
                block = X[i, j:(j + c), k:(k + c), :]
                block = block.reshape((c * c, X.shape[3]))
                mean = np.mean(block, axis=0)
                out[i, int(j / 2), int(k / 2), :] = mean
    return out

class InceptionNet():

    def __init__(self, output=False):

        # Define pretrained model
        self.model = InceptionV3(include_top=output, # Whether to include top fully connected layers as well
                                 weights='imagenet')


    def extract_all(self, files, nr_batches = 4):

        # Extract in batches
        out_list = []
        batch_size = int(np.floor(len(files) / nr_batches))
        for i in range(nr_batches - 1):
            # print("Batch", i)
            out_list.append(self.extract_batch(files[i * batch_size: (i + 1) * batch_size]))
        out_list.append(self.extract_batch(files[(nr_batches - 1) * batch_size:]))
        return np.vstack(out_list)

    def extract_batch(self, files):

        # Load all the images
        images = np.zeros((len(files), 299, 299, 3))
        for i in range(len(files)):
            img = imresize(imread(files[i], mode='RGB'), (299, 299)).astype(np.float32)
            # img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)
            images[i] = img

        # Predict
        out = self.model.predict(images)

        # Reduce dimensionality by linear interpolation
        out = bilinear_interpolation(out)
        print(out.shape)
        return out

    def extract_image_features(self, filepath):

        # Load and preprocess image
        img = imresize(imread(filepath, mode='RGB'), (299, 299)).astype(np.float32)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)

        # Predict
        out = self.model.predict(img)

        # Reduce dimensionality by bilinear interpolation
        out = tf.image.resize_images(out, [4, 4])
        return tf.Session().run(out)

"""
# Test on the imagenet dataset
# Load labels
labels = np.loadtxt('data/imagenet_labels.txt', str, delimiter=':')

# Read image and preprocess
img = imresize(imread('images/stenshuvud.png', mode='RGB'), (299, 299)).astype(np.float32)
img = np.expand_dims(img, axis=0)
img = preprocess_input(img)

# Predict
out = InceptionNet(output=False).model.predict(img) # note: the model has three outputs

print(out.shape)

# Sort ouput

top_inds = out[0].argsort()[::-1]

# Print best three labels
for i in range(3):
    print(labels[top_inds[i]])
"""