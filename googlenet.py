from scipy.misc import imread, imresize
import numpy as np
import keras
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation, concatenate
from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import SGD
from googlenet_custom_layers import PoolHelper,LRN


def create_googlenet(weights_path=None):
    # creates GoogLeNet a.k.a. Inception v1 (Szegedy, 2015)
    
    input = Input(shape=(3, 224, 224))
    
    conv1_7x7_s2 = Conv2D(64,kernel_size=(7,7),strides=(2,2),padding='same',activation='relu',name='conv1/7x7_s2',kernel_regularizer=l2(0.0002))(input)
    
    conv1_zero_pad = ZeroPadding2D(padding=(1, 1))(conv1_7x7_s2)
    
    pool1_helper = PoolHelper()(conv1_zero_pad)
    
    pool1_3x3_s2 = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='valid',name='pool1/3x3_s2')(pool1_helper)
    
    pool1_norm1 = LRN(name='pool1/norm1')(pool1_3x3_s2)
    
    conv2_3x3_reduce = Conv2D(64,kernel_size=(1,1),padding='same',activation='relu',name='conv2/3x3_reduce',kernel_regularizer=l2(0.0002))(pool1_norm1)
    
    conv2_3x3 = Conv2D(192,kernel_size=(3,3),padding='same',activation='relu',name='conv2/3x3',kernel_regularizer=l2(0.0002))(conv2_3x3_reduce)
    
    conv2_norm2 = LRN(name='conv2/norm2')(conv2_3x3)
    
    conv2_zero_pad = ZeroPadding2D(padding=(1, 1))(conv2_norm2)
    
    pool2_helper = PoolHelper()(conv2_zero_pad)
    
    pool2_3x3_s2 = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='valid',name='pool2/3x3_s2')(pool2_helper)
    
    
    inception_3a_1x1 = Conv2D(64,kernel_size=(1,1),padding='same',activation='relu',name='inception_3a/1x1',kernel_regularizer=l2(0.0002))(pool2_3x3_s2)
    
    inception_3a_3x3_reduce = Conv2D(96,kernel_size=(1,1),padding='same',activation='relu',name='inception_3a/3x3_reduce',kernel_regularizer=l2(0.0002))(pool2_3x3_s2)
    
    inception_3a_3x3 = Conv2D(128,kernel_size=(3,3),padding='same',activation='relu',name='inception_3a/3x3',kernel_regularizer=l2(0.0002))(inception_3a_3x3_reduce)
    
    inception_3a_5x5_reduce = Conv2D(16,kernel_size=(1,1),padding='same',activation='relu',name='inception_3a/5x5_reduce',kernel_regularizer=l2(0.0002))(pool2_3x3_s2)
    
    inception_3a_5x5 = Conv2D(32,kernel_size=(5,5),padding='same',activation='relu',name='inception_3a/5x5',kernel_regularizer=l2(0.0002))(inception_3a_5x5_reduce)
    
    inception_3a_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same',name='inception_3a/pool')(pool2_3x3_s2)
    
    inception_3a_pool_proj = Conv2D(32,kernel_size=(1,1),padding='same',activation='relu',name='inception_3a/pool_proj',kernel_regularizer=l2(0.0002))(inception_3a_pool)
    
    #inception_3a_output = merge([inception_3a_1x1,inception_3a_3x3,inception_3a_5x5,inception_3a_pool_proj],mode='concat',concat_axis=1,name='inception_3a/output')
    inception_3a_output = concatenate([inception_3a_1x1,inception_3a_3x3,inception_3a_5x5,inception_3a_pool_proj], axis=1)
    
    
    inception_3b_1x1 = Conv2D(128,kernel_size=(1,1),padding='same',activation='relu',name='inception_3b/1x1',kernel_regularizer=l2(0.0002))(inception_3a_output)
    
    inception_3b_3x3_reduce = Conv2D(128,kernel_size=(1,1),padding='same',activation='relu',name='inception_3b/3x3_reduce',kernel_regularizer=l2(0.0002))(inception_3a_output)
    
    inception_3b_3x3 = Conv2D(192,kernel_size=(3,3),padding='same',activation='relu',name='inception_3b/3x3',kernel_regularizer=l2(0.0002))(inception_3b_3x3_reduce)
    
    inception_3b_5x5_reduce = Conv2D(32,kernel_size=(1,1),padding='same',activation='relu',name='inception_3b/5x5_reduce',kernel_regularizer=l2(0.0002))(inception_3a_output)
    
    inception_3b_5x5 = Conv2D(96,kernel_size=(5,5),padding='same',activation='relu',name='inception_3b/5x5',kernel_regularizer=l2(0.0002))(inception_3b_5x5_reduce)
    
    inception_3b_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same',name='inception_3b/pool')(inception_3a_output)
    
    inception_3b_pool_proj = Conv2D(64,kernel_size=(1,1),padding='same',activation='relu',name='inception_3b/pool_proj',kernel_regularizer=l2(0.0002))(inception_3b_pool)
    
    # inception_3b_output = merge([inception_3b_1x1,inception_3b_3x3,inception_3b_5x5,inception_3b_pool_proj],mode='concat',concat_axis=1,name='inception_3b/output')
    inception_3b_output = concatenate([inception_3b_1x1,inception_3b_3x3,inception_3b_5x5,inception_3b_pool_proj], axis=1)
    
    inception_3b_output_zero_pad = ZeroPadding2D(padding=(1, 1))(inception_3b_output)
    
    pool3_helper = PoolHelper()(inception_3b_output_zero_pad)
    
    pool3_3x3_s2 = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='valid',name='pool3/3x3_s2')(pool3_helper)
    
    
    inception_4a_1x1 = Conv2D(192,kernel_size=(1,1),padding='same',activation='relu',name='inception_4a/1x1',kernel_regularizer=l2(0.0002))(pool3_3x3_s2)
    
    inception_4a_3x3_reduce = Conv2D(96,kernel_size=(1,1),padding='same',activation='relu',name='inception_4a/3x3_reduce',kernel_regularizer=l2(0.0002))(pool3_3x3_s2)
    
    inception_4a_3x3 = Conv2D(208,kernel_size=(3,3),padding='same',activation='relu',name='inception_4a/3x3',kernel_regularizer=l2(0.0002))(inception_4a_3x3_reduce)
    
    inception_4a_5x5_reduce = Conv2D(16,kernel_size=(1,1),padding='same',activation='relu',name='inception_4a/5x5_reduce',kernel_regularizer=l2(0.0002))(pool3_3x3_s2)
    
    inception_4a_5x5 = Conv2D(48,kernel_size=(5,5),padding='same',activation='relu',name='inception_4a/5x5',kernel_regularizer=l2(0.0002))(inception_4a_5x5_reduce)
    
    inception_4a_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same',name='inception_4a/pool')(pool3_3x3_s2)
    
    inception_4a_pool_proj = Conv2D(64,kernel_size=(1,1),padding='same',activation='relu',name='inception_4a/pool_proj',kernel_regularizer=l2(0.0002))(inception_4a_pool)
    
    # inception_4a_output = merge([inception_4a_1x1,inception_4a_3x3,inception_4a_5x5,inception_4a_pool_proj],mode='concat',concat_axis=1,name='inception_4a/output')
    inception_4a_output = concatenate([inception_4a_1x1,inception_4a_3x3,inception_4a_5x5,inception_4a_pool_proj], axis=1)
    
    loss1_ave_pool = AveragePooling2D(pool_size=(5,5),strides=(3,3),name='loss1/ave_pool')(inception_4a_output)
    
    loss1_conv = Conv2D(128,kernel_size=(1,1),padding='same',activation='relu',name='loss1/conv',kernel_regularizer=l2(0.0002))(loss1_ave_pool)
    
    loss1_flat = Flatten()(loss1_conv)
    
    loss1_fc = Dense(1024,activation='relu',name='loss1/fc',kernel_regularizer=l2(0.0002))(loss1_flat)
    
    loss1_drop_fc = Dropout(0.7)(loss1_fc)
    
    loss1_classifier = Dense(1000,name='loss1/classifier',kernel_regularizer=l2(0.0002))(loss1_drop_fc)
    
    loss1_classifier_act = Activation('softmax')(loss1_classifier)
    
    
    inception_4b_1x1 = Conv2D(160,kernel_size=(1,1),padding='same',activation='relu',name='inception_4b/1x1',kernel_regularizer=l2(0.0002))(inception_4a_output)
    
    inception_4b_3x3_reduce = Conv2D(112,kernel_size=(1,1),padding='same',activation='relu',name='inception_4b/3x3_reduce',kernel_regularizer=l2(0.0002))(inception_4a_output)
    
    inception_4b_3x3 = Conv2D(224,kernel_size=(3,3),padding='same',activation='relu',name='inception_4b/3x3',kernel_regularizer=l2(0.0002))(inception_4b_3x3_reduce)
    
    inception_4b_5x5_reduce = Conv2D(24,kernel_size=(1,1),padding='same',activation='relu',name='inception_4b/5x5_reduce',kernel_regularizer=l2(0.0002))(inception_4a_output)
    
    inception_4b_5x5 = Conv2D(64,kernel_size=(5,5),padding='same',activation='relu',name='inception_4b/5x5',kernel_regularizer=l2(0.0002))(inception_4b_5x5_reduce)
    
    inception_4b_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same',name='inception_4b/pool')(inception_4a_output)
    
    inception_4b_pool_proj = Conv2D(64,kernel_size=(1,1),padding='same',activation='relu',name='inception_4b/pool_proj',kernel_regularizer=l2(0.0002))(inception_4b_pool)
    
    #inception_4b_output = merge([inception_4b_1x1,inception_4b_3x3,inception_4b_5x5,inception_4b_pool_proj],mode='concat',concat_axis=1,name='inception_4b_output')
    inception_4b_output = concatenate([inception_4b_1x1,inception_4b_3x3,inception_4b_5x5,inception_4b_pool_proj], axis=1)
    
    
    inception_4c_1x1 = Conv2D(128,kernel_size=(1,1),padding='same',activation='relu',name='inception_4c/1x1',kernel_regularizer=l2(0.0002))(inception_4b_output)
    
    inception_4c_3x3_reduce = Conv2D(128,kernel_size=(1,1),padding='same',activation='relu',name='inception_4c/3x3_reduce',kernel_regularizer=l2(0.0002))(inception_4b_output)
    
    inception_4c_3x3 = Conv2D(256,kernel_size=(3,3),padding='same',activation='relu',name='inception_4c/3x3',kernel_regularizer=l2(0.0002))(inception_4c_3x3_reduce)
    
    inception_4c_5x5_reduce = Conv2D(24,kernel_size=(1,1),padding='same',activation='relu',name='inception_4c/5x5_reduce',kernel_regularizer=l2(0.0002))(inception_4b_output)
    
    inception_4c_5x5 = Conv2D(64,kernel_size=(5,5),padding='same',activation='relu',name='inception_4c/5x5',kernel_regularizer=l2(0.0002))(inception_4c_5x5_reduce)
    
    inception_4c_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same',name='inception_4c/pool')(inception_4b_output)
    
    inception_4c_pool_proj = Conv2D(64,kernel_size=(1,1),padding='same',activation='relu',name='inception_4c/pool_proj',kernel_regularizer=l2(0.0002))(inception_4c_pool)
    
    #inception_4c_output = merge([inception_4c_1x1,inception_4c_3x3,inception_4c_5x5,inception_4c_pool_proj],mode='concat',concat_axis=1,name='inception_4c/output')
    inception_4c_output = concatenate([inception_4c_1x1,inception_4c_3x3,inception_4c_5x5,inception_4c_pool_proj], axis=1)
    
    
    inception_4d_1x1 = Conv2D(112,kernel_size=(1,1),padding='same',activation='relu',name='inception_4d/1x1',kernel_regularizer=l2(0.0002))(inception_4c_output)
    
    inception_4d_3x3_reduce = Conv2D(144,kernel_size=(1,1),padding='same',activation='relu',name='inception_4d/3x3_reduce',kernel_regularizer=l2(0.0002))(inception_4c_output)
    
    inception_4d_3x3 = Conv2D(288,kernel_size=(3,3),padding='same',activation='relu',name='inception_4d/3x3',kernel_regularizer=l2(0.0002))(inception_4d_3x3_reduce)
    
    inception_4d_5x5_reduce = Conv2D(32,kernel_size=(1,1),padding='same',activation='relu',name='inception_4d/5x5_reduce',kernel_regularizer=l2(0.0002))(inception_4c_output)
    
    inception_4d_5x5 = Conv2D(64,kernel_size=(5,5),padding='same',activation='relu',name='inception_4d/5x5',kernel_regularizer=l2(0.0002))(inception_4d_5x5_reduce)
    
    inception_4d_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same',name='inception_4d/pool')(inception_4c_output)
    
    inception_4d_pool_proj = Conv2D(64,kernel_size=(1,1),padding='same',activation='relu',name='inception_4d/pool_proj',kernel_regularizer=l2(0.0002))(inception_4d_pool)
    
    #inception_4d_output = merge([inception_4d_1x1,inception_4d_3x3,inception_4d_5x5,inception_4d_pool_proj],mode='concat',concat_axis=1,name='inception_4d/output')
    inception_4d_output = concatenate([inception_4d_1x1,inception_4d_3x3,inception_4d_5x5,inception_4d_pool_proj], axis=1)

    
    loss2_ave_pool = AveragePooling2D(pool_size=(5,5),strides=(3,3),name='loss2/ave_pool')(inception_4d_output)
    
    loss2_conv = Conv2D(128,kernel_size=(1,1),padding='same',activation='relu',name='loss2/conv',kernel_regularizer=l2(0.0002))(loss2_ave_pool)
    
    loss2_flat = Flatten()(loss2_conv)
    
    loss2_fc = Dense(1024,activation='relu',name='loss2/fc',kernel_regularizer=l2(0.0002))(loss2_flat)
    
    loss2_drop_fc = Dropout(0.7)(loss2_fc)
    
    loss2_classifier = Dense(1000,name='loss2/classifier',kernel_regularizer=l2(0.0002))(loss2_drop_fc)
    
    loss2_classifier_act = Activation('softmax')(loss2_classifier)
    
    
    inception_4e_1x1 = Conv2D(256,kernel_size=(1,1),padding='same',activation='relu',name='inception_4e/1x1',kernel_regularizer=l2(0.0002))(inception_4d_output)
    
    inception_4e_3x3_reduce = Conv2D(160,kernel_size=(1,1),padding='same',activation='relu',name='inception_4e/3x3_reduce',kernel_regularizer=l2(0.0002))(inception_4d_output)
    
    inception_4e_3x3 = Conv2D(320,kernel_size=(3,3),padding='same',activation='relu',name='inception_4e/3x3',kernel_regularizer=l2(0.0002))(inception_4e_3x3_reduce)
    
    inception_4e_5x5_reduce = Conv2D(32,kernel_size=(1,1),padding='same',activation='relu',name='inception_4e/5x5_reduce',kernel_regularizer=l2(0.0002))(inception_4d_output)
    
    inception_4e_5x5 = Conv2D(128,kernel_size=(5,5),padding='same',activation='relu',name='inception_4e/5x5',kernel_regularizer=l2(0.0002))(inception_4e_5x5_reduce)
    
    inception_4e_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same',name='inception_4e/pool')(inception_4d_output)
    
    inception_4e_pool_proj = Conv2D(128,kernel_size=(1,1),padding='same',activation='relu',name='inception_4e/pool_proj',kernel_regularizer=l2(0.0002))(inception_4e_pool)
    
    #inception_4e_output = merge([inception_4e_1x1,inception_4e_3x3,inception_4e_5x5,inception_4e_pool_proj],mode='concat',concat_axis=1,name='inception_4e/output')
    inception_4e_output = concatenate([inception_4e_1x1,inception_4e_3x3,inception_4e_5x5,inception_4e_pool_proj], axis=1)
    
    
    inception_4e_output_zero_pad = ZeroPadding2D(padding=(1, 1))(inception_4e_output)
    
    pool4_helper = PoolHelper()(inception_4e_output_zero_pad)
    
    pool4_3x3_s2 = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='valid',name='pool4/3x3_s2')(pool4_helper)
    
    
    inception_5a_1x1 = Conv2D(256,kernel_size=(1,1),padding='same',activation='relu',name='inception_5a/1x1',kernel_regularizer=l2(0.0002))(pool4_3x3_s2)
    
    inception_5a_3x3_reduce = Conv2D(160,kernel_size=(1,1),padding='same',activation='relu',name='inception_5a/3x3_reduce',kernel_regularizer=l2(0.0002))(pool4_3x3_s2)
    
    inception_5a_3x3 = Conv2D(320,kernel_size=(3,3),padding='same',activation='relu',name='inception_5a/3x3',kernel_regularizer=l2(0.0002))(inception_5a_3x3_reduce)
    
    inception_5a_5x5_reduce = Conv2D(32,kernel_size=(1,1),padding='same',activation='relu',name='inception_5a/5x5_reduce',kernel_regularizer=l2(0.0002))(pool4_3x3_s2)
    
    inception_5a_5x5 = Conv2D(128,kernel_size=(5,5),padding='same',activation='relu',name='inception_5a/5x5',kernel_regularizer=l2(0.0002))(inception_5a_5x5_reduce)
    
    inception_5a_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same',name='inception_5a/pool')(pool4_3x3_s2)
    
    inception_5a_pool_proj = Conv2D(128,kernel_size=(1,1),padding='same',activation='relu',name='inception_5a/pool_proj',kernel_regularizer=l2(0.0002))(inception_5a_pool)
    
    #inception_5a_output = merge([inception_5a_1x1,inception_5a_3x3,inception_5a_5x5,inception_5a_pool_proj],mode='concat',concat_axis=1,name='inception_5a/output')
    inception_5a_output = concatenate([inception_5a_1x1,inception_5a_3x3,inception_5a_5x5,inception_5a_pool_proj], axis=1)
    
    
    inception_5b_1x1 = Conv2D(384,kernel_size=(1,1),padding='same',activation='relu',name='inception_5b/1x1',kernel_regularizer=l2(0.0002))(inception_5a_output)
    
    inception_5b_3x3_reduce = Conv2D(192,kernel_size=(1,1),padding='same',activation='relu',name='inception_5b/3x3_reduce',kernel_regularizer=l2(0.0002))(inception_5a_output)
    
    inception_5b_3x3 = Conv2D(384,kernel_size=(3,3),padding='same',activation='relu',name='inception_5b/3x3',kernel_regularizer=l2(0.0002))(inception_5b_3x3_reduce)
    
    inception_5b_5x5_reduce = Conv2D(48,kernel_size=(1,1),padding='same',activation='relu',name='inception_5b/5x5_reduce',kernel_regularizer=l2(0.0002))(inception_5a_output)
    
    inception_5b_5x5 = Conv2D(128,kernel_size=(5,5),padding='same',activation='relu',name='inception_5b/5x5',kernel_regularizer=l2(0.0002))(inception_5b_5x5_reduce)
    
    inception_5b_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same',name='inception_5b/pool')(inception_5a_output)
    
    inception_5b_pool_proj = Conv2D(128,kernel_size=(1,1),padding='same',activation='relu',name='inception_5b/pool_proj',kernel_regularizer=l2(0.0002))(inception_5b_pool)
    
    #inception_5b_output = merge([inception_5b_1x1,inception_5b_3x3,inception_5b_5x5,inception_5b_pool_proj],mode='concat',concat_axis=1,name='inception_5b/output')
    inception_5b_output = concatenate([inception_5b_1x1,inception_5b_3x3,inception_5b_5x5,inception_5b_pool_proj], axis=1)
    
    
    pool5_7x7_s1 = AveragePooling2D(pool_size=(7,7),strides=(1,1),name='pool5/7x7_s2')(inception_5b_output)
    
    loss3_flat = Flatten()(pool5_7x7_s1)
    
    pool5_drop_7x7_s1 = Dropout(0.4)(loss3_flat)
    
    loss3_classifier = Dense(1000,name='loss3/classifier',kernel_regularizer=l2(0.0002))(pool5_drop_7x7_s1)
    
    loss3_classifier_act = Activation('softmax',name='prob')(loss3_classifier)
    
    
    googlenet = Model(input=input, output=[loss1_classifier_act,loss2_classifier_act,loss3_classifier_act,pool5_drop_7x7_s1])
    
    if weights_path:
        googlenet.load_weights(weights_path)
    
    return googlenet


class GoogleNet():

    def __init__(self):

        # Set keras settings
        # Run with the following command: KERAS_BACKEND=theano or change the default settings
        keras.backend.set_image_data_format('channels_first')
        self.model = model = create_googlenet('weights/googlenet_weights.h5')
        self.sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(optimizer=self.sgd, loss='categorical_crossentropy')


    def extract_image_features(self, filepath):

        # Read image
        img = imresize(imread(filepath, mode='RGB'), (224, 224)).astype(np.float32)
        
        # Subtract the mean (channelwise)
        # TODO: don't know if it is necessary
        img[:, :, 0] -= np.mean(img[:, :, 0])
        img[:, :, 1] -= np.mean(img[:, :, 1])
        img[:, :, 2] -= np.mean(img[:, :, 2])
        img[:,:,[0,1,2]] = img[:,:,[2,1,0]]
        img = img.transpose((2, 0, 1))
        img = np.expand_dims(img, axis=0)

        # Produce output
        out = self.model.predict(img)
        return out[3] 


if __name__ == "__main__":

    # Set keras settings
    # Run with the following command: KERAS_BACKEND=theano or change the default settings
    keras.backend.set_image_data_format('channels_first')

    # Load labels
    labels = np.loadtxt('data/imagenet_labels.txt', str, delimiter=':')

    # Read image
    img = imresize(imread('images/stenshuvud.png', mode='RGB'), (224, 224)).astype(np.float32)
    
    # Subtract the mean (channelwise)
    # TODO: don't know if it is necessary
    img[:, :, 0] -= np.mean(img[:, :, 0]) #123.68
    img[:, :, 1] -= np.mean(img[:, :, 1]) #116.779
    img[:, :, 2] -= np.mean(img[:, :, 2]) #103.939
    img[:,:,[0,1,2]] = img[:,:,[2,1,0]]
    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, axis=0)

    # Test pretrained model
    model = create_googlenet('weights/googlenet_weights.h5')
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')
    out = model.predict(img) # note: the model has three outputs
    
    # Sort ouput
    top_inds = out[2].argsort()[::-1]
    
    # Print best three labels
    for i in range(3):
        print(labels[top_inds[0][top_inds[0].shape[0] - i - 1]])
