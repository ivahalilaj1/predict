# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 16:32:28 2018

@author: S.Primakov
"""

import tensorflow as tf
#Keras & TF
import tensorflow as tf
import keras
import keras.backend as K
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, Conv3D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D 
from keras.layers import UpSampling2D, Concatenate
from keras.layers.convolutional import Convolution3D, MaxPooling3D, ZeroPadding3D , ZeroPadding3D , UpSampling3D
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import LSTM
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.utils import np_utils
from keras.utils.training_utils import multi_gpu_model
from keras.applications.imagenet_utils import preprocess_input
from keras.optimizers import Adam , SGD
from keras.models import Model
from keras.preprocessing import image
from IPython.display import clear_output

def tf_log2(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(2, dtype=numerator.dtype))
    return numerator / denominator

def dice_coef(y_true, y_pred,coeff = 1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    
    return (2. * intersection  + coeff) / (K.sum(y_true_f) + K.sum(y_pred_f) + coeff)
    
def jaccard_distance_loss(y_true, y_pred, smooth=100):
    """
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
    
    The jaccard distance loss is usefull for unbalanced datasets. This has been
    shifted so it converges on 0 and is smoothed to avoid exploding or disapearing
    gradient.
    
    Ref: https://en.wikipedia.org/wiki/Jaccard_index
    
    @url: https://gist.github.com/wassname/f1452b748efcbeb4cb9b1d059dce6f96
    @author: wassname
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + 0.000001) / (sum_ - intersection + 0.000001)
    return (1 - jac) * smooth


def custom_losses(y_true, y_pred,coeff = 1e-5):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_pred_f)+K.sum(y_true_f)-intersection
    dice_hausdorf =K.square(1-(((intersection + coeff)/(K.sum(y_true_f) + K.sum(y_pred_f) + coeff))  
                           +((intersection+coeff)/(2.*union+coeff))))
    
    if K.sum(y_true_f)==0:
        regular1 = keras.losses.binary_crossentropy(y_true,y_pred)#tf.exp(1+K.sum(y_pred_f))
        return regular1
    else:
    #regular1 = (1+(K.abs(K.sum(1-y_pred_f)-K.sum(1-y_true_f))+coeff)/(K.sum(1-y_pred_f)+K.sum(1-y_true_f)+coeff))
        cost_func = dice_hausdorf#jaccard_distance_loss(y_true, y_pred)#0.8*dice_hausdorf +0.2*keras.losses.binary_crossentropy(y_true,y_pred)#+ 0.1*regular1 
        return cost_func




def Unet(input_shape):
    
    X_input = Input(input_shape)
    print("input shape:",X_input.shape)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(X_input)
    print("conv1 shape:",conv1.shape)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    print("conv1 shape:",conv1.shape)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    print ("pool1 shape:",pool1.shape)

    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    print ("conv2 shape:",conv2.shape)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    print ("conv2 shape:",conv2.shape)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    print ("pool2 shape:",pool2.shape)

    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    print ("conv3 shape:",conv3.shape)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    print ("conv3 shape:",conv3.shape)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    print ("pool3 shape:",pool3.shape)

    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    print ("conv4 shape:",conv4.shape)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    print ("conv4 shape:",conv4.shape)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    print ("pool4 shape:",pool4.shape)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    print ("conv5 shape:",conv5.shape)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    print ("conv5 shape:",conv5.shape)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = Concatenate()([drop4,up6])#merge([drop4,up6], mode = 'concat', concat_axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    print ("conv6 shape:",conv6.shape)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
    print ("conv6 shape:",conv6.shape)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = Concatenate()([conv3,up7])#merge([conv3,up7], mode = 'concat', concat_axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    print ("conv7 shape:",conv7.shape)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
    print ("conv7 shape:",conv7.shape)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = Concatenate()([conv2,up8])#merge([conv2,up8], mode = 'concat', concat_axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    print ("conv8 shape:",conv8.shape)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    print ("conv8 shape:",conv8.shape)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = Concatenate()([conv1,up9])#merge([conv1,up9], mode = 'concat', concat_axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    print ("conv9 shape:",conv9.shape)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    print ("conv9 shape:",conv9.shape)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    print ("conv9 shape:",conv9.shape)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)
    print ("output shape:",conv10.shape)

    model = Model(inputs = X_input, outputs = conv10)

    model.compile(Adam(lr = 3.5e-6), loss = custom_losses, metrics = [dice_coef])
        #optimizer = Adam(lr = 1e-5), loss = 'binary_crossentropy', metrics = [dice_coef])
                  #Adam(lr = 3.5e-6), loss = custom_losses, metrics = [dice_coef])

    return model