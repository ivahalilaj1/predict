# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import numpy as np
from .Unet import Unet
import SimpleITK as sitk
import cv2
import os
#Plot graphs
import matplotlib.pyplot as plt
from django.conf import settings as djangoSettings

#Keras & TF
import keras
import keras.backend as K
from keras.models import load_model
#Open model from JSON, load weights
#json_file = open(os.path.join(g_path,'ax_model.json'), 'r')
#loaded_model_json = json_file.read()
#json_file.close()
#model1 = keras.models.model_from_json(loaded_model_json)
#model1 = Unet((512,512,1))
#model1 = load_model(os.path.join(g_path, 'New_model.h5'))
#weights = os.path.join(g_path, 'New_model')
#model1.load_weights(weights)



def Get_segmentation(dicom_array):
    #Open model from JSON, load weights
    json_file = open(os.path.join(djangoSettings.STATIC_ROOT, 'ax_model_0.64_val', 'ax_model.json'))
    loaded_model_json = json_file.read()
    json_file.close()
    model1 = keras.models.model_from_json(loaded_model_json)
    #weights = os.path.join(djangoSettings.STATIC_ROOT, 'ax_model_0.64_val','mixed_weights_after_0.75.h5')
    weights = "https://drive.google.com/open?id=1TuHegq9mEoreAbAkIwVEkyGRWjUnZxVo"
    model1.load_weights(weights)
    #resize img
    temp_image_array = cv2.resize(np.squeeze(dicom_array),dsize=(512,512),interpolation = cv2.INTER_CUBIC)
    #Apply lung filter
    temp_image_array[temp_image_array<-1000] = -1000
    temp_image_array[temp_image_array>150] = 150
    #Normalize image
    temp_image_array-=-908.
    temp_image_array/=266.
    predictions = model1.predict(temp_image_array.reshape(1,512,512,1)).reshape(512,512)

    fig,ax = plt.subplots(1,2,figsize=(20,10))
    ax[0].imshow(dicom_array,cmap='bone')
    ax[0].axis('off')
    ax[0].set_title('Initial image',fontsize=25)
    line1 = ax[0].contour(predictions>0.9,colors='red',linewidths=1)#linestyles='dashed',dashes=[6, 2])
    #for c in line1.collections: #to make the line dashed
    #    c.set_dashes([(0, (15.0, 10.0))])
    h1,_ = line1.legend_elements()
    ax[0].legend([h1[0]], ['Predicted contour'], loc='lower left',fontsize=20,fancybox=True, framealpha=1)
    ax[1].imshow(predictions,cmap='bone')
    ax[1].axis('off')
    ax[1].set_title('Predicted mask',fontsize=25)
    plt.savefig( os.path.join('./static/images/tmp','image_and_predicted_mask.jpg'))
    plt.close()
    return os.path.join('./static/images/tmp','image_and_predicted_mask.jpg')


#Just to test the function
#dicom_img = r'Z:\Data\TCIA-CT-Lung3-Genomics-NSCL\LUNG3-01\Image (0048).dcm'
#temp_data = sitk.ReadImage(dicom_img)
#dicom_array = np.squeeze(np.array(sitk.GetArrayFr
# omImage(temp_data),np.float))
#
#from time import time
#
#time1 = time()
#predictions = Get_segmentation(dicom_array)
#print(time()-time1)
