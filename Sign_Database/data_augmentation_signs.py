# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 15:01:38 2021

Executing the following script will automatically create an extensive image 
database from 6 different speed limit signs using data augmentation. 

The data augmentation process involves introducing random rotation, zoom, 
contrast gaussian noise and to the images. Such actions are necessary to
decrease the impact of overfitting.

It is necessary to execute the script in the path containing folder 
Sign_database in order for it to work correctly.

@author: FilipSacha
"""

# %% Importing relevant libraries and declaring the gaussian noise function

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers


#gaussian noise multiplied then added over image: 
#noise increases with image value
def add_gaussian_noise(img_array):
    gaussian_noise_img = []
    gauss = np.random.normal(0,1,size=img_array.shape)
    gaussian_noise_img = np.clip((img_array*(1 + gauss*0.4)),0,1)
    return gaussian_noise_img



# %% Preparing input image

IMG_SIZE = 100
IMG_BATCH = 1000

types_of_limits = ['1','3','5','6','9','10']


for i in range (len(types_of_limits)):
    # Loading the images
    path_in = 'Sign_Database/' + types_of_limits[i] + '0signs/original_' + types_of_limits[i] + '0.png'
    original_sign = plt.imread(path_in)
    
    
    # Deleting png alpha channel responsible for transparency (rgba2rgb)
    original_sign = original_sign[:,:,:3]
    
    
    # Resizing images to a set size of IMG_SIZE^2
    resize = tf.keras.Sequential([
        layers.experimental.preprocessing.Resizing(IMG_SIZE, IMG_SIZE)
    ])
    
    original_sign = resize(original_sign)
    
    
    # Expanding dimension to allow further pictures to be added to the data structure
    
    original_sign = tf.expand_dims(original_sign, 0)
    
    
    
    # Data Augmentation - Random rotation, zoom, contrast, noise
    
    data_augmentation = tf.keras.Sequential([
      layers.experimental.preprocessing.RandomRotation(factor=0.15, fill_mode='constant', seed=0),
      layers.experimental.preprocessing.RandomZoom(.3, .3, fill_mode='constant'),
      layers.experimental.preprocessing.RandomContrast(.02)
    ])
    
    
    
    for x in range(IMG_BATCH):
      augmented_image = data_augmentation(original_sign)
      augmented_image = augmented_image.numpy()
      augmented_image = add_gaussian_noise(augmented_image)
      img_name = types_of_limits[i] + "0_" + str(x) + ".jpg"
      tmp_path = 'Sign_Database/' + types_of_limits[i] + '0signs/'+ img_name
      tf.keras.preprocessing.image.save_img(tmp_path,augmented_image[0])   
      


