# -*- coding: utf-8 -*-
"""
Created on Sat Aug 21 10:52:31 2021

@author: filip
"""

# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os, os.path


# %% Number of files in original folder (excluding original one)
DIR = 'Sign_Database/10signs'
DATASET_SIZE = (len([name for name in os.listdir(DIR) if 
                     os.path.isfile(os.path.join(DIR, name))])) - 1

# %% Loading the image base and their corresponding labels

types_of_limits = ['1','3','5','6','9','10']


sign_images = np.empty([len(types_of_limits)*DATASET_SIZE, 100, 100, 3], dtype=int)
sign_labels = []

for i in range (len(types_of_limits)):  
    for x in range(DATASET_SIZE):
        tmp_path = 'Sign_Database/' + types_of_limits[i] + '0signs/'
        tmp_file_name = tmp_path+types_of_limits[i] + '0_' + str(x) + ".jpg"
        sign_labels.append(i)
        img=mpimg.imread(tmp_file_name)
        sign_images[x+DATASET_SIZE*i][:][:][:] = img

        
        
# %% Preparing the relevant data
length = len(sign_labels) 

RATIO = 0.8
train_cases = int(length*RATIO)
test_cases = length - train_cases


train_images = np.empty([0, 100, 100, 3], dtype=int)
test_images = np.empty([0, 100, 100, 3], dtype=int)

train_labels = []
test_labels = []

for i in range (len(types_of_limits)):  
    
    tmp = i*DATASET_SIZE
    
    train_images = np.append(train_images, sign_images[tmp:int(RATIO*DATASET_SIZE)+tmp][:][:][:], axis=0)
    test_images = np.append(test_images, sign_images[tmp+int(RATIO*DATASET_SIZE):tmp+DATASET_SIZE][:][:][:], axis=0)
    
    train_labels += sign_labels[tmp:int(RATIO*DATASET_SIZE)+tmp]
    test_labels += sign_labels[tmp+int(RATIO*DATASET_SIZE):tmp+DATASET_SIZE]
    


train_labels = np.array(train_labels)
test_labels = np.array(test_labels)    
       
class_names = ['10', '30', '50', '60', '90',
                '100']

# %% Preparing the NN model and feeding it the training data

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(100, 100, 3)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(len(class_names))
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

epochs=10
history = model.fit(train_images, train_labels, epochs)



# %%  Model Evaluation
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)


# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']

# loss = history.history['loss']
# val_loss = history.history['val_loss']

# epochs_range = range(epochs)

# plt.figure(figsize=(8, 8))
# plt.subplot(1, 2, 1)
# plt.plot(epochs_range, acc, label='Training Accuracy')
# plt.plot(epochs_range, val_acc, label='Validation Accuracy')
# plt.legend(loc='lower right')
# plt.title('Training and Validation Accuracy')

# plt.subplot(1, 2, 2)
# plt.plot(epochs_range, loss, label='Training Loss')
# plt.plot(epochs_range, val_loss, label='Validation Loss')
# plt.legend(loc='upper right')
# plt.title('Training and Validation Loss')
# plt.show()




