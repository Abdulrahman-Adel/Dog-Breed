# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 18:57:21 2020

@author: Abdelrahman
"""

import pandas as pd
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt


def extract_img(directory):
    X = []
    IDs = []
    for i in os.listdir(directory):
        img_arr = cv2.imread(os.path.join(directory,i),cv2.IMREAD_COLOR)
        new_img_arr = cv2.resize(img_arr, (400,400), interpolation = cv2.INTER_AREA)
        X.append(np.array(new_img_arr).reshape(400,400,3))
        ID , ex  = i.split(".")
        IDs.append(ID)
        
    X = np.array(X).astype(np.float32)
    X = X/255.0
    
    return X , IDs

def show_img(df):
    for i in range(9):
        plt.subplot(330 + 1 + i)
        plt.imshow(df[i])
    plt.show()    


X , _ = extract_img("train") 
X_test , name = extract_img("test")

labels = pd.read_csv("labels.csv")

y = pd.Series(labels.breed)
y = pd.get_dummies(y)

show_img(X)
show_img(X_test)

"""from keras.models import Model
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from keras.layers.merge import Concatenate
from keras.utils import plot_model
from keras.layers import Input


def inception_module(layer_in,f1,f2_in,f2_out,f3_in,f3_out,f4_out):
    
    conv1 = Conv2D(f1,(1,1),padding="same",activation="relu")(layer_in)
    
    conv3 = Conv2D(f2_in,(1,1),padding="same",activation="relu")(layer_in)
    conv3 = Conv2D(f2_out,(3,3),padding="same",activation="relu")(conv3)
    
    conv5 = Conv2D(f3_in,(1,1),padding="same",activation="relu")(layer_in)
    conv5 = Conv2D(f3_out,(5,5),padding="same",activation="relu")(conv5)
    
    pool = MaxPooling2D((3,3),strides=(1,1), padding='same')(layer_in)
    pool = Conv2D(f4_out,(1,1),padding="same",activation="relu")(pool)

    layer_out = Concatenate(axis=-1)([conv1,conv3,conv5,pool])
    
    return layer_out

input_img = Input(shape=(256,256,3))

layer1 = inception_module(input_img,64,96,128,16,32,32)

layer2 = inception_module(layer1,128,128,192,32,96,64)

flat = Flatten()(layer2)
dense1 = Dense(1024,activation="relu")(flat)
dense2 = Dense(120,activation="softmax")(dense1)


model = Model(inputs=input_img,outputs=dense2)

plot_model(model, show_shapes=True, to_file='dog_breed_model.png')

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X, y, epochs=10, batch_size=128, validation_split=0.25)


model.save_weights("dog_breed_model.h5")

Emojify_model = model.to_json()

with open("dog_breed_model.json", "w") as json_file:  
    json_file.write(Emojify_model)  """
    
from keras.applications import InceptionV3
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import SGD

opt = SGD(lr=0.001,momentum=0.9)

def model(dataAugmentation = False):
    
    model = Sequential()
    model.add(InceptionV3(include_top=False,weights="imagenet",classes=120,input_shape=(224,224,3)))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(1024,activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(120,activation="softmax"))

    model.compile(optimizer=opt, 
                  loss='categorical_crossentropy',
                  metrics=['accuracy']) 
    
    if not dataAugmentation:
        print('Not using data augmentation.')
        model.fit(X, y,
                  batch_size=32,
                  epochs=10,
                  validation_split=0.2)
    else:
        print('Using real-time data augmentation.')
    
        datagen = ImageDataGenerator(
            featurewise_center=False,  
            samplewise_center=False,  
            featurewise_std_normalization=False,  
            samplewise_std_normalization=False,  
            zca_whitening=False,  
            rotation_range=0,  
            width_shift_range=0.1,  
            height_shift_range=0.1,  
            horizontal_flip=True,  
            vertical_flip=False)  

        
        datagen.fit(X)

        
        model.fit_generator(datagen.flow(X, y, batch_size=32),
                            steps_per_epoch=X.shape[0] // 64,
                            epochs=10, verbose=1)
    
    return model     
        
history = model()       

y_pred = history.predict(X_test)


output = pd.DataFrame(y_pred)

output.columns = y.columns

output.insert(loc=0,column="id",value=name)

output.to_csv("sub0.csv",index=False)

history.save_weights("dog_breed_model.h5")

dog_breed_model = history.to_json()

with open("dog_breed_model.json", "w") as json_file:  
    json_file.write(dog_breed_model)  
      

 


        
        

   

        