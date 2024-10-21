import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
import h5py
import os
from keras import Sequential
from keras.layers import Dense,MaxPooling2D,Conv2D,Flatten,Dropout,Activation
from keras.layers import BatchNormalization
from sklearn import svm
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from oauth2client.client import GoogleCredentials

import csv
#from keras.optimizers import RMSprop
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from keras import applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model 
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k 
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping

# from google.colab import files, drive, auth
# drive.mount('/content/gdrive/')
# PATH = "/content/gdrive/My Drive/"
# os.listdir(PATH)
# if os.path.isfile(PATH+"clouds") and not os.path.isdir("clouds"):
#     print('unzip')
#     !unzip -q drive/My\ Drive/clouds -d clouds #on dezip dans un repertoire
# else:
#     print("data directory already ready")

# import os
# os.listdir("/content/gdrive/My Drive/")



import os

# Path to the folder containing images
image_directory = 'C:/Users/theop/Desktop/Malo/M2/cours/programmes_M2/clouds'

# Get a list of all files in the directory
files = os.listdir(image_directory)

# Filter the list to include only image files
image_files = [file for file in files if file.lower().endswith('.jpg')]

# Count the number of image files
num_images = len(image_files)
print(f"Number of images: {num_images}")




train_data = ImageDataGenerator(
#    horizontal_flip =True,
#    vertical_flip = True,
    rescale = 1./255, # Permet de déterminer la proportion de train/validation
    #horizontal_flip = True,
    #zoom_range = 0.15,
    #width_shift_range = 0.15,
    #height_shift_range=0.15,
    #rotation_range=15,
    validation_split = 0.25)

# Permet d'acceder aux données d'un zip ultra rapidement
train_generator = train_data.flow_from_directory(
        'C:/Users/theop/Desktop/Malo/M2/cours/programmes_M2/clouds', # Lien des images dans le fichier zip (à adapter)
        target_size=(256, 256),
        color_mode='rgb',
        batch_size=128,
        shuffle = True,
        class_mode="categorical")



# 20% des données de Train (target + other)
valid_generator = train_data.flow_from_directory(
        'C:/Users/theop/Desktop/Malo/M2/cours/programmes_M2/clouds', 
        target_size=(256, 256),
        batch_size=128,
        class_mode='categorical')


test_data = ImageDataGenerator(rescale=1./255)

test_generator = test_data.flow_from_directory(
    directory='C:/Users/theop/Desktop/Malo/M2/cours/programmes_M2/clouds',
    target_size=(256, 256) ,
    batch_size=1,
    #class_mode="binary",
    shuffle=False
)

print(len(test_generator.filenames))




num_classes=4
img_width = 256
img_height = 256

#model = Sequential()  # Création d'un réseau de neurones vide 

# Ajout de la première couche, suivie d'une couche ReLU
model_in=Input(shape=[img_width, img_height, 3])
#le model va prendre des matrices de taille 256*256 et on va le transformer en vecteurs de taille 128
model=Conv2D(16,(5,5))(model_in)
model=Activation('relu')(model)
model=MaxPooling2D()(model)

model=Conv2D(16,(3,3))(model_in)
model=Activation('relu')(model)
model=MaxPooling2D()(model)

#model.add(Dropout(0.2)) #regularisation pr eviter le surapprentissage

# Ajout de la deuxieme couche, suivie d'une couche ReLU
#model.add(Dense(64, activation='relu'))
model=Conv2D(32,(3,3))(model)
model=Activation('relu')(model)
model=MaxPooling2D()(model)
#model.add(Dropout(0.2))

model=Flatten()(model)

#model=Dense(16,activation='relu')(model)
#model=Dropout(0.2)(model)

#derniere couche: softmax
#model2=Dense(1, activation=tf.nn.softmax)(model2)
model=Dense(num_classes, activation='softmax')(model)
model_final = Model(model_in,model)



model_final.summary()
#model.add(Dense(num_classes, activation='sigmoid'))


epochs=3
step_size_train=train_generator.n//train_generator.batch_size
nb_validation_samples = valid_generator.n//valid_generator.batch_size//5
print(step_size_train)
print(nb_validation_samples)

#Compilation 
model_final.compile(loss='categorical_crossentropy',
              optimizer="RMSprop", # to be checked 
              metrics=['accuracy'])

print(train_generator.class_indices)  # Doit afficher les classes dans vos données d'entraînement
print(valid_generator.class_indices)  # Doit afficher les classes dans vos données de validation

model_final.fit(
              train_generator,
              #samples_per_epoch = nb_train_samples,
              steps_per_epoch=step_size_train,
              epochs = epochs,
              validation_data = valid_generator,
              validation_steps=nb_validation_samples
              #nb_val_samples = 10
              #callbacks = [checkpoint]
              )

#score = model_final.evaluate(pre_data_train, pre_data_train_name)
#print('Test loss:', score[0])
#print('Test accuracy:', score[1])


#Affichage des prédictions

filenames = test_generator.filenames
nb_samples = len(filenames)
print(nb_samples)

predict = model_final.predict(test_generator,steps = nb_samples)
print(predict)

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

# last_img = 17855
last_img  = 10
f = open('myfile.csv', 'w') 
f.write("name;prediction\n")
for i in range(last_img): 
  filename = str(i)+".jpg"
  img = load_img('C:/Users/theop/Desktop/Malo/M2/cours/programmes_M2/clouds'+filename)  # this is a PIL image
  x = img_to_array(img) 
  x = x/255
  x = x.reshape((1,) + x.shape) 
  predict = model_final.predict(x)
  #print(filename+";"+str(int(predict>.5)))
  f.write(filename+";"+str(int(predict>.5))+"\n")
f.close()
files.download('myfile.csv')

accuracy = history.history['acc']
loss = history.history['loss']
val_accuracy = history.history['val_acc']
val_loss = history.history['val_loss']

print(f'Training Accuracy: {np.max(accuracy)}')
print(f'Training Loss: {np.min(loss)}')
print(f'Validation Accuracy: {np.max(val_accuracy)}')
print(f'Validation Loss: {np.min(val_loss)}')

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
