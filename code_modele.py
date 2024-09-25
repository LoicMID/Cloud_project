import os
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD

# création des directorys
#data_clear_dir = "C:\Users\33619\Desktop\Master_MODE\M2\Machine learning\data\clouds\clear"
#data_cloudy_dir = "C:\Users\33619\Desktop\Master_MODE\M2\Machine learning\data\clouds\cloudy"
#data_haze_dir = "C:\Users\33619\Desktop\Master_MODE\M2\Machine learning\data\clouds\haze"
#data_partly_cloudy_dir = "C:\Users\33619\Desktop\Master_MODE\M2\Machine learning\data\clouds\partly_cloudy"

data_dir = r"C:\Users\33619\Desktop\Master_MODE\M2\Machine learning\data\clouds"

# paramètres globaux
img_width, img_height = 256, 256
batch_size = 128
num_classes = 4
epochs = 10

# Prétraitement des donénes et augmentation avec ImageDataGenerator
train_data = ImageDataGenerator(
    rescale = 1./255,
    validation_split = 0.25,
    horizontal_flip = True,
    vertical_flip = True,
    zoom_range = 0.15,
    width_shift_range = 0.15,
    height_shift_range = 0.15,
    rotation_range = 15
)

# Générateur pour les données d'entrainement 
train_generator = train_data.flow_from_directory(
    data_dir,
    target_size = (img_width,img_height),
    color_mode = 'rgb',
    batch_size = batch_size,
    class_mode = "categorical",
    subset = "training",
    shuffle = True

)

# Générateur pour les données de validation
valid_generator = train_data.flow_from_directory(
    data_dir,
    target_size = (img_width,img_height),
    batch_size = batch_size,
    class_mode = "categorical",
    subset = 'validation',
    shuffle = False
)


#print("Classes détectées :")
#print(train_generator.class_indices) # Vérifier que les classes sont bien détectées

# Définition du modèle de réseau de neurones CNN
model_in = Input(shape=(img_width, img_height,3))

# Couche de convolution + activation ReLU + max-pooling
model = Conv2D(16, (5,5), padding = "same")(model_in)
model = Activation("relu")(model)
model = MaxPooling2D(pool_size=(2,2))(model)

# Deuxième couche de convolution
model = Conv2D(64, (3,3), padding = "same")(model)
model = Activation("relu")(model)
model = MaxPooling2D(pool_size = (2,2))(model)

# Troisième couche de convolution
model = Conv2D(64, (3,3), padding = "same")(model)
model = Activation("relu")(model)
model = MaxPooling2D(pool_size = (2,2))(model)

# Ajout d'une quatrième couche convolutionnelle
model = Conv2D(256, (3, 3), padding='same')(model)
model = Activation('relu')(model)
model = MaxPooling2D(pool_size=(2, 2))(model)

# Aplatissement de la sortie pour l'envoyer dans une couche dense
model = Flatten()(model)

# Couche dense entièrement connectée
model = Dense(128)(model)
model = Activation("relu")(model)

# Dropout pour éviter le surapprentissage
model = Dropout(0.5)(model)

# Sortie avec activation softmax pour la classification
model = Dense(num_classes)(model)
model_out = Activation("softmax")(model)

# Compilation du modèle
model_final = Model(model_in, model_out)
model_final.compile(optimizer= SGD(learning_rate=0.001, momentum=0.9),
                    loss = "categorical_crossentropy",
                    metrics = ["accuracy"])

# Sauvegarder le modèle entraîné
model_final.save("cloud_classifier_model.h5")

# Evaluer le modèle sur les données de validation
val_loss, val_accuracy = model_final.evaluate(valid_generator)
print(f"Validation loss : {val_loss}")
print(f"Validation accuracy: {val_accuracy}")

