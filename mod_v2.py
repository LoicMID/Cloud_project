import os, signal
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import sys
import zipfile
from datetime import datetime
import random
import shutil # pour copier coller les fichiers de clouds vers organized_clouds

import tensorflow as tf
from tensorflow import keras
# import tensorflow_addons as tfa # nécessite version tensorflow antèrieur : pip install tensorflow==2.13.0

from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img # pour preprocessing img et plot img validation
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout
from tensorflow.keras.models import Model # pour compilation model
from tensorflow.keras.optimizers import SGD, RMSprop

from sklearn.model_selection import cross_val_predict, cross_val_score, train_test_split # pour validation model
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc, f1_score, precision_score, recall_score # pour evaluation prédiction model
from sklearn.linear_model import SGDClassifier

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ici classification multiclass. Cas multilabel si plusieurs classes peuvent être ensembles

# =============================================================================
# PARAMETRES ENVIRONNEMENT ET MODELE
# =============================================================================

# chemins d'accès
data_dir = "C:/Users/User/Desktop/MASTER/M2/MLB/PROJET/" # directory for images
organized_dir = "C:/Users/User/Desktop/MASTER/M2/MLB/PROJET/organized_clouds" # organized directory
augmented_dir = "C:/Users/User/Desktop/MASTER/M2/MLB/PROJET/augmented_clouds" # dataset organisé et augmenté
save_mod_dir = "C:/Users/User/Desktop/MASTER/M2/MLB/PROJET/models_trained/"
print(os.listdir(data_dir))

# Vérifier si le fichier ZIP existe et si le dossier de destination n'existe pas
if os.path.isfile(data_dir + "clouds.zip") and not os.path.isdir(data_dir + "clouds"):
    print('unzip')
    # Extraire le fichier ZIP
    with zipfile.ZipFile(data_dir + "clouds.zip", 'r') as zip_ref:
        zip_ref.extractall(data_dir + "clouds")  # Extraire dans un répertoire "clouds"
else:
    print("data directory already ready")

img_width, img_height = 256, 256
nb_class = 4 # clear / partly couldy / couldy / haze
class_names = ["clear","partly_cloudy","cloudy","haze"]
epoch_mod = 30 # nb de fois où les input sont pris en compte
batch_size_mod = 128 # nb d'échantillons traités ensembles. Après avoir traité tout les lots = une époch complète

### Creation seed pour reproductibilité de l'aleatoire : a activer si besoin
# random_seed = 42 #choix arbitraire de 42 
# np.random.seed(random_seed)  # renseignement de la seed au generateur de nombres aleatoires de NumPy
# random.seed(random_seed)  # renseignement de la seed au module aleatoire intege de base dans python    


# =============================================================================
# CREATION DATA IMG / TRAITEMENT IMG
# =============================================================================

# A VOIR /!\ données déséquilibrés dans chaque classes ?! Si oui utiliser courbe precision/rappel pour validation :
class_folders = os.listdir(data_dir + "clouds")

# Compter les fichiers dans chaque sous-dossier (classe)
for class_folder in class_folders:
    class_path = os.path.join(data_dir + "clouds", class_folder)
    if os.path.isdir(class_path):
        file_count = len([f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))])
        print(f"Classe {class_folder}: {file_count} images")

# classe fortement déséquilibré pour clear et un peu pour partly_cloudy
# Classe clear: 28432 images
# Classe cloudy: 2089 images
# Classe haze: 2697 images
# Classe partly_cloudy: 7261 images

###Separation des donnees Test vs Train vs Validation
if os.path.exists(organized_dir):
    user_response = input(f" Voulez-vous créer/remplacer '{organized_dir}'? (oui/non): ").strip().lower()
    if user_response == "oui":
        replace_organized_folder = True
    else:
        replace_organized_folder = False
else:
    replace_organized_folder = True

# si dossier organized_clouds à remplacer ou recréer:
if replace_organized_folder:
    if os.path.exists(organized_dir):
        shutil.rmtree(organized_dir)  # supprime le dossier organized_clouds si il existe
        print(f"Suppression du dossier {organized_dir}")
        
    # Création des dossiers train, test, et validation
    os.makedirs(os.path.join(organized_dir, 'train'), exist_ok=True)  # Dossier pour train
    os.makedirs(os.path.join(organized_dir, 'test'), exist_ok=True)   # Dossier pour test
    os.makedirs(os.path.join(organized_dir, 'valid'), exist_ok=True)  # Dossier pour validation
    
    # Ratios pour les splits
    train_ratio = 0.7  # 70% pour train
    valid_ratio = 0.15  # 15% pour validation
    test_ratio = 0.15  # 15% pour test
    
    # Boucle pour séparer les images dans chaque classe
    for class_name in class_names:
        print(class_name)
        class_path = os.path.join(data_dir + "clouds", class_name)  # Chemin vers le dossier de classe actuel
        images = os.listdir(class_path)  # Liste de toutes les images du dossier
        
        # Vérification de la présence d'images dans le dossier
        if len(images) == 0:
            print(f"Pas d'image trouvée dans le dossier de la classe {class_path}. Saut de cette classe.")
            continue  # Passe à la classe suivante s'il n'y a pas d'images
            
        # Séparation des images en trois ensembles : train, validation, et test
        train_images, valid_test_images = train_test_split(images, train_size=train_ratio) # random_state=random_seed
        valid_images, test_images = train_test_split(valid_test_images, test_size=test_ratio/(valid_ratio + test_ratio)) # random_state=random_seed
        
        # Création des sous-dossiers pour chaque classe dans train, validation et test
        os.makedirs(os.path.join(organized_dir, 'train', class_name), exist_ok=True)
        os.makedirs(os.path.join(organized_dir, 'valid', class_name), exist_ok=True)
        os.makedirs(os.path.join(organized_dir, 'test', class_name), exist_ok=True)

        # Copier les images dans les dossiers correspondants
        for img in train_images:
            shutil.copy2(os.path.join(class_path, img), os.path.join(organized_dir, 'train', class_name, img))
        for img in valid_images:
            shutil.copy2(os.path.join(class_path, img), os.path.join(organized_dir, 'valid', class_name, img))
        for img in test_images:
            shutil.copy2(os.path.join(class_path, img), os.path.join(organized_dir, 'test', class_name, img))

    print(f"Le dossier {organized_dir} a été remplacé.")


# Vérification du nombre de fichiers dans chaque dossier (train, validation, test)
for class_name in class_names:
    train_class_path = os.path.join(organized_dir, 'train', class_name)
    valid_class_path = os.path.join(organized_dir, 'valid', class_name)
    test_class_path = os.path.join(organized_dir, 'test', class_name)

    # Compte les fichiers dans chaque classe pour train
    num_train_files = len(os.listdir(train_class_path)) if os.path.exists(train_class_path) else 0
    print(f"TRAIN'{class_name}': {num_train_files}")

    # Compte les fichiers dans chaque classe pour validation
    num_valid_files = len(os.listdir(valid_class_path)) if os.path.exists(valid_class_path) else 0
    print(f"VALID '{class_name}': {num_valid_files}")
    
    # Compte les fichiers dans chaque classe pour test
    num_test_files = len(os.listdir(test_class_path)) if os.path.exists(test_class_path) else 0
    print(f"TEST'{class_name}': {num_test_files}")

print("Dataset organisé en dossiers train, validation et test.")

### Augmentation des images pour équilibrer nb d'indivs par classe et apporter variabilité dans les données d'entraînement et validation

## Paramètres d'augmentation

# nombre d'images augmentées générées à partir de l'image
augmentation_factor = 3
# Dictionnaire du nombre d'images par classe pour trouver le nb max d'images dans une classe
class_image_counts_train = {}
class_image_counts_valid = {}

## Générateur d'images augmentées
datagen_aug = ImageDataGenerator(
    rescale=1./255,  # transformation des valeurs RGB en float
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=0.15,
)

# Demander si il faut remplacer le dossier augmented_clouds
if os.path.exists(augmented_dir):
    user_response = input(f"'{augmented_dir}' existe déjà. Voulez-vous le remplacer? (oui/non): ").strip().lower()
    if user_response == "oui":
        replace_augmented_folder = True
    else:
        replace_augmented_folder = False
else:
    replace_augmented_folder = True

# Si dossier non-existant ou à remplacer:
if replace_augmented_folder:
    if os.path.exists(augmented_dir):
        shutil.rmtree(augmented_dir)  # supprime le dossier augmented_clouds s'il existe
        print(f"Suppression du dossier {augmented_dir}")
    
    # Création des dossiers train, valid et test pour augmentation
    for class_name in class_names:
        os.makedirs(os.path.join(augmented_dir, 'train', class_name), exist_ok=True)
        os.makedirs(os.path.join(augmented_dir, 'valid', class_name), exist_ok=True)
        os.makedirs(os.path.join(augmented_dir, 'test', class_name), exist_ok=True)
        
        ## DONNÉES TEST (non augmentées)
        class_test_path = os.path.join(organized_dir, 'test', class_name)  # Chemin du dossier test dans organized_clouds
        images_test = os.listdir(class_test_path)
        
        # Vérification de la présence d'images dans le dossier test
        if len(images_test) == 0:
            print(f"Pas d'image trouvée dans le dossier test de la classe {class_name}. Saut de cette classe.")
            continue
        
        # Copier les images de test non augmentées
        for img in images_test:
            shutil.copy2(os.path.join(class_test_path, img), os.path.join(augmented_dir, 'test', class_name, img))
        
    ## DONNÉES TRAIN (avec augmentation)

    # Comptage du nombre d'images dans chaque classe d'entraînement
    for class_name in class_names:
        class_train_path = os.path.join(organized_dir, 'train', class_name)
        images_train = os.listdir(class_train_path)
        class_image_counts_train[class_name] = len(images_train)

    # Recherche du nombre moyen d'images dans une classe d'entraînement
    mean_images_train = int(np.mean(list(class_image_counts_train.values())))
    print(f"Nombre moyen d'images dans une classe d'entraînement : {mean_images_train}")

    # Boucle pour générer le bon nombre d'images pour chaque classe d'entraînement
    for class_name in class_names:
        print("Train : ",class_name)
        class_train_path = os.path.join(organized_dir, 'train', class_name)
        images_train = os.listdir(class_train_path)
        
        # Sélectionner aléatoirement des images pour augmentation (pour équilibrer le nombre d'images)
        selected_images_train = random.choices(images_train, k=int(mean_images_train/augmentation_factor)) ########### PB NB IMG HERE <==============
        
        # Chemin de sauvegarde des images augmentées pour train
        augmented_train_class_path = os.path.join(augmented_dir, 'train', class_name)
        os.makedirs(augmented_train_class_path, exist_ok=True)
        
        # Augmenter et sauvegarder les images d'entraînement
        for img_name in selected_images_train:
            img_path = os.path.join(class_train_path, img_name)
            img = load_img(img_path, target_size=(img_width, img_height))
            x = img_to_array(img)
            x = np.expand_dims(x, axis=0)
            
            # Boucle pour augmenter et sauvegarder les images augmentées
            i = 0
            for batch in datagen_aug.flow(x, batch_size=1, save_to_dir=augmented_train_class_path, save_prefix='aug', save_format='jpeg'):
                i += 1
                if i >= augmentation_factor:
                    break

    ## DONNÉES VALID (avec augmentation)

    # Comptage du nombre d'images dans chaque classe de validation
    for class_name in class_names:
        class_valid_path = os.path.join(organized_dir, 'valid', class_name)
        images_valid = os.listdir(class_valid_path)
        class_image_counts_valid[class_name] = len(images_valid)

    # Recherche du nombre moyen d'images dans une classe de validation
    mean_images_valid = int(np.mean(list(class_image_counts_valid.values())))

    print(f"Nombre moyen d'images dans une classe de validation : {mean_images_valid}")

    # Boucle pour générer le bon nombre d'images pour chaque classe de validation
    for class_name in class_names:
        print("Valid : ",class_name)
        class_valid_path = os.path.join(organized_dir, 'valid', class_name)
        images_valid = os.listdir(class_valid_path)
        
        # Sélectionner aléatoirement des images pour augmentation
        selected_images_valid = random.choices(images_valid, k=int(mean_images_valid/augmentation_factor))
        
        # Chemin de sauvegarde des images augmentées pour validation
        augmented_valid_class_path = os.path.join(augmented_dir, 'valid', class_name)
        os.makedirs(augmented_valid_class_path, exist_ok=True)
        
        # Augmenter et sauvegarder les images de validation
        for img_name in selected_images_valid:
            img_path = os.path.join(class_valid_path, img_name)
            img = load_img(img_path, target_size=(img_width, img_height))
            x = img_to_array(img)
            x = np.expand_dims(x, axis=0)
            
            # Boucle pour augmenter et sauvegarder les images augmentées
            i = 0
            for batch in datagen_aug.flow(x, batch_size=1, save_to_dir=augmented_valid_class_path, save_prefix='aug', save_format='jpeg'):
                i += 1
                if i >= augmentation_factor:
                    break
    
    print(f"Le dossier {augmented_dir} a été créé/remplacé.")

# Vérification du nombre de fichiers dans chaque dossier (train, valid, test)
for class_name in class_names:
    train_class_path = os.path.join(augmented_dir, 'train', class_name)
    valid_class_path = os.path.join(augmented_dir, 'valid', class_name)
    test_class_path = os.path.join(augmented_dir, 'test', class_name)

    # Compte les fichiers dans chaque classe du dossier train
    num_train_files = len(os.listdir(train_class_path)) if os.path.exists(train_class_path) else 0
    print(f"TRAIN '{class_name}': {num_train_files}")

    # Compte les fichiers dans chaque classe du dossier valid
    num_valid_files = len(os.listdir(valid_class_path)) if os.path.exists(valid_class_path) else 0
    print(f"VALID '{class_name}': {num_valid_files}")

    # Compte les fichiers dans chaque classe du dossier test
    num_test_files = len(os.listdir(test_class_path)) if os.path.exists(test_class_path) else 0
    print(f"TEST '{class_name}': {num_test_files}")

###Creation de generateur d'images qui ne sert qu'à interpréter les images test (non augmentées) et train validation (déja augmentées)
generator = ImageDataGenerator(rescale=1./255)

# Appel du generateur pour creer les objets contenant les images traitées pour entrainer, valider et tester le modele 
# Creation data train
train_generator = generator.flow_from_directory(
    data_dir + "augmented_clouds/train", # augmented_clouds/train
    target_size = (img_width,img_height),
    color_mode = 'rgb',
    batch_size = batch_size_mod, 
    class_mode = "sparse", # fonction de perte => cross entropy
    shuffle = True, #  empeche le modèle d'apprendre sur ordre des échantillons
    )


# Creation data validation
valid_generator = generator.flow_from_directory(
    data_dir + "augmented_clouds/valid", # augmented_clouds/train
    target_size = (img_width,img_height),
    batch_size = batch_size_mod,
    class_mode = "sparse",
    shuffle = False
)

# Creation data test
test_generator = generator.flow_from_directory(
    data_dir + "augmented_clouds/test", # augmented_clouds/train
    target_size = (256, 256),
    batch_size = 1,
    class_mode = "sparse",
    shuffle = False
)

# nombre d'images pour chaque dataset 
print("train : ", len(train_generator.filenames))
print("valid : ", len(valid_generator.filenames))
print("test  : ", len(test_generator.filenames))

# nb de classes détéctés
print("nb classes : ", train_generator.class_indices)
print("nb classes : ", valid_generator.class_indices)
print("nb classes : ", test_generator.class_indices)

###### Afficher image ---------------------------------------------------------
nb_test = 1
type_generator = train_generator
# type_generator = valid_generator
# type_generator = test_generator

for i in range(nb_test):
    images, labels = next(type_generator)
    
    image = images[0]
    label = labels[0]
    
    class_index = np.argmax(label)
    
    plt.imshow(image)  
    plt.axis('off')
    plt.title(round(label))
    plt.show()
# clear : 0, 
# cloudy : 1
# haze : 2
# partly_cloudy : 3
###### ------------------------------------------------------------------------

# modif teste en tensorshape
images, labels = next(train_generator)
# train_generator = map(lambda x: (tf.convert_to_tensor(x[0], dtype=tf.float32), tf.convert_to_tensor(x[1], dtype=tf.float32)), train_generator)
print(len(images))
print(labels)
print(images.shape)
print(labels.shape)

images, labels = next(valid_generator)
# valid_generator = map(lambda x: (tf.convert_to_tensor(x[0], dtype=tf.float32), tf.convert_to_tensor(x[1], dtype=tf.float32)), valid_generator)
print(images.shape)
print(labels.shape)

images, labels = next(test_generator)
# test_generator = map(lambda x: (tf.convert_to_tensor(x[0], dtype=tf.float32), tf.convert_to_tensor(x[1], dtype=tf.float32)), test_generator)
print(images.shape)
print(labels.shape)

#### /!\ Dataset assez grand pour ne pas avoir à faire de la validation croisée !

# =============================================================================
# CREATION ARCHITECTURE MODEL
# =============================================================================

# Création d'un réseau de neurones vide 
model = keras.models.Sequential()

# Input 
model_input = Input(shape=(img_width, img_height,3)) # 3 car RVB

# 1ère couche - PARTIE 1 : convolution + activation ReLU + max-pooling + Dropout
model = Conv2D(16, (5,5), padding = "same")(model_input)
model = Activation("relu")(model)
model = MaxPooling2D(pool_size=(2,2))(model) # 2,2 taille par défault
#regularisation pr eviter le surapprentissage, permet d'éteindre des neurones à chaque époch. Valeur = % de neurones à éteindre => à mettre entre couche dense et parfois entre couches convolutionelles
model = Dropout(0.2)(model) 

# 1ère couche - PARTIE 2: convolution + activation ReLU + max-pooling + Dropout
model = Conv2D(16, (3,3), padding = "same")(model)
model = Activation("relu")(model)
model = MaxPooling2D(pool_size = (2,2))(model)
model = Dropout(0.2)(model) 

# 2ème couche : convolution + activation ReLU + max-pooling + Dropout
model = Conv2D(32, (3,3), padding = "same")(model)
model = Activation("relu")(model)
model = MaxPooling2D(pool_size = (2,2))(model)
model = Dropout(0.2)(model) 

# 3ème couche : applatissement + couche Dense + activation ReLU + Dropout
model = Flatten()(model)
model = Dense(16)(model)
model = Activation("relu")(model)
model = Dropout(0.2)(model) 

# Output
model = Dense(nb_class)(model)
model_output = Activation("softmax")(model)

# summary
model_final = Model(model_input, model_output)
model_final.summary()

# =============================================================================
# COMPILATION MODEL
# =============================================================================

# COMPILATION AVEC SGD
# model_final.compile(optimizer= SGD(learning_rate=0.001, momentum=0.9), # jouer sur les paramètres
#                     loss = "sparse_categorical_crossentropy",
#                     metrics = ["accuracy"]) # ajouter autre métriques ?

# momentum = 90 % => Indique que 90 % de la mise à jour précédente sera pris en compte lors de la mise à jour actuelle des poids

# COMPILATION AVEC RSMPROP (permet d'ajuste le taux d'apprentissage en continu)

model_final.compile(optimizer=RMSprop(learning_rate=0.001, rho=0.9),  # rho de RMSprop joue le même rôle que le momentum : 
                    loss="sparse_categorical_crossentropy",
                    metrics = [
                    'accuracy',
                    # tfa.metrics.F1Score(num_classes=num_classes, average='macro'),  # ou 'micro' selon votre besoin
                    # tfa.metrics.CohenKappa(),
                    # tfa.metrics.Precision(),
                    # tfa.metrics.Recall()
                    ]
                    )

# =============================================================================
# ENTRAINEMENT MODEL
# =============================================================================

## si bug tensorboard : 
# - kill processus : netstat -aon | findstr :6006 ##### puis : taskkill /PID <PID> /F # PID = identifiant 1er processus (à la toute fin)
# - relancer le code : F5

#### implémentation tensorboard -----------------------------------------------
root_logdir = os.path.join(os.curdir, "my_logs")
def get_run_logdir():
 import time
 run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
 return os.path.join(root_logdir, run_id)
run_logdir = get_run_logdir()
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
#### --------------------------------------------------------------------------

# Lancer TensorBoard automatiquement
%reload_ext tensorboard
%tensorboard --logdir ./my_logs --port=6006

# option de réduction d'apprentissage lorsqu'il n'y a plus de progrès
lr_scheduler = keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=4)

# changer les poids pour réequilibrer
# Nombre d'images par classe
for class_name in class_names:
    train_class_path = os.path.join(augmented_dir, 'train', class_name) # augmented_dir
    valid_class_path = os.path.join(augmented_dir, 'valid', class_name) # augmented_dir
    test_class_path = os.path.join(augmented_dir, 'test', class_name) # augmented_dir

    # Compte les fichiers dans chaque classe pour train
    num_train_files = len(os.listdir(train_class_path)) if os.path.exists(train_class_path) else 0
    print(f"TRAIN'{class_name}': {num_train_files}")

class_counts = {
    'clear': 5010,
    'cloudy': 5125,
    'haze': 5052,
    'partly_cloudy': 5053}

# Calcule le poids de chaque classe. Poids de la classe= total_images/ nombre de classes x nombre d’images dans la classe
total_images = sum(class_counts.values())
class_weights_mod = {i: round(float(total_images) / (len(class_counts) * count), 2) for i, count in enumerate(class_counts.values())}
print("Poids :", class_weights_mod)

# test avec load_weight avant + tester avec liste à la place dico
# entrainemtn du model ### voir si besoin de transformer image en tf.float32
history = model_final.fit(
              train_generator,
              # class_weight = class_weights_mod, ## <===================== ne marche pas si activé => utiliser sklearn.utils.compute_class_weight()
              epochs = epoch_mod,
              validation_data = valid_generator,
              callbacks=[tensorboard_cb,lr_scheduler] # tensorboard_cb pour appel tensorboard. On peut aussi ajouter lr_scheduler. # checkpoints ??
              )

#### --------------------------------------------------------------------------
# 1. Aller sur : http://localhost:6006
# 2. mettre en temps réel les calculs dans paramètres Tensorboard
# attention les courbe en opacité basse sont les valeurs réelle. En couleur c'est juste un smooth
#### --------------------------------------------------------------------------

# Sauvegarder le modèle entraîné
date_str = datetime.now().strftime("%d_%m")  # JJ_MM
model_final.save(save_mod_dir + f"cloud_classifier_model_{date_str}.h5")
# model = keras.models.load_model(save_mod_dir + "my_keras_model.h5") # pour load un model

sys.exit()

# =============================================================================
# VALIDATION MODEL
# =============================================================================

# VALIDATION sur des données de validation 

# Evolution accuracy => historique du modèle
# capacité à bien prédire qui évolue
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='valid accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.show()

# Evolution Fonction de perte (CE) => historique du modèle
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

# Evolution learning_rate => historique du modèle
plt.plot(history.history['learning_rate'], label='learning_rate')
plt.title('learning_rate')
plt.legend()
plt.show()

# Evolution Précision
# plt.plot(history.history['precision'], label='Precision')
# plt.plot(history.history['val_precision'], label='Validation Precision')
# plt.title('Training and Validation Precision')
# plt.legend()
# plt.show()

# # Evolution Rappel
# plt.plot(history.history['recall'], label='Recall')
# plt.plot(history.history['val_recall'], label='Validation Recall')
# plt.title('Training and Validation Recall')
# plt.legend()
# plt.show()

# # Evolution F1 Score
# plt.plot(history.history['f1_score'], label='F1 Score')
# plt.plot(history.history['val_f1_score'], label='Validation F1 Score')
# plt.title('Training and Validation F1 Score')
# plt.legend()
# plt.show()

# interprétation fct loss : 
#    - Train + valid diminue => good apprentissage 
#    - Train diminue + valid ré-augmente => Overfitting
#    - Train et valid restent haut => Underfitting (modèle trop simple)
#    - Fluctuation Train et Valid +> Lamda trop élevé ou mauvaises data valid


# =============================================================================
# PREDICTIONS ET EVALUATION
# =============================================================================

# clear : 0, 
# cloudy : 1
# haze : 2
# partly_cloudy : 3


###### PREDICTIONS : 
y_test_pred = model_final.predict(test_generator)
y_test_pred_classes = np.argmax(y_test_pred, axis=1)

###### EVALUATION sur données test
### 1. Accuracy
test_loss, test_accuracy = model_final.evaluate(test_generator)
print(f"Test loss : {test_loss}")
print(f"Test accuracy: {test_accuracy}")

### 2. Matrice de confusion  ----------------------------------------------------------------------------------------------------------------

### TOTALE : 
y_test_true = test_generator.classes  # Récupérer vrais labels

conf_matrix = confusion_matrix(y_test_true, y_test_pred_classes)
print(conf_matrix)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('Matrice de Confusion')
plt.xlabel('Prédictions')
plt.ylabel('Classes Réelles')
plt.show()

### POUR CHAQUE CLASSE : 
for i in range(nb_class):
    TP = conf_matrix[i, i]  # Vrais positifs
    FP = conf_matrix[:, i].sum() - TP  # Faux positifs
    FN = conf_matrix[i, :].sum() - TP  # Faux négatifs
    TN = conf_matrix.sum() - (FP + FN + TP)  # Vrais négatifs
    
    print(f"Classe {class_names[i]} :")
    print(f"  Vrais Positifs (TP): {TP}")
    print(f"  Faux Positifs (FP): {FP}")
    print(f"  Faux Négatifs (FN): {FN}")
    print(f"  Vrais Négatifs (TN): {TN}")
    print(" ")

for i in range(nb_class):
    # Créer une matrice de confusion pour chaque classe
    class_conf_matrix = np.zeros((2, 2), dtype=int)
    class_conf_matrix[0, 0] = conf_matrix[i, i]  # TP
    class_conf_matrix[0, 1] = conf_matrix[:, i].sum() - class_conf_matrix[0, 0]  # FP
    class_conf_matrix[1, 0] = conf_matrix[i, :].sum() - class_conf_matrix[0, 0]  # FN
    class_conf_matrix[1, 1] = conf_matrix.sum() - (class_conf_matrix[0, 1] + class_conf_matrix[1, 0] + class_conf_matrix[0, 0])  # TN

    # Visualisation
    plt.figure(figsize=(5, 4))
    sns.heatmap(class_conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Prédit Positif', 'Prédit Négatif'], 
                yticklabels=['Réel Positif', 'Réel Négatif'])
    plt.title(f'Matrice de Confusion pour la Classe : {class_names[i]}')
    plt.xlabel('Prédictions')
    plt.ylabel('Véritables Étiquettes')
    plt.show()

### 3. Courbe précision/rappel pour validation si données déséquilibrées -------------------------------------------------- 
precision_per_class = []
recall_per_class = []

for i in range(nb_class):
    TP = conf_matrix[i, i]  # Vrais positifs
    FP = conf_matrix[:, i].sum() - TP  # Faux positifs
    FN = conf_matrix[i, :].sum() - TP  # Faux négatifs
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0  # Précision
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0      # Rappel
    
    precision_per_class.append(precision)
    recall_per_class.append(recall)

    print(f"Classe {class_names[i]} :")
    print(f"  Précision : {precision:.2f}")
    print(f"  Rappel : {recall:.2f}")
    print(" ")

# Tracer les scores de précision et de rappel
x = np.arange(nb_class)  # l'emplacement des classes
width = 0.35  # largeur des barres

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, precision_per_class, width, label='Précision', color='blue')
bars2 = ax.bar(x + width/2, recall_per_class, width, label='Rappel', color='orange')

# Ajouter des étiquettes et un titre
ax.set_xlabel('Classes')
ax.set_ylabel('Scores')
ax.set_title('Précision et Rappel par Classe')
ax.set_xticks(x)
ax.set_xticklabels(class_names)
ax.legend()

# Afficher les scores au-dessus des barres
for bar in bars1:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom')

for bar in bars2:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom')

plt.show()

## 4. ROC curve et AUC ----------------------------------------------------------------------------------------------------------------------------- 
y_scores = model_final.predict(test_generator)[:, 1]  # Récupérer les scores de probabilité pour la classe positive

fpr, tpr, thresholds = roc_curve(y_test_true, y_scores)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='blue', label='Courbe ROC (AUC = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # ligne de base
plt.xlabel('Faux Positifs (FPR)')
plt.ylabel('Vrais Positifs (TPR)')
plt.title('Courbe ROC')
plt.legend()
plt.show()

### 5. F1 Score ---------------------------------------------------------------------------------------------------- 
f1 = f1_score(y_test_true, y_test_pred_classes, average='weighted')
print(f"F1 Score: {f1}")



os.kill(os.getpid(), signal.SIGKILL)

