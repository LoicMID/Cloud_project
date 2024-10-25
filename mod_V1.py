import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import sys
import zipfile
from datetime import datetime

import tensorflow as tf
from tensorflow import keras
# import tensorflow_addons as tfa # nécessite version tensorflow antèrieur : pip install tensorflow==2.13.0

from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img # pour preprocessing img et plot img validation
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout
from tensorflow.keras.models import Model # pour compilation model
from tensorflow.keras.optimizers import SGD, RMSprop

from sklearn.model_selection import cross_val_predict, cross_val_score # pour validation model
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc, f1_score, precision_score, recall_score # pour evaluation prédiction model
from sklearn.linear_model import SGDClassifier


import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shutil # pour copier coller les fichiers de clouds vers organized_clouds
from sklearn.model_selection import train_test_split # pour separer jeu de donnees en train vs test
import random



### implémentation tensorboard -----------------------------------------------
# root_logdir = os.path.join(os.curdir, "my_logs")
# def get_run_logdir():
#   import time
#   run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
#   return os.path.join(root_logdir, run_id)
# run_logdir = get_run_logdir()
# tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
### --------------------------------------------------------------------------

#### --------------------------------------------------------------------------
# 1. éxécuter la commande dans la console : !tensorboard --logdir=./my_logs --port=6006
# 2. puis aller sur : http://localhost:6006
# (3. mettre en temps réel les calculs dans paramètres Tensorboard)
# 4. ourvrir une nouvelle console pour l'éxécution du code
#### --------------------------------------------------------------------------

#### EXECUTER A PARTIR DE LA APRES

# =============================================================================
# PARAMETRES ENVIRONNEMENT ET MODELE
# =============================================================================



### DEFINITION DES CHEMINS D'ACCES ET VERIFICATION DE LA PRESENCE DES DONNEES

#chemin pour acceder aux donnees originales
data_dir = "C:/Users/theop/Desktop/Malo/M2/cours/programmes_M2/" 
print(os.listdir(data_dir))

# chemin pour creer dataset organisé
organized_dir = "C:/Users/theop/Desktop/Malo/M2/cours/programmes_M2/organized_clouds"

#chemin pour creer dataset organisé et augmenté
augmented_dir = "C:/Users/theop/Desktop/Malo/M2/cours/programmes_M2/augmented_clouds"

#chemin pour sauvegarder le modele
save_mod_dir = "C:/Users/theop/Desktop/Malo/M2/cours/programmes_M2/"

# Vérifier si le fichier ZIP existe et si le dossier de destination n'existe pas
if os.path.isfile(data_dir + "clouds.zip") and not os.path.isdir(data_dir + "clouds"):
    print('unzip')
    # Extraire le fichier ZIP
    with zipfile.ZipFile(data_dir + "clouds.zip", 'r') as zip_ref:
        zip_ref.extractall(data_dir + "clouds")  # Extraire dans un répertoire "clouds"
else:
    print("data directory already ready")


### Creation seed pour reproductibilité de l'aleatoire
 
random_seed = 42 #choix arbitraire de 42 
np.random.seed(random_seed)  # renseignement de la seed au generateur de nombres aleatoires de NumPy
random.seed(random_seed)  # renseignement de la seed au module aleatoire intege de base dans python    


###Definition parametres

class_names = ["clear", "partly_cloudy", "cloudy", "haze"]
img_width, img_height = 256, 256
nb_class = 4 # clear / partly couldy / couldy / haze
epoch_mod = 5 # nb de fois où les input sont pris en compte
batch_size_mod = 128 # nb d'échantillons traités ensembles. Après avoir traité tous les lots = une époch complète


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

# demander si il faut remplacer le dossier organized_clouds
if os.path.exists(organized_dir):
    user_response = input(f"'{organized_dir}' existe déjà. Voulez-vous le remplacer? (oui/non): ").strip().lower()
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
        class_path = os.path.join(data_dir + "clouds", class_name)  # Chemin vers le dossier de classe actuel
        images = os.listdir(class_path)  # Liste de toutes les images du dossier
        
        # Vérification de la présence d'images dans le dossier
        if len(images) == 0:
            print(f"Pas d'image trouvée dans le dossier de la classe {class_path}. Saut de cette classe.")
            continue  # Passe à la classe suivante s'il n'y a pas d'images
            
        # Séparation des images en trois ensembles : train, validation, et test
        train_images, valid_test_images = train_test_split(images, train_size=train_ratio, random_state=random_seed)
        valid_images, test_images = train_test_split(valid_test_images, test_size=test_ratio/(valid_ratio + test_ratio), random_state=random_seed)
        
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
    print(f"Nombre d'images d'entraînement dans '{class_name}': {num_train_files}")

    # Compte les fichiers dans chaque classe pour validation
    num_valid_files = len(os.listdir(valid_class_path)) if os.path.exists(valid_class_path) else 0
    print(f"Nombre d'images de validation dans '{class_name}': {num_valid_files}")
    
    # Compte les fichiers dans chaque classe pour test
    num_test_files = len(os.listdir(test_class_path)) if os.path.exists(test_class_path) else 0
    print(f"Nombre d'images de test dans '{class_name}': {num_test_files}")

print("Dataset organisé en dossiers train, validation et test.")











# Nombre d'images augmentées générées à partir de l'image
augmentation_factor = 3
# Dictionnaire du nombre d'images par classe pour trouver le nb max d'images dans une classe
class_image_counts_train = {}
class_image_counts_valid = {}

# Fonction pour appliquer une ou plusieurs augmentations aléatoires basées sur OpenCV sans répétition
def augment_image(image):
    # Définir les types d'augmentations possibles
    def rotate_image(image):
        angle = random.choice([0, 90, 180, 270])
        if angle == 90:  
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            return cv2.rotate(image, cv2.ROTATE_180)
        elif angle == 270:
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return image

    def zoom_image(image):
        scale = random.uniform(0.8, 1.0)
        h, w = image.shape[:2]
        new_w = int(w * scale)
        new_h = int(h * scale)
        zoomed_image = cv2.resize(image, (new_w, new_h))
        
        # Centrer le zoom dans l'image originale
        # if scale < 1.0:
        #     pad_w = (w - new_w) // 2
        #     pad_h = (h - new_h) // 2
        #     zoomed_image = cv2.copyMakeBorder(zoomed_image, pad_h, h - new_h - pad_h, pad_w, w - new_w - pad_w, cv2.BORDER_REPLICATE)
        
        return zoomed_image

    def flip_image(image):
        flip_type = random.choice([1, 0])  # 1 pour flip horizontal, 0 pour flip vertical
        return cv2.flip(image, flip_type)

    # Liste des augmentations possibles
    available_augmentations = [rotate_image, zoom_image, flip_image]

    # Sélectionner aléatoirement de 1 à 3 augmentations sans répétition (tirage sans remise)
    selected_augmentations = random.sample(available_augmentations, k=random.randint(1, 3))

    # Appliquer les augmentations sélectionnées
    augmented_image = image
    for augmentation in selected_augmentations:
        augmented_image = augmentation(augmented_image)

    return augmented_image


# Demander si il faut remplacer le dossier augmented_clouds
if os.path.exists(augmented_dir):
    user_response = input(f"'{augmented_dir}' existe déjà. Voulez-vous le remplacer? (oui/non): ").strip().lower()
    replace_augmented_folder = user_response == "oui"
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

    # Recherche du nombre max d'images dans une classe d'entraînement
    max_images_train = max(list(class_image_counts_train.values()))
    print(f"Nombre maximum d'images dans une classe d'entraînement : {max_images_train}")

    # Boucle copier les images originales puis générer des images jusqu'à atteindre max_images_train pour chaque classe
    for class_name in class_names:
        print("train ", class_name)
        class_train_path = os.path.join(organized_dir, 'train', class_name)
        images_train = os.listdir(class_train_path)
        nb_img_originales = class_image_counts_train[class_name]
        
        # Copier les images de train non augmentées
        for img in images_train:
            shutil.copy2(os.path.join(class_train_path, img), os.path.join(augmented_dir, 'train', class_name, img))
        
        # Compléter avec images augmentées
        nb_img_manquantes = max_images_train - nb_img_originales
        print(nb_img_manquantes)
        # Sélectionner aléatoirement des images pour augmentation (pour équilibrer le nombre d'images)
        selected_images_train = random.choices(images_train, k=nb_img_manquantes)
        print(f"nombre d'images dans selected_images_train pour la classe {class_name}: ", len(selected_images_train))
        # Chemin de sauvegarde des images augmentées pour train
        augmented_train_class_path = os.path.join(augmented_dir, 'train', class_name)
        os.makedirs(augmented_train_class_path, exist_ok=True)
        
        # Augmenter et sauvegarder les images d'entraînement
        for img in selected_images_train:
            
            print(class_name, selected_images_train.index(img))
            
            img = cv2.imread(img_path)
    
            # Appliquer les augmentations OpenCV
            augmented_img = augment_image(img)
   
            # Sauvegarder les images augmentées
            save_path = os.path.join(augmented_train_class_path, f'aug_{img_name}')
            cv2.imwrite(save_path, augmented_img)
    
    ## DONNÉES VALID (avec augmentation)

    # Comptage du nombre d'images dans chaque classe de validation
    for class_name in class_names:
        class_valid_path = os.path.join(organized_dir, 'valid', class_name)
        images_valid = os.listdir(class_valid_path)
        class_image_counts_valid[class_name] = len(images_valid)

    # Recherche du nombre max d'images dans une classe de validation
    max_images_valid = max(list(class_image_counts_valid.values()))
    print(f"Nombre maximum d'images dans une classe de validation : {max_images_valid}")

    # Boucle pour générer le bon nombre d'images pour chaque classe de validation
    for class_name in class_names:
        print("valid ", class_name)
        class_valid_path = os.path.join(organized_dir, 'valid', class_name)
        images_valid = os.listdir(class_valid_path)
        
        # Copier les images de validation non augmentées
        for img in images_valid:
            shutil.copy2(os.path.join(class_valid_path, img), os.path.join(augmented_dir, 'valid', class_name, img))
        
        # Compléter avec images augmentées
        nb_img_manquantes = max_images_valid - len(images_valid)
        
        # Sélectionner aléatoirement des images pour augmentation
        selected_images_valid = random.choices(images_valid, k=nb_img_manquantes)
        
        # Chemin de sauvegarde des images augmentées pour validation
        augmented_valid_class_path = os.path.join(augmented_dir, 'valid', class_name)
        os.makedirs(augmented_valid_class_path, exist_ok=True)
        
        # Augmenter et sauvegarder les images de validation
        for img in selected_images_valid:
            
            print(class_name, selected_images_train.index(img))
            
            img = cv2.imread(img_path)
            
            # Appliquer les augmentations OpenCV
            augmented_img = augment_image(img)
            
            # Sauvegarder les images augmentées
            save_path = os.path.join(augmented_valid_class_path, f'aug_{img_name}')
            cv2.imwrite(save_path, augmented_img)

    print(f"Le dossier {augmented_dir} a été créé/remplacé.")

# Vérification du nombre de fichiers dans chaque dossier (train, valid, test)
for class_name in class_names:
    train_class_path = os.path.join(augmented_dir, 'train', class_name)
    valid_class_path = os.path.join(augmented_dir, 'valid', class_name)
    test_class_path = os.path.join(augmented_dir, 'test', class_name)

    # Compte les fichiers dans chaque classe du dossier train
    num_train_files = len(os.listdir(train_class_path)) if os.path.exists(train_class_path) else 0
    print(f"Nombre d'images d'entraînement dans '{class_name}': {num_train_files}")

    # Compte les fichiers dans chaque classe du dossier valid
    num_valid_files = len(os.listdir(valid_class_path)) if os.path.exists(valid_class_path) else 0
    print(f"Nombre d'images de validation dans '{class_name}': {num_valid_files}")

    # Compte les fichiers dans chaque classe du dossier test
    num_test_files = len(os.listdir(test_class_path)) if os.path.exists(test_class_path) else 0
    print(f"Nombre d'images de test dans '{class_name}': {num_test_files}")











###Creation de generateur d'images qui ne sert qu'à interpréter les images test (non augmentées) et train validation (déja augmentées)
generator = ImageDataGenerator(
    rescale=1./255,  # transformation des valeurs RGB en float
    )


# Appel du generateur pour creer les objets contenant les images traitées pour entrainer, valider et tester le modele 
# Creation data train
train_generator = generator.flow_from_directory(
    data_dir + "augmented_clouds/train",
    target_size = (img_width,img_height),
    color_mode = 'rgb',
    batch_size = batch_size_mod, 
    class_mode = "sparse", # fonction de perte => cross entropy
    shuffle = True, #  empeche le modèle d'apprendre sur ordre des échantillons
    )

# Creation data validation
valid_generator = generator.flow_from_directory(
    data_dir + "augmented_clouds/valid",
    target_size = (img_width,img_height),
    batch_size = batch_size_mod,
    class_mode = "sparse",
    shuffle = False
)

# Creation data test
test_generator = generator.flow_from_directory(
    data_dir + "augmented_clouds/test",
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


#### /!\ Dataset assez grand pour ne pas avoir à faire de la validation croisée !

###### Afficher image ---------------------------------------------------------
images, labels = next(train_generator)

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
# - kill processus : netstat -aon | findstr :6006 ##### puis : taskkill /PID <PID> /F
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
class_counts = {
    'clear': 28432,
    'cloudy': 2089,
    'haze': 2697,
    'partly_cloudy': 7261}

# Calcule le poids de chaque classe. Poids de la classe= total_images/ nombre de classes x nombre d’images dans la classe
total_images = sum(class_counts.values())
class_weights_mod = {i: total_images / (len(class_counts) * count) for i, count in enumerate(class_counts.values())}
print("Poids :", class_weights)

# entrainemtn du model
history = model_final.fit(
              train_generator,
              epochs = epoch_mod,
              validation_data = valid_generator,
              callbacks=[tensorboard_cb,lr_scheduler], # tensorboard_cb pour appel tensorboard. On peut aussi ajouter lr_scheduler
              class_weight = class_weights_mod
              )

#### --------------------------------------------------------------------------
# 1. Aller sur : http://localhost:6006
# 2. mettre en temps réel les calculs dans paramètres Tensorboard
# attention les courbe en opacité basse sont les valeurs réelle. En couleur c'est juste un smooth
#### --------------------------------------------------------------------------

# Sauvegarder le modèle entraîné
date_str = datetime.now().strftime("%d_%m")  # JJ_MM
model_final.save(save_mod_dir + f"cloud_classifier_model_{date_str}.h5")
# model = keras.models.load_model("my_keras_model.h5") # pour load un model

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



# PREDICTIONS : 
y_test_pred = model_final.predict(test_generator)
y_test_pred_classes = np.argmax(y_test_pred, axis=1)

###### EVALUATION sur données test
### 1. Accuracy
test_loss, test_accuracy = model_final.evaluate(test_generator)
print(f"Test loss : {test_loss}")
print(f"Test accuracy: {test_accuracy}")

### 2. Matrice de confusion  ----------------------------------------------------------------------------------------------------------------

### TOTALE : 
# rows represent actual classes, while columns represent predicted classes
y_test_true = test_generator.classes # Récupérer vrais labels

conf_matrix = confusion_matrix(y_test_true, y_test_pred_classes)
print(conf_matrix)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('Matrice de Confusion')
plt.xlabel('Prédictions')
plt.ylabel('classe réelles')
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


### 3. courbe precision/rappel pour validation si données déséquilibrés (c'est le cas ici) --------------------------------------------------
## score ----------------------------------------------------------------------
# A. Precision = proportion des vrais positifs
# A. Précision élevée :  lorsque le modèle prédit une instance comme positive, il a raison dans une grande majorité des cas.
# B. Rapell/sensibilité = proportion des vrais positifs parmi toutes les instances qui sont réellement positives (vrai positif + faux négatif)
# B. Rappel élevé : le modèle détecte la plupart des instances positives, même si cela signifie qu'il peut aussi classer certaines instances négatives comme positives.

# TOT


## SEUIL --------------------------------------
# Dans un contexte de classification binaire, le seuil détermine à partir de quelle probabilité on considère qu'une observation appartient à la classe positive.
# par défault, seuil = 0.5


## curve : à faire ------------------------------------------------------------
# [...]

### 4. ROC curve et AUC -----------------------------------------------------------------------------------------------------------------------------


# 5. f1 score = = moyenne harmonique de la précision et du rappel ----------------------------------------------------------------------------------------------------
# Le score F1 permet de mieux évaluer la capacité du modèle à identifier correctement la/les classe(s) minoritaire(s)