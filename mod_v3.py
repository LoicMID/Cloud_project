import os, signal
import pickle
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import sys
import zipfile
from datetime import datetime
import random
import shutil # pour copier coller les fichiers de clouds vers organized_clouds
import cv2
from imgaug import augmenters as iaa
from PIL import Image

import tensorflow as tf
from tensorflow import keras
# import tensorflow_addons as tfa # nécessite version tensorflow antèrieur : pip install tensorflow==2.13.0

from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img # pour preprocessing img et plot img validation
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout
from tensorflow.keras.models import Model # pour compilation model
from tensorflow.keras.optimizers import SGD, RMSprop, Adam

from sklearn.model_selection import cross_val_predict, cross_val_score, train_test_split # pour validation model
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score, roc_curve, auc, f1_score, precision_score, recall_score # pour evaluation prédiction model
from sklearn.linear_model import SGDClassifier

# from kerastuner import HyperParameters 

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

###### Hyperparamètres à ajuster : 
# Petites tailles de batch (16 - 64) : utilisées lorsque la généralisation est prioritaire, ou si les ressources mémoire sont limitées.
# Tailles de batch moyennes (128 - 256) : un bon compromis entre stabilité et efficacité, souvent utilisé en pratique.
# Grandes tailles de batch (512 et plus) : adaptées si le modèle est entraîné sur des données massives ou sur du matériel très performant, avec un apprentissage souvent plus stable.
batch_size_mod = 256 # nb d'échantillons traités ensembles. Après avoir traité tout les lots = une époch complète 
epoch_mod = 8 # nb de fois où les input sont pris en compte
dropout_mod = 0.1
learning_rate_mod = 0.001
taille_lot_augm_train = 6000
taille_lot_augm_valid = 1000

# paramètres architecture model à ajouter
nb_filtre_1, taille_filtre_1 = 16, (5,5) # pour couche convolution 1
nb_filtre_2, taille_filtre_2 = 16, (3,3) # pour couche convolution 2
nb_filtre_3, taille_filtre_3 = 32, (3,3) # pour couche convolution 3
taille_pool_size = (2,2) # ici chaque passage réduit la hauteur et la largeur de moitié
nb_neurones_dense = 16 # pour couche dense 4

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
print("\n")
for class_folder in class_folders:
    class_path = os.path.join(data_dir + "clouds", class_folder)
    if os.path.isdir(class_path):
        file_count = len([f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))])
        print(f"TOTAL : Classe {class_folder}: {file_count} images")
print("\n")
# classe fortement déséquilibré pour clear et un peu pour partly_cloudy
# Classe clear: 28432 images
# Classe cloudy: 2089 images
# Classe haze: 2697 images
# Classe partly_cloudy: 7261 images

###Separation des donnees Test vs Train vs Validation
if os.path.exists(organized_dir):
    user_response = input(f"Voulez-vous créer/remplacer '{organized_dir}'? (oui/non): ").strip().lower()
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


# # Vérification du nombre de fichiers dans chaque dossier (train, validation, test)
# for class_name in class_names:
#     train_class_path = os.path.join(organized_dir, 'train', class_name)
#     num_train_files = len(os.listdir(train_class_path)) if os.path.exists(train_class_path) else 0
#     print(f"TRAIN'{class_name}': {num_train_files}")
# print("\n")
# for class_name in class_names:
#     valid_class_path = os.path.join(organized_dir, 'valid', class_name)
#     num_valid_files = len(os.listdir(valid_class_path)) if os.path.exists(valid_class_path) else 0
#     print(f"VALID '{class_name}': {num_valid_files}")
# print("\n")    
# for class_name in class_names:
#     test_class_path = os.path.join(organized_dir, 'test', class_name)
#     num_test_files = len(os.listdir(test_class_path)) if os.path.exists(test_class_path) else 0
#     print(f"TEST'{class_name}': {num_test_files}")




if os.path.exists(augmented_dir):
    user_response = input(f"Voulez-vous créer/remplacer '{augmented_dir}'? (oui/non): ").strip().lower()
    replace_augmented_folder = user_response == "oui"
else:
    replace_augmented_folder = True

### Augmentation des images pour équilibrer nb d'indivs par classe et apporter variabilité dans les données d'entraînement et validation

def augmenter_images(class_path, target_count):
    augmenter = iaa.Sequential([
        iaa.Resize((img_height, img_width)), 
        iaa.Fliplr(0.5),  # flip vertical proba 0.5
        iaa.Flipud(0.5), #" flip horizontal proba 0.5
        # iaa.Affine(rotate=(-15, 15)), # rotation angulaire possible, mais ajoute barres noires
        iaa.AdditiveGaussianNoise(scale=(0, 0.05*255)), # ajout bruit
        iaa.Multiply((0.8, 1.2)), # ajout luminosité
        iaa.LinearContrast((0.8, 1.2)), # ajout contraste
    ])
    
    images = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))] # /!\ capter comment ça choisit les images
    num_images = len(images)
    
    print("augmentation de chaque classe pour respectivement train et valid")
    
    if num_images >= target_count:
        print(f"Classe '{os.path.basename(class_path)}' a déjà {num_images} images. Pas besoin d'augmentation.")
        # Supprime les images en excès si elles dépassent la limite cible
        excess_images = images[target_count:]  # Sélectionne les images au-delà de la limite
        for img_name in excess_images:
            img_path = os.path.join(class_path, img_name)
            os.remove(img_path)
        print(f"Classe '{os.path.basename(class_path)}' a maintenant {target_count} images après suppression.")
        return
    
    while len(images) < target_count:
        for img_name in images:
            if len(images) >= target_count:
                break
            img_path = os.path.join(class_path, img_name)
            image = Image.open(img_path).convert("RGB")  # Convertir en RGB pour compatibilité JPEG
            image = np.array(image)
            augmented_image = augmenter(image=image)
            augmented_img = Image.fromarray(augmented_image)
            
            # Nommer les images de manière concise
            augmented_img_name = f"aug_{len(images)}_{os.path.basename(img_name)}"
            augmented_img_path = os.path.join(class_path, augmented_img_name)
            augmented_img.save(augmented_img_path)
            images.append(augmented_img_name)
    
    print(f"Classe '{os.path.basename(class_path)}' a maintenant {len(images)} images.")

# Gestion du dossier d'augmentation
if replace_augmented_folder:
    if os.path.exists(augmented_dir):
        shutil.rmtree(augmented_dir)
        print(f"Suppression du dossier {augmented_dir}")
    
    # Création des dossiers train, valid et test pour augmentation
    for class_name in class_names:
        os.makedirs(os.path.join(augmented_dir, 'train', class_name), exist_ok=True)
        os.makedirs(os.path.join(augmented_dir, 'valid', class_name), exist_ok=True)
        os.makedirs(os.path.join(augmented_dir, 'test', class_name), exist_ok=True)

        # DONNÉES TEST (non augmentées)
        class_test_path = os.path.join(organized_dir, 'test', class_name)
        images_test = os.listdir(class_test_path)
        
        if len(images_test) == 0:
            print(f"Pas d'image trouvée dans le dossier test de la classe {class_name}. Saut de cette classe.")
            continue
        
        # Copier les images de test non augmentées
        for img in images_test:
            shutil.copy2(os.path.join(class_test_path, img), os.path.join(augmented_dir, 'test', class_name, img))
    
    # Appliquer aux dossiers d'entraînement et de validation
    for class_name in class_names:
        train_class_path = os.path.join(augmented_dir, 'train', class_name)
        organized_train_class_path = os.path.join(organized_dir, 'train', class_name)
        shutil.copytree(organized_train_class_path, train_class_path, dirs_exist_ok=True)
        augmenter_images(train_class_path, taille_lot_augm_train)
    
        valid_class_path = os.path.join(augmented_dir, 'valid', class_name)
        organized_valid_class_path = os.path.join(organized_dir, 'valid', class_name)
        shutil.copytree(organized_valid_class_path, valid_class_path, dirs_exist_ok=True)
        augmenter_images(valid_class_path, taille_lot_augm_valid)
        
print("\n")
# Vérification du nombre d'images dans chaque ensemble
for class_name in class_names:
    train_class_path = os.path.join(augmented_dir, 'train', class_name)
    num_train_files = len(os.listdir(train_class_path)) if os.path.exists(train_class_path) else 0
    print(f"TRAIN '{class_name}': {num_train_files}")
print("\n")
for class_name in class_names:
    valid_class_path = os.path.join(augmented_dir, 'valid', class_name)
    num_valid_files = len(os.listdir(valid_class_path)) if os.path.exists(valid_class_path) else 0
    print(f"VALID '{class_name}': {num_valid_files}")
print("\n")
for class_name in class_names:
    test_class_path = os.path.join(augmented_dir, 'test', class_name)
    num_test_files = len(os.listdir(test_class_path)) if os.path.exists(test_class_path) else 0
    print(f"TEST '{class_name}': {num_test_files}")
print("\n")


###Creation de generateur d'images qui ne sert qu'à interpréter les images test (non augmentées) et train validation (déja augmentées)
generator = ImageDataGenerator(rescale=1./255)

# Appel du generateur pour creer les objets contenant les images traitées pour entrainer, valider et tester le modele 
# Creation data train
train_generator = generator.flow_from_directory(
    data_dir + "augmented_clouds/train", # augmented_clouds/train # organized_clouds
    target_size = (img_width,img_height),
    color_mode = 'rgb',
    batch_size = batch_size_mod, 
    class_mode = "sparse", # fonction de perte => cross entropy
    shuffle = True, #  empeche le modèle d'apprendre sur ordre des échantillons
    )


# Creation data validation
valid_generator = generator.flow_from_directory(
    data_dir + "augmented_clouds/valid", # augmented_clouds/train # organized_clouds
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
# print("train : ", len(train_generator.filenames))
# print("valid : ", len(valid_generator.filenames))
# print("test  : ", len(test_generator.filenames))

# # nb de classes détéctés
# print("nb classes : ", train_generator.class_indices)
# print("nb classes : ", valid_generator.class_indices)
# print("nb classes : ", test_generator.class_indices)

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

# images, labels = next(train_generator)
# # train_generator = map(lambda x: (tf.convert_to_tensor(x[0], dtype=tf.float32), tf.convert_to_tensor(x[1], dtype=tf.float32)), train_generator)
# print(len(images))
# print(labels)
# print(images.shape)
# print(labels.shape)

# images, labels = next(valid_generator)
# # valid_generator = map(lambda x: (tf.convert_to_tensor(x[0], dtype=tf.float32), tf.convert_to_tensor(x[1], dtype=tf.float32)), valid_generator)
# print(images.shape)
# print(labels.shape)

# images, labels = next(test_generator)
# # test_generator = map(lambda x: (tf.convert_to_tensor(x[0], dtype=tf.float32), tf.convert_to_tensor(x[1], dtype=tf.float32)), test_generator)
# print(images.shape)
# print(labels.shape)

#### /!\ Dataset assez grand pour ne pas avoir à faire de la validation croisée !

# =============================================================================
# CREATION ARCHITECTURE MODEL
# =============================================================================

# Création d'un réseau de neurones vide 
model = keras.models.Sequential()

# Input 
model_input = Input(shape=(img_width, img_height,3)) # 3 car RVB

# 1ère couche : convolution + activation ReLU + max-pooling + Dropout
model = Conv2D(nb_filtre_1, taille_filtre_1, padding = "same")(model_input)
model = Activation("relu")(model)
model = MaxPooling2D(pool_size = taille_pool_size)(model) # 2,2 taille par défault = chaque passage réduit la hauteur et la largeur de moitié.
#regularisation pr eviter le surapprentissage, permet d'éteindre des neurones à chaque époch. Valeur = % de neurones à éteindre => à mettre entre couche dense et parfois entre couches convolutionelles
model = Dropout(dropout_mod)(model) 

#2ème couche : convolution + activation ReLU + max-pooling + Dropout
model = Conv2D(nb_filtre_2, taille_filtre_2, padding = "same")(model)
model = Activation("relu")(model)
model = MaxPooling2D(pool_size = taille_pool_size)(model)
model = Dropout(dropout_mod)(model) 

# 3ème couche : convolution + activation ReLU + max-pooling + Dropout
model = Conv2D(nb_filtre_3, taille_filtre_3, padding = "same")(model)
model = Activation("relu")(model)
model = MaxPooling2D(pool_size = taille_pool_size)(model)
model = Dropout(dropout_mod)(model) 

# 4ème couche : applatissement + couche Dense + activation ReLU + Dropout
model = Flatten()(model)
model = Dense(nb_neurones_dense)(model)
model = Activation("relu")(model)
model = Dropout(dropout_mod)(model) 

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

# COMPILATION AVEC RSMPROP OU ADAM

model_final.compile(optimizer = Adam(learning_rate = learning_rate_mod), 
                    loss = "sparse_categorical_crossentropy",
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
lr_scheduler = keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=3)

# ajustement des poids => non fonctionel avec générateur à la place de x et y en entrée dans la fonction fiy=t
# class_counts = {
#     'clear': 5010,
#     'cloudy': 5125,
#     'haze': 5052,
#     'partly_cloudy': 5053}
# # Calcule le poids de chaque classe. Poids de la classe= total_images/ nombre de classes x nombre d’images dans la classe
# total_images = sum(class_counts.values())
# class_weights_mod = {i: round(float(total_images) / (len(class_counts) * count), 2) for i, count in enumerate(class_counts.values())}
# print("Poids :", class_weights_mod)

print(f"\nEntrainement du model sur {taille_lot_augm_train} images par classes sur {epoch_mod} epochs composé de batch de {batch_size_mod} images \ntaux d'apprentissage initial = {learning_rate_mod}\nDropout = {dropout_mod}\n")


# entrainemtn du model 
history = model_final.fit(
              train_generator,
              # class_weight = class_weights_mod, # <===================== ne marche pas si activé 
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
date_str = datetime.now().strftime("%d_%m_%Hh%M")  # JJ_MM
model_final.save(save_mod_dir + f"cloud_classifier_model_{date_str}_avec_{epoch_mod}_epochs.h5")
# model_final = keras.models.load_model(save_mod_dir + "cloud_classifier_model_25_10_avec_10_epochs.h5") # pour load un model

# Sauvegarde de l'historique
with open('history.pkl', 'wb') as f:
    pickle.dump(history.history, f)

# # Rechargement de l'historique
# with open('history.pkl', 'rb') as f:
#     history = pickle.load(f)
    
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

little_test = True
if little_test:
    # possibilité de réaliser les test sur un ensemble + petit pour équilibrer les classe lors de l'interprétation
    test_dir = "C:/Users/User/Desktop/MASTER/M2/MLB/PROJET/organized_clouds/test"  # Dossier d'origine
    little_test_dir = "C:/Users/User/Desktop/MASTER/M2/MLB/PROJET/little_test"  # Nouveau dossier
    max_images = 314 # pour cloudy = 314
    
    if os.path.exists(little_test_dir):
        user_response = input(f"Le dossier '{little_test_dir}' existe déjà. Supprimer le ! ? (oui/non): ").strip().lower()
        if user_response == "oui":
            shutil.rmtree(little_test_dir)
            print(f"Le dossier '{little_test_dir}' a été supprimé.")
        else:
            print("Opération annulée.") # Arrête le script si l'utilisateur choisit de ne pas remplacer
    else:
        print(f"Création du dossier '{little_test_dir}'.")
    
    # Création du dossier `little_test` et sous-dossiers pour chaque classe
    os.makedirs(little_test_dir, exist_ok=True)
    
    # Copie des images dans `little_test` en respectant la limite
    for class_name in class_names:
        class_test_path = os.path.join(test_dir, class_name)
        class_little_test_path = os.path.join(little_test_dir, class_name)
        os.makedirs(class_little_test_path, exist_ok=True)
        
        # Récupérer toutes les images de la classe et en choisir aléatoirement un sous-ensemble
        images = [img for img in os.listdir(class_test_path) if os.path.isfile(os.path.join(class_test_path, img))]
        selected_images = random.sample(images, min(len(images), max_images))
        
        # Copier les images sélectionnées
        for img in selected_images:
            src_path = os.path.join(class_test_path, img)
            dst_path = os.path.join(class_little_test_path, img)
            shutil.copy2(src_path, dst_path)
    
    little_test_generator = generator.flow_from_directory(
        little_test_dir,
        target_size = (256, 256),
        batch_size = 1,
        class_mode = "sparse",
        shuffle = False
    )

if little_test:
    test_generator = little_test_generator
    
    for class_name in class_names:
        train_class_path = os.path.join(little_test_dir, class_name)
        num_train_files = len(os.listdir(train_class_path))
        print(f"TEST '{class_name}': {num_train_files}")

else:
    test_generator = generator.flow_from_directory(
        data_dir + "augmented_clouds/test", # augmented_clouds/train
        target_size = (256, 256),
        batch_size = 1,
        class_mode = "sparse",
        shuffle = False
    )



###### PREDICTIONS : 
y_test_pred_prob = model_final.predict(test_generator)
y_test_pred_classes = np.argmax(y_test_pred_prob, axis=1)

###### EVALUATION sur données test
### 1. Accuracy
test_loss, test_accuracy = model_final.evaluate(test_generator)
print(f"Test loss : {test_loss}")
print(f"Test accuracy: {test_accuracy}")

### 2. Matrice de confusion  ----------------------------------------------------------------------------------------------------------------

### TOTALE : 
y_test_true = test_generator.classes  # Récupérer vrais labels

conf_matrix = confusion_matrix(y_test_true, y_test_pred_classes)
print(conf_matrix) #### attention biaisé par le grand nombre d'images dans clear ! 

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Greys', 
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
    
    print("\n")
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
    sns.heatmap(class_conf_matrix, annot=True, fmt='d', cmap='Greys', 
                xticklabels=['Prédit Positif', 'Prédit Négatif'], 
                yticklabels=['Réel Positif', 'Réel Négatif'])
    plt.title(f'Matrice de Confusion pour la Classe : {class_names[i]}')
    plt.xlabel('Prédictions')
    plt.ylabel('Véritables Étiquettes')
    plt.show()

### 3. Précision/rappel pour validation si données déséquilibrées -------------------------------------------------- 
# Précision : 
# Haute précision : le modèle fait peu d'erreurs lorsqu'il prédit la classe positive. Cela signifie que les échantillons classés comme positifs sont généralement corrects.
# Basse précision : le modèle fait de nombreuses erreurs en classant comme positifs des échantillons qui ne le sont pas, ce qui signifie qu'il produit beaucoup de faux positifs.

# Sensibilité/Rappel
# Haut rappel : le modèle identifie bien les cas positifs, minimisant les faux négatifs.
# Bas rappel : le modèle manque de nombreux exemples de la classe positive, produisant beaucoup de faux négatifs.

#### =============== SCORES ===============
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

x = np.arange(nb_class)  # l'emplacement des classes
width = 0.35  # largeur des barres

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, precision_per_class, width, label='Précision', color='grey')
bars2 = ax.bar(x + width/2, recall_per_class, width, label='Rappel', color='black')

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

#### =============== CURVES ===============

plt.figure(figsize=(10, 8))

for class_index in range(nb_class):
    precision, recall, thresholds = precision_recall_curve(y_test_true == class_index, y_test_pred_prob[:, class_index])
    average_precision = average_precision_score(y_test_true == class_index, y_test_pred_prob[:, class_index])
    
    plt.plot(recall, precision, marker='.', label=f'Classe {class_names[class_index]} (AP = {average_precision:.2f})')

plt.xlabel('Rappel')
plt.ylabel('Précision')
plt.title('Courbe Précision-Rappel pour toutes les classes')
plt.legend()
plt.grid()
plt.show()


## 4. ROC curve et AUC ----------------------------------------------------------------------------------------------------------------------------- 
# curve  = taux de vrais positif (sensibilité/rappel) en fonction du taux de faux positifs
#  La courbe est tracée en changeant le seuil de classification et en calculant le TPR et le FPR pour chaque seuil. 
# Plus la courbe est proche du coin supérieur gauche du graphique (0,1), meilleure est la performance du modèle

plt.figure(figsize=(10, 6))

for i in range(nb_class):
    fpr, tpr, thresholds = roc_curve(y_test_true, y_test_pred_prob[:, i], pos_label=i)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'ROC curve pour {class_names[i]} (AUC = {roc_auc:.2f})')

# Ajouter des éléments au graphique
plt.plot([0, 1], [0, 1], 'k--')  # ligne diagonale
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taux de faux positifs')
plt.ylabel('Taux de vrais positifs')
plt.title('Courbes ROC par classe')
plt.legend(loc="lower right")
plt.show()

# AUC est un indicateur de performance. Elle varie entre 0 et 1 :
# AUC = 0.5 : Le modèle n'a pas de pouvoir discriminant, équivalent à un tirage aléatoire.
# AUC < 0.5 : Le modèle est moins performant qu'un modèle aléatoire.
# AUC = 1 : Le modèle fait une classification parfaite.

### 5. F1 Score ---------------------------------------------------------------------------------------------------- 
# Calcul du F1 score
f1_per_class = f1_score(y_test_true, y_test_pred_classes, average=None)  # F1 score pour chaque classe
f1_weighted = f1_score(y_test_true, y_test_pred_classes, average='weighted')  # F1 score global

# Affichage des résultats
for i, class_name in enumerate(class_names):
    print(f"F1 Score pour {class_name}: {f1_per_class[i]:.2f}")

print(f"F1 Score global (weighted): {f1_weighted:.2f}")

plt.figure(figsize=(10, 6))
bars = plt.bar(class_names, f1_per_class, color='skyblue', label='F1 Score par Classe')
plt.axhline(y=f1_weighted, color='orange', linestyle='--', label=f"F1 Score Global (weighted) : {f1_weighted:.2f}")

# Ajouter des annotations
for bar, score in zip(bars, f1_per_class):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 0.05, f"{score:.2f}", ha='center', va='bottom', color='black')

plt.xlabel("Classes")
plt.ylabel("F1 Score")
plt.title("Histogramme des F1 Scores par Classe et F1 Score Global")
plt.legend()
plt.ylim(0, 1)  # Score F1 est entre 0 et 1
plt.show()


os.kill(os.getpid(), signal.SIGTERM)
