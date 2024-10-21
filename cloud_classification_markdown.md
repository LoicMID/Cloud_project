# Classifieur de couverture nuageuse

```{python}
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import sys
import zipfile

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img # pour preprocessing img et plot img validation
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout
from tensorflow.keras.models import Model # pour compilation model
from tensorflow.keras.optimizers import SGD

from sklearn.model_selection import cross_val_predict, cross_val_score # pour validation model
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc, f1_score, precision_score, recall_score # pour evaluation prédiction model
from sklearn.linear_model import SGDClassifier

import numpy as np
import matplotlib.pyplot as plt
import shutil # pour copier coller les fichiers de clouds vers organized_clouds
from sklearn.model_selection import train_test_split # pour separer jeu de donnees en train vs test
import random

```
