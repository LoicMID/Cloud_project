import os
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout
from tensorflow.keras.models import Model

# création des directorys
data_clear_dir = "C:\Users\33619\Desktop\Master_MODE\M2\Machine learning\data\clouds\clear"
data_cloudy_dir = "C:\Users\33619\Desktop\Master_MODE\M2\Machine learning\data\clouds\cloudy"
data_haze_dir = "C:\Users\33619\Desktop\Master_MODE\M2\Machine learning\data\clouds\haze"
data_partly_cloudy_dir = "C:\Users\33619\Desktop\Master_MODE\M2\Machine learning\data\clouds\partly_cloudy"

# paramètres globaux
img_width, img_height = 256, 256
batch_size = 128
num_class = 4
epochs = 10

# Prétraitement des donénes et augmentation avec ImageDataGenerator
train_data = ImageDataGenerator(
    rescale = 1./255,
    validation_spilt = 0.25,
    horizontal_flip = True,
    vertical_flip = True,
    zoom_range = 0.15,
    width_shift_range = 0.15,
    height_shift_range = 0.15,
    rotation_range = 15
)