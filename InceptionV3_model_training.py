import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
import matplotlib.pyplot as plt
# %matplotlib inline


import os
os.chdir('/mnt/Documents/Project⁄Thesis/InceptionV3/tomato')
os.listdir()
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from PIL import Image

from tensorflow import keras
from tensorflow.keras.applications import ResNet152V2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Flatten, Activation, Dropout, BatchNormalization
from tensorflow.keras.models import Model

datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1/255, validation_split=0.3)
# Training and validation dataset
train = datagen.flow_from_directory('/mnt/Documents/Project⁄Thesis/InceptionV3/tomato/train', seed=123, subset='training')
val = datagen.flow_from_directory('/mnt/Documents/Project⁄Thesis/InceptionV3/tomato/train', seed=123, subset='validation')

# Test dataset for evaluation
datagen2 = keras.preprocessing.image.ImageDataGenerator(rescale=1/255)

test = datagen2.flow_from_directory('/mnt/Documents/Project⁄Thesis/InceptionV3/tomato/val')

classes = os.listdir('/mnt/Documents/Project⁄Thesis/InceptionV3/tomato/train')

from tensorflow.keras.applications import InceptionV3
from keras.callbacks import ModelCheckpoint
filepath = "InceptionV3.h5"
checkpoint = ModelCheckpoint(filepath, 
                             monitor='val_acc', 
                             verbose=1, 
                             save_best_only=True, 
                             mode='max')
callbacks_list = [checkpoint]


def get_model_incept():
    
  base_model = InceptionV3(input_shape=(256,256,3), include_top=False)
  for layers in base_model.layers:
    layers.trainable = False
       
    
  x = base_model.output
  x = GlobalAveragePooling2D()(x)
  x = Dense(1000, activation='relu')(x)
  pred = Dense(10, activation='softmax')(x)
    
  model = Model(inputs=base_model.input, outputs=pred)
    
  return model




model = get_model_incept()
# model.summary()

model.save('InceptionV3.h5')

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics='accuracy')
history = model.fit(train, batch_size=80, epochs=5, validation_data=val,callbacks=[callbacks_list,tf.keras.callbacks.CSVLogger('history_inceptionv3.csv')])

plt.plot(model.history.history['accuracy'])
plt.plot(model.history.history['val_accuracy'])
plt.title("Model Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(['Train','Test'],loc='best')
plt.savefig('inceptionv3_model_accuracy_DataGenerator.jpeg')
plt.savefig('inceptionv3_model_accuracy_DataGenerator.svg')
plt.show()

plt.plot(model.history.history['loss'])
plt.plot(model.history.history['val_loss'])
plt.title("Model Loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(['Train','Test'],loc='best')
plt.savefig('inceptionv3_model_loss_DataGenerator.jpeg')
plt.savefig('inceptionv3_model_loss_DataGenerator.svg')
plt.show()

#model = tf.keras.models.load_model('/mnt/Documents/Project⁄Thesis/InceptionV3/InceptionV3.h5')


model.evaluate(test)
