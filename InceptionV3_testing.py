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

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix
from tensorflow.keras.applications import ResNet152V2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model

datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1/255, validation_split=0.3)

val = datagen.flow_from_directory('/mnt/Documents/Project⁄Thesis/InceptionV3/tomato/train', seed=123, subset='validation')

 # Test dataset for evaluation
datagen2 = keras.preprocessing.image.ImageDataGenerator(rescale=1/255)

test = datagen2.flow_from_directory('/mnt/Documents/Project⁄Thesis/InceptionV3/tomato/val')

model = tf.keras.models.load_model('/mnt/Documents/Project⁄Thesis/InceptionV3/InceptionV3.h5')
# model.summary()


classes = os.listdir('/mnt/Documents/Project⁄Thesis/InceptionV3/tomato/val/')

plt.figure(figsize=(18,28))

for i in enumerate(classes):
    pic = os.listdir('/mnt/Documents/Project⁄Thesis/InceptionV3/tomato/val/'+i[1])
    pic = pic[np.random.randint(len(pic)-1)]
    image = Image.open('/mnt/Documents/Project⁄Thesis/InceptionV3/tomato/val/'+i[1]+'/'+pic)
    image = np.asarray(image)
    
    pred = np.argmax(model.predict(image.reshape(-1,256,256,3)/255))
    for j in list(enumerate(list(test.class_indices.keys()))):
        if pred == j[0]:
            prediction =  j[1]
    
    plt.subplot(5,2,i[0]+1)
    plt.title('Actual: {0} / Predicted: {1}'.format(i[1], prediction))
    plt.imshow(image)
plt.show()

# con_matrix = tf.math.confusion_matrix(labels=test,predictions=pred)
# print(con_matrix)