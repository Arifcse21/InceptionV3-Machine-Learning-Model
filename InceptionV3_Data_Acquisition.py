import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
import matplotlib.pyplot as plt
%matplotlib inline
from PIL import Image
from tensorflow import keras

datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1/255, validation_split=0.3)
# Training and validation dataset
train = datagen.flow_from_directory('/mnt/Documents/Project⁄Thesis/InceptionV3/tomato/train', seed=123, subset='training')
val = datagen.flow_from_directory('/mnt/Documents/Project⁄Thesis/InceptionV3/tomato/train', seed=123, subset='validation')

datagen2 = keras.preprocessing.image.ImageDataGenerator(rescale=1/255)


classes = os.listdir('/mnt/Documents/Project⁄Thesis/InceptionV3/tomato/train')

plt.figure(figsize=(25,10))

for i in enumerate(classes):
    pic = os.listdir('/mnt/Documents/Project⁄Thesis/InceptionV3/tomato/train/'+i[1])[0]
    image = Image.open('/mnt/Documents/Project⁄Thesis/InceptionV3/tomato/train/'+i[1]+'/'+pic)
    image = np.asarray(image)
    plt.subplot(2,5,i[0]+1)
    plt.title('{0} / Shape = {1}'.format(i[1], image.shape))
    plt.imshow(image)
plt.show()