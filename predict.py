import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from skimage import io
from scipy.misc import imresize
import sys
img_height = 150
img_width = 150

def get_img(data_path):
    # Getting image array from path:
    img_size = 150
    img = io.imread(data_path)
    img = imresize(img, (img_size, img_size, 3))
    return img
img_dir = sys.argv[1]
img = get_img(img_dir)
X = np.zeros((1, 150, 150, 3), dtype='float64')
X[0] = img
def create_model():
    model = tf.keras.Sequential([
                                        Conv2D(16,3,input_shape=(img_height,img_width,3),activation='relu',padding='same'),
                                        MaxPooling2D(),

                                        Conv2D(32,3,padding='same',activation='relu'),
                                        MaxPooling2D(),

                                        Conv2D(64,3,padding='same',activation='relu'),
                                        MaxPooling2D(),

                                        Flatten(),

                                        Dense(512,activation='relu'),

                                        Dense(1)
                                ])
    model.compile(optimizer='adam',loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),metrics=['accuracy'])
    return model

check_point ='check_point_dog_cat.cpkt'
model = create_model()
model.load_weights(check_point)
predictions = model.predict(X)

if predictions > 0:
    print('Its A Dog')
else:
    print('Its A Cat')
img=mpimg.imread(img_dir)
imgplot = plt.imshow(img)
plt.show()
