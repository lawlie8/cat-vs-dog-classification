import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D

PATH = 'cats_and_dogs_filtered'#path to training data

train_data = os.path.join(PATH,'train')#train data extraction
validation_data = os.path.join(PATH,'validation')#validation data extraction

#train data
train_cats_dir = os.path.join(train_data,'cats')
train_dogs_dir = os.path.join(train_data,'dogs')

#validation data
validation_cats_dir = os.path.join(validation_data,'cats')
validation_dogs_dir = os.path.join(validation_data,'dogs')

#hyper parameters
batch_size =128
img_height = 150
img_width = 150

train_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
validation_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,directory=train_data,shuffle=True,target_size=(img_width,img_height),class_mode='binary')
val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,directory=validation_data,shuffle=True,target_size=(img_width,img_height),class_mode='binary')

check_point = 'check_point_dog_cat.cpkt'#check point
checkpoint_dir = os.path.dirname(check_point)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=check_point,save_weights_only=True,verbose=1)
num_cats_tr = len(os.listdir(train_cats_dir))
num_dogs_tr = len(os.listdir(train_dogs_dir))

num_cats_val = len(os.listdir(validation_cats_dir))
num_dogs_val = len(os.listdir(validation_dogs_dir))

total_train = num_cats_tr + num_dogs_tr
total_val = num_cats_val + num_dogs_val
#actual model
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

history = model.fit(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=15,
    validation_data=val_data_gen,
    validation_steps=total_val // batch_size,
    callbacks=[cp_callback]
)
