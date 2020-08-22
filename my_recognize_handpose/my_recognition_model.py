import numpy as np
import pandas as pd # data processing, CSV input
import matplotlib.pyplot as plt
import os
import math
import tensorflow as tf

# get data, CSV input
train = pd.read_csv('data/real_dataset_train.csv')
test = pd.read_csv('data/real_dataset_test.csv')
#print(train.head(), train.shape)

# get labels data 
labels = train['label'].values
label = np.unique(np.array(labels))
#print(label) 

# delete label column from train data
train.drop('label', axis=1, inplace=True)
#print(train.head())

# Reshaping the images
images = train.values
#print(images, images.shape)
images = np.array([np.reshape(i, (28,28)) for i in images])
images = np.array([i.flatten() for i in images])
#print(images, images.shape)

# label binarizer
from sklearn.preprocessing import LabelBinarizer
label_binrizer = LabelBinarizer()
labels = label_binrizer.fit_transform(labels)
#print(labels)

# show images
#plt.imshow(images[26].reshape(28,28))
#plt.show()

# spliting the dataset into train and test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.3, random_state=101)



# using keras library for deep learning
import keras

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout

batch_size = 128
num_classes = 9
epochs = 150

x_train = x_train/255	# normalizing 0.0 ~ 1.0
x_test = x_test/255

x_train = x_train.reshape(x_train.shape[0],28,28,1)
x_test = x_test.reshape(x_test.shape[0],28,28,1)

#print(x_train.shape)
#plt.imshow(x_train[0].reshape(28,28))
#plt.show()



# Image Augmentation
from tensorflow.keras.preprocessing.image import ImageDataGenerator

image_generator = ImageDataGenerator(
	rotation_range=10, 
	zoom_range=0.1, 
	shear_range=0.5, 
	width_shift_range=0.1, 
	height_shift_range=0.1, 
	horizontal_flip=True,
	vertical_flip=False)

augment_size = 530

#x_augmented = image_generator.flow(np.tile(x_train[0].reshape(28*28), 100).reshape(-1, 28, 28, 1), np.zeros(augment_size), batch_size=augment_size, shuffle=False).next()[0]

# show images
#plt.figure(figsize=(10, 10))
#for c in range(100):
#	plt.subplot(10, 10, c+1)
#	plt.axis('off')
#	plt.imshow(x_augmented[c].reshape(28, 28), cmap='gray')
#plt.show()

randidx = np.random.randint(x_train.shape[0], size=augment_size)	# x_train : (7210, 28, 28, 1)
x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()
x_augmented = image_generator.flow(x_augmented, np.zeros(augment_size), batch_size=augment_size, shuffle=False).next()[0]

# original data + augmented data
x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))
#print(x_train.shape)



# CNN Model
model = Sequential()
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(64, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(64, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=2))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=batch_size)

# save model
model_name = 'model/my_hand_poses_recognition'+ '.h5'
model.save(model_name)

# show graph
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['train', 'test'])
plt.show()



# validate with the test data
test_labels = test['label']
test.drop('label', axis=1, inplace=True)

test_images = test.values
test_images = np.array([np.reshape(i, (28,28)) for i in test_images])
test_images = np.array([i.flatten() for i in test_images])

test_labels = label_binrizer.fit_transform(test_labels)

test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)
#print(test_images.shape)

# predecting with test images
y_pred = model.predict(test_images)

from sklearn.metrics import accuracy_score

print(accuracy_score(test_labels, y_pred.round()))