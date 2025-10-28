import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#import libraries
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Flatten
from keras.utils import to_categorical
from keras.datasets import mnist
import matplotlib.pyplot as plt

#Load the dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.dtype)
print(x_train.shape)
print(x_train[0])
#plt.imshow(x_train[0],cmap='gray')
#print(f"Label: {y_train[0]}")
#plt.show()

#Normalize the data
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

#to categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
#print(f"new label: {y_train[0]}")

#Build the model
model = Sequential()
model.add(Flatten(input_shape=(28,28)))
model.add(Dense(128, activation='relu'))#128 neurons in hidden layer
model.add(Dense(10, activation='softmax'))#10 neurons in output layer for 10 classes

#Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy')

#Train the model
model.fit(x_train, y_train, epochs=10, batch_size=64)
