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
#print(x_train[0])
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
model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
#Train the model
history=model.fit(x_train, y_train, epochs=10, batch_size=64,validation_split=0.2)

#test the model
loss,accuracy=model.evaluate(x_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print(f"Test Loss: {loss:.4f}")
print(history.history)

#Visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Subplot 1: Accuracy
ax1.plot(history.history['accuracy'], label='Training Accuracy', marker='o')
ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='o')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Accuracy')
ax1.set_title('Model Accuracy')
ax1.legend()
ax1.grid(True)

# Subplot 2: Loss
ax2.plot(history.history['loss'], label='Training Loss', marker='o')
ax2.plot(history.history['val_loss'], label='Validation Loss', marker='o')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Loss')
ax2.set_title('Model Loss')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()