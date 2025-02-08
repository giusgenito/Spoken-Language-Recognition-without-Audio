##----------------------------------------------------------------------------------------------------------------------

import pandas as pd
import numpy as np
import cv2
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
from tensorflow.keras.layers import ConvLSTM2D, MaxPooling2D, Flatten, Dense
from sklearn.preprocessing import OneHotEncoder

##----------------------------------------------------------------------------------------------------------------------

## Leggiamo i dati

Train_set = np.load("C:/Users/CASALAB/Desktop/Gruppo_20_Favb/SIMS_MOSEI/Preprocessing/Train_set.npy",allow_pickle=True)
Test_set = np.load("C:/Users/CASALAB/Desktop/Gruppo_20_Favb/SIMS_MOSEI/Preprocessing/Test_set.npy",allow_pickle=True)

x_train = []
y_train = []

for variabile, lingua in Train_set:
    x_train.append(variabile)
    y_train.append(lingua)

x_train = np.array(x_train)
y_train = np.array(y_train)
#np.save('C:/Users/CASALAB/Desktop/Gruppo_20_Favb/SIMS_MOSEI/Preprocessing/x_train.npy', x_train)
#np.save('C:/Users/CASALAB/Desktop/Gruppo_20_Favb/SIMS_MOSEI/Preprocessing/y_train.npy', y_train)

x_test = []
y_test = []

for variabile, lingua in Test_set:
    x_test.append(variabile)
    y_test.append(lingua)

x_test = np.array(x_test)
y_test = np.array(y_test)
#np.save('C:/Users/CASALAB/Desktop/Gruppo_20_Favb/SIMS_MOSEI/Preprocessing/x_test.npy', x_test)
#np.save('C:/Users/CASALAB/Desktop/Gruppo_20_Favb/SIMS_MOSEI/Preprocessing/y_test.npy', y_test)

callback = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=3)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv1D(filters=16, kernel_size=(5), activation='relu',input_shape=(100,50,50,3)))
#model.add(tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2)))
model.add(tf.keras.layers.Flatten())
#model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

# Compila il modello
model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=300, batch_size = 64, callbacks=[callback])
print("Valutazione sul test set: ")
model.evaluate(x_test,y_test)

y_proba = model.predict(x_test)
y_pred = y_proba.argmax(axis=-1)
print(y_pred)
print(y_test)

#Accuracy del train e del validation
plt.plot(history.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()

#Loss del train e del validation
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()

#model.save('C:/Users/CASALAB/Desktop/Gruppo_20_Favb/SIMS_MOSEI/Modelli/tentativo58')