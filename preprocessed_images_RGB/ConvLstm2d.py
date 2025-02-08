import tensorflow as tf
import matplotlib.pyplot as dr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## Leggiamo i dati
X_train = np.load("C:/Users/CASALAB/Desktop/Gruppo_20_Favb/PreProcessing/Training_Set_NpArr.npy", allow_pickle=True)
X_valid = np.load("C:/Users/CASALAB/Desktop/Gruppo_20_Favb/PreProcessing/Validation_Set_NpArr.npy", allow_pickle=True)
X_test = np.load("C:/Users/CASALAB/Desktop/Gruppo_20_Favb/PreProcessing/Test_Set_NpArr.npy", allow_pickle=True)
y_train = np.load("C:/Users/CASALAB/Desktop/Gruppo_20_Favb/PreProcessing/Training_Set_Y.npy", allow_pickle=True)
y_valid = np.load("C:/Users/CASALAB/Desktop/Gruppo_20_Favb/PreProcessing/Validation_Set_Y.npy", allow_pickle=True)
y_test = np.load("C:/Users/CASALAB/Desktop/Gruppo_20_Favb/PreProcessing/Test_Set_Y.npy", allow_pickle=True)

class_names = ["Italian", "English", "German", "Spanish", "Dutch", "Russian", "Japanese", "French"]

#tf.random.set_seed(42)

callback = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=1)

model = tf.keras.Sequential()
model.add(tf.keras.layers.ConvLSTM2D(filters = 3, kernel_size = 3, input_shape=[150, 50, 50, 3], padding="same"))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(8, activation="softmax"))
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

history = model.fit(X_train, y_train, epochs=300, batch_size = 86, shuffle = True, validation_data=(X_valid, y_valid), callbacks=[callback])

print("Valutazione sul test set: ")
model.evaluate(X_test,y_test)

print("Previsioni: ")
y_proba = model.predict(X_test)
y_pred = y_proba.argmax(axis=-1)

#Accuracy del train e del validation
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

#Loss del train e del validation
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

#model.save('C:/Users/CASALAB/Desktop/Gruppo_20_Favb/Modelli/tentativo50')