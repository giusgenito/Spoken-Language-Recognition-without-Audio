import pandas as pd
import numpy as np
import cv2
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, classification_report
from tabulate import tabulate

## Leggiamo i dati
x_train = np.load("C:/Users/CASALAB/Desktop/Gruppo_20_Favb/SIMS_MOSEI/Preprocessing/x_train.npy", allow_pickle=True)
x_test = np.load("C:/Users/CASALAB/Desktop/Gruppo_20_Favb/SIMS_MOSEI/Preprocessing/x_test.npy", allow_pickle=True)
y_train = np.load("C:/Users/CASALAB/Desktop/Gruppo_20_Favb/SIMS_MOSEI/Preprocessing/y_train.npy", allow_pickle=True)
y_test = np.load("C:/Users/CASALAB/Desktop/Gruppo_20_Favb/SIMS_MOSEI/Preprocessing/y_test.npy", allow_pickle=True)

model = tf.keras.models.load_model('C:/Users/CASALAB/Desktop/Gruppo_20_Favb/SIMS_MOSEI/Modelli/tentativo56')
classes = ["English", "Chinese"]

y_proba = model.predict(x_test)

y_proba = y_proba.astype(int)

#Create confusion matrix
result = confusion_matrix(y_test, y_proba)

plt.figure(figsize=(10, 10))
plt.imshow(result, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Matrice di confusione')
plt.colorbar()
plt.xticks(np.arange(len(classes)), classes, rotation=45)
plt.yticks(np.arange(len(classes)), classes)
plt.xlabel('Etichetta predetta')
plt.ylabel('Etichetta reale')
thresh = result.max() / 2.
for i in range(result.shape[0]):
    for j in range(result.shape[1]):
        plt.text(j, i, format(result[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if result[i, j] > thresh else "black")

plt.show()

# Calcola l'accuratezza
accuracy = accuracy_score(y_test, y_proba)
print("Accuratezza: ", accuracy)

#Accuratezza per ogni classe
def accuracy_per_class(confusion_matrix):
    num_classes = confusion_matrix.shape[0]
    accuracies = np.zeros(num_classes)

    for i in range(num_classes):
        correct_predictions = confusion_matrix[i, i]
        total_samples = np.sum(confusion_matrix[i, :])
        accuracies[i] = correct_predictions / total_samples

    return accuracies

class_accuracies = accuracy_per_class(result)
#print(class_accuracies)

# Calcola la precision
precision = precision_score(y_test, y_proba, average=None)
#print("Precisione per classe: ", precision)

# Calcola il recall
recall = recall_score(y_test, y_proba, average=None)
#print("Recupero per classe: ", recall)

# Calcola il punteggio F1
f1 = f1_score(y_test, y_proba, average=None)
#print("Punteggio F1 per classe: ", f1)

# Crea un DataFrame con i valori
df = pd.DataFrame({'Classe': classes, 'Precision': precision, 'Recall': recall, 'F1-score': f1,'Accuracy': class_accuracies})

# Formatta le colonne del DataFrame
df['Precision'] = df['Precision'].apply(lambda x: '{:.2f}'.format(x))
df['Recall'] = df['Recall'].apply(lambda x: '{:.2f}'.format(x))
df['F1-score'] = df['F1-score'].apply(lambda x: '{:.2f}'.format(x))
df['Accuracy'] = df['Accuracy'].apply(lambda x: '{:.2f}'.format(x))

# Converte il DataFrame in una tabella formattata
table = tabulate(df, headers='keys', tablefmt='pretty', showindex=False)

# Stampa la tabella
print(table)


model.summary()