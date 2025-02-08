## --------------------
## Importiamo le librerie
## --------------------

import warnings
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, classification_report
import csv
import pandas as pd
import numpy as np
from tabulate import tabulate

## --------------------
## Importiamo i file csv

## Previsioni_dense = Probabilità previste dall'ultimo layer della rete neurale
## Nomi_test_set = Nomi dei video del dataset di test
## pred_probaGNB = Probabilità previste dal modello GaussianGNB sul test set di 160 video
## pred_probaGPC = Probabilità previste dal modello GaussianGPC sul test set di 160 video
## Nomi_PNN = Nomi dei video nel dataset di test preso dal gruppo 24
## y_lable = Le classi, ossia la lingua parlata da quel video
## --------------------

Dense_layer = pd.read_csv("C:/Users/CASALAB/Desktop/Gruppo_20_Favb/FileCsv/Previsioni_dense.csv")
Nomi_test_set = pd.read_csv("C:/Users/CASALAB/Desktop/Gruppo_20_Favb/FileCsv/Nomi_test_set.csv")
Proba_GNB = pd.read_csv("C:/Users/CASALAB/Desktop/Gruppo_20_Favb/FileCsv/pred_probaGNB.csv")
Proba_GPC = pd.read_csv("C:/Users/CASALAB/Desktop/Gruppo_20_Favb/FileCsv/pred_probaGPC.csv")
Nomi_PNN_test_set = pd.read_csv("C:/Users/CASALAB/Desktop/Gruppo_20_Favb/FileCsv/nomi_PNN.csv")
y_lable = pd.read_csv("C:/Users/CASALAB/Desktop/Gruppo_20_Favb/FileCsv/Y_lable_test_con_nomi.csv")

from pandas._config.display import detect_console_encoding
## --------------------
## I nomi dei video vengono uniti con le probabilità previste
## --------------------
Dens_layer_nome_video = pd.concat([Nomi_test_set,Dense_layer], axis=1)
Proba_GNB_nome_video = pd.concat([Nomi_PNN_test_set,Proba_GNB], axis=1)
Proba_GPC_nome_video = pd.concat([Nomi_PNN_test_set,Proba_GPC], axis=1)

## --------------------
## Viene fatta la Join sui nomi in modo da avere la corrispondenza tra le probabilità previste
## --------------------

Dense_GNB =  pd.merge(Dens_layer_nome_video, Proba_GNB_nome_video, on="Nomi_video")
Dense_GNB =  pd.merge(Dense_GNB,y_lable, on="Nomi_video")

Dense_GPC =  pd.merge(Dens_layer_nome_video, Proba_GNB_nome_video, on='Nomi_video')
Dense_GPC =  pd.merge(Dense_GPC,y_lable, on="Nomi_video")

## --------------------
## Verifichiamo se la creazione del dataset è avvenuta correttamente
## L'opzione pd.options.display.max_columns serve a mostrare tutte le colonne
## --------------------

pd.options.display.max_columns = None

#print("\n")
#print("Dense_GNB: ")
#print(Dense_GNB.head(5))
#print("\n")
#print("Dense_GPC: ")
#print(Dense_GPC.head(5))

## --------------------
## Dividiamo gli individui
## Se le prime 4 cifre del nome del video sono le stesse allora
## l'individuo è lo stesso.
## --------------------

conteggio_sequenze = {}
train_set=pd.DataFrame(columns=["Nomi_video","Ita","Ing","Ted","Spa","Oland","Rus","Giap","Fra","Italiano",
                                "Inglese","Tedesco", "Spagnolo", "Olandese","Russo","Giapponese","Francese","Etichette"])

test_set=pd.DataFrame(columns=["Nomi_video","Ita","Ing","Ted","Spa","Oland","Rus","Giap","Fra","Italiano",
                               "Inglese","Tedesco", "Spagnolo", "Olandese","Russo","Giapponese","Francese","Etichette"])

for i in range(len(Dense_GNB)):

  sequenza = Dense_GNB["Nomi_video"][i]

  # Separare i numeri utilizzando il carattere delimitatore "_"
  numeri = sequenza.split("_")

  # Selezionare solo i primi 4 numeri
  primi_quattro_numeri = tuple(numeri[:4])
  if primi_quattro_numeri in conteggio_sequenze:
    conteggio_sequenze[primi_quattro_numeri] += 1
    if conteggio_sequenze[primi_quattro_numeri] <= 3:
      train_set.loc[i]=Dense_GNB.loc[i]
    if conteggio_sequenze[primi_quattro_numeri] > 3:
      test_set.loc[i]=Dense_GNB.loc[i]

  else:
    conteggio_sequenze[primi_quattro_numeri] = 1
    train_set.loc[i]=Dense_GNB.loc[i]

x_train_GNB = train_set.iloc[:, 1:17]
y_train_GNB = train_set["Etichette"]
y_train_GNB = list(y_train_GNB)
#print(type(list(y_train_GNB)))
x_test_GNB = test_set.iloc[:, 1:17]
y_test_GNB = test_set["Etichette"]
y_test_GNB = list(y_test_GNB)

## ----------------------
## Ignoriamo alcuni warining che derivano da alcuni classificatori
## ----------------------

warnings.filterwarnings("ignore")

modello = RandomForestClassifier()
modello.fit(x_train_GNB, y_train_GNB)
classifiers_predictions = modello.predict(x_test_GNB)
classifiers_accuracy = accuracy_score(y_test_GNB, classifiers_predictions)
print(f"Accuratezza {modello}: ", classifiers_accuracy)

classes = ["Italian", "English", "German", "Spanish", "Dutch", "Russian", "Japanese", "French"]

result = confusion_matrix(y_test_GNB, classifiers_predictions)

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
accuracy = accuracy_score(y_test_GNB, classifiers_predictions)
print("Accuratezza: ", accuracy)
print(len(y_test_GNB))
print(len(classifiers_predictions))
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
precision = precision_score(y_test_GNB, classifiers_predictions, average=None)
#print("Precisione per classe: ", precision)

# Calcola il recall
recall = recall_score(y_test_GNB, classifiers_predictions, average=None)
#print("Recupero per classe: ", recall)

# Calcola il punteggio F1
f1 = f1_score(y_test_GNB, classifiers_predictions, average=None)
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



'''
x_train_GPC = train_set.iloc[:, 1:17]
y_train_GPC = train_set["Etichette"]
y_train_GPC = list(y_train_GPC)
print(type(list(y_train_GPC)))
x_test_GPC = test_set.iloc[:, 1:17]
y_test_GPC = test_set["Etichette"]
y_test_GPC = list(y_test_GPC)

## ----------------------
## Ignoriamo alcuni warining che derivano da alcuni classificatori
## ----------------------
warnings.filterwarnings("ignore")

modello = RandomForestClassifier()
modello.fit(x_train_GPC, y_train_GPC)
classifiers_predictions = modello.predict(x_test_GPC)
classifiers_accuracy = accuracy_score(y_test_GPC, classifiers_predictions)
print(f"Accuratezza {modello}: ", classifiers_accuracy)

classes = ["Italian", "English", "German", "Spanish", "Dutch", "Russian", "Japanese", "French"]

result = confusion_matrix(y_test_GPC, classifiers_predictions)

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
accuracy = accuracy_score(y_test_GPC, classifiers_predictions)
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
precision = precision_score(y_test_GPC, classifiers_predictions, average=None)
#print("Precisione per classe: ", precision)

# Calcola il recall
recall = recall_score(y_test_GPC, classifiers_predictions, average=None)
#print("Recupero per classe: ", recall)

# Calcola il punteggio F1
f1 = f1_score(y_test_GPC, classifiers_predictions, average=None)
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
'''

