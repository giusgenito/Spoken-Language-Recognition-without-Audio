import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
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
import csv

train = pd.read_csv('C:/Users/CASALAB/Desktop/Gruppo_20_Favb/Dataset_Feature_level_score/Dataset/Train_babele_cityblock_random6.csv')
test = pd.read_csv('C:/Users/CASALAB/Desktop/Gruppo_20_Favb/Dataset_Feature_level_score/Dataset/Test_babele_cityblock_random6.csv')

#Preprocessing dei dati
del train["video-frame"]
x_train = train
y_train = train.pop("target")
#print(x_train)

del test["video-frame"]
x_test = test
y_test = test.pop("target")
#print(pd.array(y_test))

y_test = list(y_test)
with open('C:/Users/CASALAB/Desktop/Gruppo_20_Favb/FileCsv/y_test.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["label"])

    y_test_transposed = [[value] for value in y_test]

    # Scrivi i valori lungo la colonna
    writer.writerows(y_test_transposed)

classifiers = [
    KNeighborsClassifier(),
    SVC(kernel="rbf"),
    SVC(),
    GaussianProcessClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    MLPClassifier(max_iter=2000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
]

for modello in classifiers:
    modello.fit(x_train, y_train)

    classifiers_predictions = modello.predict(x_test)

    classifiers_accuracy = accuracy_score(y_test, classifiers_predictions)
    #print(f"Accuratezza {modello}: ", classifiers_accuracy)

#GaussianProcessClassifier
modello_GPC = GaussianProcessClassifier()
modello_GPC.fit(x_train, y_train)

classifiers_predictions = modello_GPC.predict(x_test)
classifiers_accuracy = accuracy_score(y_test, classifiers_predictions)
print("Accuratezza_GPC:", classifiers_accuracy)
pred_proba_GPC = modello_GPC.predict_proba(x_test)

'''
#Creazione del csv
with open('/content/drive/Shareddrives/FVAB_BABELE/csv_Classificatore/pred_probaGPC.csv', 'w') as csvfile:
   writer = csv.writer(csvfile)
   writer.writerow(["Italiano", "Inglese", "Tedesco", "Spagnolo", "Olandese", "Russo", "Giapponese", "Francese"])
   for riga in pred_proba_GPC:
      writer.writerow(riga)
'''

#GaussianNB
modello_GNB = GaussianNB()
modello_GNB.fit(x_train, y_train)

classifiers_predictions = modello_GNB.predict(x_test)
classifiers_accuracy = accuracy_score(y_test, classifiers_predictions)
print("Accuratezza_GNB:", classifiers_accuracy)
pred_proba_GNB = modello_GNB.predict_proba(x_test)

'''
#Creazione del csv
with open('/content/drive/Shareddrives/FVAB_BABELE/csv_Classificatore/pred_probaGNB.csv', 'w') as csvfile:
   writer = csv.writer(csvfile)
   writer.writerow(["Italiano", "Inglese", "Tedesco", "Spagnolo", "Olandese", "Russo", "Giapponese", "Francese"])
   for riga in pred_proba_GNB:
      writer.writerow(riga)
'''