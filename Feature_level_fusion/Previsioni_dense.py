import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv

X_train = np.load("C:/Users/CASALAB/Desktop/Gruppo_20_Favb/PreProcessing/Training_Set_NpArr.npy", allow_pickle=True)
X_valid = np.load("C:/Users/CASALAB/Desktop/Gruppo_20_Favb/PreProcessing/Validation_Set_NpArr.npy", allow_pickle=True)
X_test = np.load("C:/Users/CASALAB/Desktop/Gruppo_20_Favb/PreProcessing/Test_Set_NpArr.npy", allow_pickle=True)
y_train = np.load("C:/Users/CASALAB/Desktop/Gruppo_20_Favb/PreProcessing/Training_Set_Y.npy", allow_pickle=True)
y_valid = np.load("C:/Users/CASALAB/Desktop/Gruppo_20_Favb/PreProcessing/Validation_Set_Y.npy", allow_pickle=True)
y_test = np.load("C:/Users/CASALAB/Desktop/Gruppo_20_Favb/PreProcessing/Test_Set_Y.npy", allow_pickle=True)
path_test = "C:/Users/CASALAB/Desktop/Gruppo_20_Favb/Dataset/Lips_video_10_seconds_division_train_test_val/Test"

test_df = pd.read_csv("C:/Users/CASALAB/Desktop/Gruppo_20_Favb/FileCsv/file_test.csv")
dir_test = path_test
video_name_paths = test_df["VideoName"].values.tolist()
video_name_paths = [name.replace(".avi", "") for name in video_name_paths]
video_name_paths = [name.replace("_m", "") for name in video_name_paths]

model = tf.keras.models.load_model('C:/Users/CASALAB/Desktop/Gruppo_20_Favb/Modelli/tentativo47')
pred_dense = model.predict(X_test)

with open('C:/Users/CASALAB/Desktop/Gruppo_20_Favb/FileCsv/Previsioni_dense.csv', 'w', newline='') as csvfile:
   writer = csv.writer(csvfile)
   writer.writerow(["Ita","Ing","Ted","Spa","Oland","Rus","Giap","Fra"])
   for pred in pred_dense:
      writer.writerow(pred)

with open('C:/Users/CASALAB/Desktop/Gruppo_20_Favb/FileCsv/Nomi_test_set.csv', 'w', newline='') as csvfile:
   writer = csv.writer(csvfile)
   writer.writerow(["Nomi_video"])
   for value in video_name_paths:
      writer.writerow([value])



