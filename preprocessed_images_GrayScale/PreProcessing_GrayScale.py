import pandas as pd
import numpy as np
import cv2

path_train = "C:/Users/CASALAB/Desktop/Gruppo_20_Favb/Dataset/Lips_video_10_seconds_division_train_test_val/Train"
path_valid = "C:/Users/CASALAB/Desktop/Gruppo_20_Favb/Dataset/Lips_video_10_seconds_division_train_test_val/Validation"
path_test = "C:/Users/CASALAB/Desktop/Gruppo_20_Favb/Dataset/Lips_video_10_seconds_division_train_test_val/Test"

train_df = pd.read_csv("C:/Users/CASALAB/Desktop/Gruppo_20_Favb/FileCsv/file_train.csv")
valid_df = pd.read_csv("C:/Users/CASALAB/Desktop/Gruppo_20_Favb/FileCsv/file_val.csv")
test_df = pd.read_csv("C:/Users/CASALAB/Desktop/Gruppo_20_Favb/FileCsv/file_test.csv")

#------------------------------------------------------------------------------------------------

dir_train = path_train
dir_valid = path_valid
dir_test = path_test

def num_less_zero(num):
  if num == 1:
    return num-1
  if num == 2:
    return num-1
  if num == 3:
    return num-1
  if num == 4:
    return num-1
  if num == 5:
    return num-1
  if num == 6:
    return num-1
  if num == 7:
    return num-1
  if num == 8:
    return num-1

def prepare_train(train_df, dir):
    num_samples = len(train_df)
    video_name_paths = train_df["VideoName"].values.tolist()
    labels = []
    path_video = []
    for i in range(num_samples):
        path_video.append(dir + "/" + video_name_paths[i])
        labels.append(num_less_zero(int(str(video_name_paths[i])[0])))
    return path_video, labels

def prepare_validation(validation_df, dir):
    num_samples = len(validation_df)
    video_name_paths = validation_df["VideoName"].values.tolist()
    labels = []
    path_video = []
    for i in range(num_samples):
        path_video.append(dir + "/" + video_name_paths[i])
        labels.append(num_less_zero(int(str(video_name_paths[i])[0])))
    return path_video, labels

def prepare_test(test_df, dir):
    num_samples = len(test_df)
    video_name_paths = test_df["VideoName"].values.tolist()
    labels = []
    path_video = []
    for i in range(num_samples):
        path_video.append(dir + "/" + video_name_paths[i])
        labels.append(num_less_zero(int(str(video_name_paths[i])[0])))
    return path_video, labels

path_train, labels_train = prepare_train(train_df, dir_train)
path_valid, labels_valid = prepare_train(valid_df, dir_valid)
path_test, labels_test = prepare_test(test_df, dir_test)

#------------------------------------------------------------------------------------------------

def pre_processing_train(path_train, labels_train):
    path_train = path_train
    labels_train = labels_train
    array_videos = []

    for i in range(len(path_train)):
        video_array = []
        n_frame = 0
        cap = cv2.VideoCapture(path_train[i])
        #print(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

        while cap.isOpened():
            ret, image = cap.read()
            n_frame += 1
            if not ret or n_frame > 150:
                break
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, (50, 50), interpolation=cv2.INTER_AREA)
            image.shape = (50, 50, 1)
            video_array.append(image)

        video_array = np.array(video_array)
        array_videos.append([video_array, labels_train[i]])

    x = []
    y = []
    for variabile, lingua in array_videos:
        x.append(variabile)
        y.append(lingua)

    x = np.array(x)
    y = np.array(y)

    np.save('C:/Users/CASALAB/Desktop/Gruppo_20_Favb/PreProcessing/Training_Set_NpArr_GS.npy', x)
    np.save('C:/Users/CASALAB/Desktop/Gruppo_20_Favb/PreProcessing/Training_Set_Y_GS.npy', y)


pre_processing_train(path_train, labels_train)

#------------------------------------------------------------------------------------------------

def pre_processing_validation(path_valid, labels_valid):
    path_valid = path_valid
    labels_valid = labels_valid
    array_videos = []

    for i in range(len(path_valid)):
        video_array = []  ## ogni elemento è un frame di un video
        n_frame = 0
        cap = cv2.VideoCapture(path_valid[i])

        while cap.isOpened():
            ret, image = cap.read()
            n_frame += 1
            if not ret or n_frame > 150:
                break
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, (50, 50), interpolation=cv2.INTER_AREA)
            image.shape = (50, 50, 1)
            video_array.append(image)

        video_array = np.array(video_array)
        array_videos.append([video_array, labels_valid[i]])

    x = []
    y = []

    for variabile, lingua in array_videos:
        x.append(variabile)
        y.append(lingua)

    x = np.array(x)
    y = np.array(y)

    np.save('C:/Users/CASALAB/Desktop/Gruppo_20_Favb/PreProcessing/Validation_Set_NpArr_GS.npy', x)
    np.save('C:/Users/CASALAB/Desktop/Gruppo_20_Favb/PreProcessing/Validation_Set_Y_GS.npy', y)


pre_processing_validation(path_valid, labels_valid)

#------------------------------------------------------------------------------------------------

def pre_processing_test(path_test, labels_test):
    path_test = path_test
    labels_test = labels_test
    array_videos = []

    for i in range(len(path_test)):
        video_array = []  #
        n_frame = 0
        cap = cv2.VideoCapture(path_test[i])

        while cap.isOpened():
            ret, image = cap.read()
            n_frame += 1
            if not ret or n_frame > 150:
                break
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, (50, 50), interpolation=cv2.INTER_AREA)
            image.shape = (50, 50, 1)
            video_array.append(image)

        video_array = np.array(video_array)
        array_videos.append([video_array, labels_test[i]])

    x = []
    y = []
    for variabile, lingua in array_videos:
        x.append(variabile)
        y.append(lingua)
    x = np.array(x)
    y = np.array(y)

    np.save('C:/Users/CASALAB/Desktop/Gruppo_20_Favb/PreProcessing/Test_Set_NpArr_GS.npy', x)
    np.save('C:/Users/CASALAB/Desktop/Gruppo_20_Favb/PreProcessing/Test_Set_Y_GS.npy', y)

pre_processing_test(path_test, labels_test)

#------------------------------------------------------------------------------------------------