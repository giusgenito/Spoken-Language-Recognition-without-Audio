##----------------------------------------------------------------------------------------------------------------------

import pandas as pd
import numpy as np
import cv2
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix

##----------------------------------------------------------------------------------------------------------------------



##----------------------------------------------------------------------------------------------------------------------

"""Importiamo i dataset. Da notare che non esiste il validation set"""

path_train_MOSEI = "C:/Users/CASALAB/Desktop/SIMSMOSEI/CMU-MOSEI/SubjectIndependent-m/Training"
path_test_MOSEI = "C:/Users/CASALAB/Desktop/SIMSMOSEI/CMU-MOSEI/SubjectIndependent-m/Testing"
path_train_SIMS = "C:/Users/CASALAB/Desktop/SIMSMOSEI/CH-SIMS/SubjectIndependent-m/Training"
path_test_SIMS = "C:/Users/CASALAB/Desktop/SIMSMOSEI/CH-SIMS/SubjectIndependent-m/Testing"

##----------------------------------------------------------------------------------------------------------------------


##----------------------------------------------------------------------------------------------------------------------

"""Preprocessing del train set del dataset MOSEI"""

cartella_principale = path_train_MOSEI
array_videos = []

"""Il primo problema da risolvere è che i video sono dentro a delle sottocartelle"""

for nome_cartella in os.listdir(cartella_principale):

    sottocartella = os.path.join(cartella_principale, nome_cartella)

    """Controlla se l'elemento è una sottocartella"""

    if os.path.isdir(sottocartella):
        #print(f"Elaborazione della sottocartella '{nome_cartella}':")

        """Itera su tutti i file nella sottocartella"""

        for nome_file in os.listdir(sottocartella):
            percorso_file = os.path.join(sottocartella, nome_file)
            video_array = []
            n_frame = 0
            cap = cv2.VideoCapture(percorso_file)

            """Apriamo il video e cambiamo la shape e scaliamo i video in formato 50x50"""

            while cap.isOpened():
                ret, image = cap.read()
                n_frame += 1
                if not ret or n_frame > 100:
                    break
                image = cv2.resize(image, (50, 50), interpolation=cv2.INTER_AREA)
                image.shape = (50, 50, 3)
                video_array.append(image)

            """Rendiamo i video elaborati in formato np.array"""

            video_array = np.array(video_array)

            """Salviamo il video in forma matriciale in una lista, 0 indica la classe, ossia Inglese"""
            array_videos.append([video_array, 0])


""" Trasformiamo e salviamo la lista di matrici in formato np.array per poi poterla importare in altri script """

np.save('C:/Users/CASALAB/Desktop/Gruppo_20_Favb/SIMS_MOSEI/Preprocessing/Train_MOSEI.npy', np.array(array_videos, dtype=object))
##----------------------------------------------------------------------------------------------------------------------



##----------------------------------------------------------------------------------------------------------------------

"""Preprocessing del TRAIN SET del dataset SIMS"""

cartella_principale = path_train_SIMS
array_videos = []

"""Svolgiamo la stessa operazione con il dataset SIMS, prendiamo i video dalle sottocartelle e li modifichiamo"""

for nome_cartella in os.listdir(cartella_principale):

    sottocartella = os.path.join(cartella_principale, nome_cartella)

    """Controlla se l'elemento è una sottocartella"""

    if os.path.isdir(sottocartella):

        #print(f"Elaborazione della sottocartella '{nome_cartella}':")

        """Itera su tutti i file nella sottocartella"""

        for nome_file in os.listdir(sottocartella):
            percorso_file = os.path.join(sottocartella, nome_file)
            video_array = []
            n_frame = 0
            cap = cv2.VideoCapture(percorso_file)

            """Apriamo il video e cambiamo la shape e scaliamo i video in formato 50x50"""

            while cap.isOpened():
                ret, image = cap.read()
                n_frame += 1
                if not ret or n_frame > 100:
                    break
                image = cv2.resize(image, (50, 50), interpolation=cv2.INTER_AREA)
                # print(image.shape)
                image.shape = (50, 50, 3)
                video_array.append(image)

            """Rendiamo i video elaborati in formato np.array"""

            video_array = np.array(video_array)

            """Salviamo il video in forma matriciale in una lista"""

            array_videos.append([video_array, 1])

""" Trasformiamo e salviamo la lista di matrici in formato np.array per poi poterla importare in altri script """

np.save('C:/Users/CASALAB/Desktop/Gruppo_20_Favb/SIMS_MOSEI/Preprocessing/Train_SIMS.npy', np.array(array_videos, dtype=object))
##----------------------------------------------------------------------------------------------------------------------



##----------------------------------------------------------------------------------------------------------------------

"""Preprocessing del TEST SET del dataset MOSEI"""

cartella_principale = path_test_MOSEI
array_videos = []

for nome_cartella in os.listdir(cartella_principale):

    sottocartella = os.path.join(cartella_principale, nome_cartella)

    """Controlla se l'elemento è una sottocartella"""

    if os.path.isdir(sottocartella):
        #print(f"Elaborazione della sottocartella '{nome_cartella}':")

        """Itera su tutti i file nella sottocartella"""

        for nome_file in os.listdir(sottocartella):
            percorso_file = os.path.join(sottocartella, nome_file)
            video_array = []
            n_frame = 0
            cap = cv2.VideoCapture(percorso_file)

            while cap.isOpened():
                ret, image = cap.read()
                n_frame += 1
                if not ret or n_frame > 100:
                    break
                image = cv2.resize(image, (50, 50), interpolation=cv2.INTER_AREA)
                image.shape = (50, 50, 3)
                video_array.append(image)

            video_array = np.array(video_array)
            array_videos.append([video_array, 0])

""" Trasformiamo e salviamo la lista di matrici in formato np.array per poi poterla importare in altri script """

np.save('C:/Users/CASALAB/Desktop/Gruppo_20_Favb/SIMS_MOSEI/Preprocessing/Test_MOSEI_full.npy', np.array(array_videos, dtype=object))

##----------------------------------------------------------------------------------------------------------------------



##----------------------------------------------------------------------------------------------------------------------

"""Preprocessing del TEST SET del dataset SIMS"""

cartella_principale = path_test_SIMS
array_videos = []

for nome_cartella in os.listdir(cartella_principale):

    sottocartella = os.path.join(cartella_principale, nome_cartella)

    """Controlla se l'elemento è una sottocartella"""

    if os.path.isdir(sottocartella):
        #print(f"Elaborazione della sottocartella '{nome_cartella}':")

        """Itera su tutti i file nella sottocartella"""

        for nome_file in os.listdir(sottocartella):
            percorso_file = os.path.join(sottocartella, nome_file)
            video_array = []
            n_frame = 0
            cap = cv2.VideoCapture(percorso_file)

            while cap.isOpened():
                ret, image = cap.read()
                n_frame += 1
                if not ret or n_frame > 100:
                    break
                image = cv2.resize(image, (50, 50), interpolation=cv2.INTER_AREA)
                image.shape = (50, 50, 3)
                video_array.append(image)

            video_array = np.array(video_array)
            array_videos.append([video_array, 1])

""" Trasformiamo e salviamo la lista di matrici in formato np.array per poi poterla importare in altri script """

np.save('C:/Users/CASALAB/Desktop/Gruppo_20_Favb/SIMS_MOSEI/Preprocessing/Test_SIMS_full.npy', np.array(array_videos, dtype=object))

##----------------------------------------------------------------------------------------------------------------------
