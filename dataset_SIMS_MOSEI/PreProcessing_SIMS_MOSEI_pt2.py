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

""" Importiamo i dati """

percorso_Train_MOSEI = 'C:/Users/CASALAB/Desktop/Gruppo_20_Favb/SIMS_MOSEI/Preprocessing/Train_MOSEI.npy'
percorso_Train_SIMS = 'C:/Users/CASALAB/Desktop/Gruppo_20_Favb/SIMS_MOSEI/Preprocessing/Train_SIMS.npy'

percorso_Test_MOSEI = 'C:/Users/CASALAB/Desktop/Gruppo_20_Favb/SIMS_MOSEI/Preprocessing/Test_MOSEI_full.npy'
percorso_Test_SIMS = 'C:/Users/CASALAB/Desktop/Gruppo_20_Favb/SIMS_MOSEI/Preprocessing/Test_SIMS_full.npy'

Train_MOSEI = np.load(percorso_Train_MOSEI, allow_pickle=True)
Train_SIMS = np.load(percorso_Train_SIMS, allow_pickle=True)
Train_set = np.concatenate((Train_MOSEI, Train_SIMS), axis=0)

Test_MOSEI = np.load(percorso_Test_MOSEI, allow_pickle=True)
Test_SIMS = np.load(percorso_Test_SIMS, allow_pickle=True)
Test_set = np.concatenate((Test_MOSEI, Test_SIMS), axis=0)

""" Mischiamo le righe e salviamo """
np.random.shuffle(Train_set)
np.random.shuffle(Test_set)

np.save('C:/Users/CASALAB/Desktop/Gruppo_20_Favb/SIMS_MOSEI/Preprocessing/Train_set.npy', Train_set)
np.save('C:/Users/CASALAB/Desktop/Gruppo_20_Favb/SIMS_MOSEI/Preprocessing/Test_set.npy', Test_set)

##----------------------------------------------------------------------------------------------------------------------
