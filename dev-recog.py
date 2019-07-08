import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, Zeropadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import Averagepooling2D, Maxpooling2D, Dropout, GlobalMaxpooling, GlobalAveragepooling2D
from keras.untils import np_until, print_summary
import pandas as pd
from keras.models import sequential
from keras.callbacks import ModelCheckpoint
import keras.backend as K


data=pd.read_csv("data.csv")
dataset=np.array(data)
np.random.shuffle(dataset)
X=dataset
Y=dataset
X=X[:, 0:1024]
Y=Y[:, 1024]
X_train=X[0: 70000, :]
X_train=X_train/255
X_text=X[70000:72001, :]
X_text=X_text/255

Y=Y.reshape(Y.shape[0],1)
Y_train=Y[0:70000,:]
Y_train=Y_train.T
Y_text=Y[70000:72001,:]
Y_text=Y_text.T

