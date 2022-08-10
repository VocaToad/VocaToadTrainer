import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from sklearn.model_selection import train_test_split

from paths import *
from vocatoadRNN import VocatoadRNN
from vocatoadTrainingData import VocatoadTrainingData

class VocatoadFamilyCNN(VocatoadRNN):
    def __init__(self):
        self.savePath = familyCNNModelWeightsFile
        self.labels = "familiesarray"
        self.RedefineModel()
        self.ReloadModel()

    def RedefineModel(self):
        input_shape=(128,1000,3)
        CNNmodel = keras.models.Sequential()
        CNNmodel.add(Conv2D(128, (3, 3), activation='relu', input_shape=input_shape))
        CNNmodel.add(MaxPooling2D((2, 2)))
        CNNmodel.add(Dropout(0.2))
        CNNmodel.add(Conv2D(100, (3, 3), activation='relu'))
        CNNmodel.add(MaxPooling2D((2, 2)))
        CNNmodel.add(Dropout(0.2))
        CNNmodel.add(Conv2D(100, (3, 3), activation='relu'))
        CNNmodel.add(Flatten())
        CNNmodel.add(Dense(100, activation='relu'))
        CNNmodel.add(Dropout(0.2))
        CNNmodel.add(Dense(50, activation='relu'))
        CNNmodel.add(Dense(25, activation='softmax'))

        CNNmodel.compile(optimizer='adam',loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),metrics=['accuracy'])

        self.model = CNNmodel

if __name__ == "__main__":
    vocatoad = VocatoadFamilyCNN()
    vocatoad.Train(VocatoadTrainingData("familiesarray","soundimages"))
