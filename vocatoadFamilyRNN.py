from tensorflow import keras
from tensorflow.keras.layers import LSTM, Dense, Dropout

from paths import *
from vocatoadRNN import VocatoadRNN

class VocatoadFamilyRNN(VocatoadRNN):
    def __init__(self):
        self.savePath = familyModelWeightsFile
        self.labels = "familiesarray"
        self.RedefineModel()
        self.ReloadModel()

    def RedefineModel(self):
        input_shape=(128,1000)
        model = keras.Sequential()
        model.add(LSTM(128,input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(160, activation='relu'))
        model.add(Dropout(0.4))
        model.add(Dense(80, activation='relu'))
        model.add(Dropout(0.4))
        model.add(Dense(40, activation='softmax'))
        model.summary()

        model.compile(optimizer='adam',loss='SparseCategoricalCrossentropy',metrics=['accuracy'])

        self.model = model

if __name__ == "__main__":
    vocatoad = VocatoadFamilyRNN()
    vocatoad.Train()
