import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from sklearn.model_selection import train_test_split

from paths import *
from vocatoadRNN import VocatoadRNN
from vocatoadTrainingData import VocatoadTrainingData
from soundProcessing import IndividualSound


class VocatoadTrainingComplexLabels(VocatoadTrainingData):
    def __init__(self, features="soundarray"):
        super().__init__(labels="speciesarray", features=features)
    
    def SetAttributes(self):
        X = self.audiodata[self.features]
        y = []
        for index in range(self.audiodata[self.labels].size):
            y.append([self.audiodata["familiesarray"][index],\
                self.audiodata["gendersarray"][index],\
                    self.audiodata["speciesarray"][index]])
        return np.array(X),np.array(y)
    
    def splitTestData(self, X, y):
        print(str(len(y[0])))
        X_test, X_val, y_test, y_val = train_test_split(X, y, test_size=0.25, random_state=123, stratify=y)
        return X_test, X_val, y_test, y_val


class VocatoadSpeciesCNN(VocatoadRNN):
    def __init__(self):
        self.modelFile = savedModelsPath.joinpath(type(self).__name__).absolute()
        self.savePath = speciesCNNModelWeightsFile
        self.labels = "speciesarray"
        self.historyFile = speciesCNNModelPath.joinpath("history").absolute()
        self.loss = "loss"
        self.accuracy = "accuracy"
        self.validationLoss = "val_loss"
        self.validationAccuracy = "val_accuracy"
        self.RedefineModel()
        self.ReloadModel()

    def CompileModel(self):
        self.model.compile(optimizer='adam',loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),metrics=['accuracy'])

    def RedefineModel(self):
        input_shape=(128,1000,3)
        CNNmodel = keras.models.Sequential()
        CNNmodel.add(Conv2D(128, (3, 3), activation='relu', input_shape=input_shape))
        CNNmodel.add(MaxPooling2D((2, 2)))
        CNNmodel.add(Dropout(0.2))
        CNNmodel.add(Conv2D(64, (3, 3), activation='relu'))
        CNNmodel.add(MaxPooling2D((2, 2)))
        CNNmodel.add(Dropout(0.2))
        CNNmodel.add(Conv2D(64, (3, 3), activation='relu'))
        CNNmodel.add(Flatten())
        CNNmodel.add(Dense(64, activation='relu'))
        CNNmodel.add(Dropout(0.2))
        CNNmodel.add(Dense(64, activation='relu'))
        CNNmodel.add(Dense(700, activation='softmax'))
        CNNmodel.summary()

        self.model = CNNmodel

        self.CompileModel()
    
    def IdentifySoundImage(self,soundfile, baseLength=None, outputFolder=None):
        data, baseLength = IndividualSound(soundfile,baseLength,outputFolder).getSoundImages()
        prediction = self.Predict(np.array([data]))
        return prediction

    def IdentifySoundImages(self,soundlist, baseLength=None, outputFolder=None):
        features = []
        for soundfile in soundlist:      
            data, baseLength = IndividualSound(soundfile,baseLength,outputFolder).getSoundImages()
            features.append([data])
        return self.Predict(np.array(np.concatenate(features,axis=0)))

if __name__ == "__main__":
    vocatoad = VocatoadSpeciesCNN()
    vocatoad.Train(VocatoadTrainingData("speciesarray","soundimages"))