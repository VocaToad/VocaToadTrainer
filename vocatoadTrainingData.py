import numpy as np
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

from paths import *
from soundProcessing import SoundProcessing



class VocatoadTrainingData():
    def __init__(self,labels="speciesarray",features="soundarray"):
        self.labels = labels
        self.features = features
        if audioDataFile.exists():
            self.audiodata = np.load(audioDataFile)
        else:
            self.__ProcessAudios()

    def __ProcessAudios(self):
        soundProcessor = SoundProcessing()
        soundProcessor.ProcessSound()
        self.audiodata = soundProcessor.audiodata

    def SetAttributes(self):
        X = self.audiodata[self.features]
        y = self.audiodata[self.labels]
        return np.array(X),np.array(y)

    def splitTestData(self, X, y):
        X_test, X_val, y_test, y_val = train_test_split(X, y, test_size=0.25, random_state=123)
        return X_test, X_val, y_test, y_val
    
    def GetTrainingData(self):
        
        X, y = self.SetAttributes()

        X_train = X
        y_train = y

        #Split twice to get the validation set
        X_test, X_val, y_test, y_val = self.splitTestData(X_train, y_train)
        #Print the shapes
        print(str(X_train.shape), str(X_test.shape), str(X_val.shape), str(len(y_train)), str(len(y_test)), str(len(y_val)))
        print("Conjuntos definidos")

        return X_train, X_test, X_val, y_train, y_test, y_val