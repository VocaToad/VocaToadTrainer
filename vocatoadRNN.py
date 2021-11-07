import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import LSTM, Dense, Dropout
from keras.callbacks import ModelCheckpoint

from paths import *
from soundProcessing import SoundProcessing, IndividualSound

class VocatoadTrainingData():
    def __init__(self):
        if audioDataFile.exists():
            self.audiodata = np.load(audioDataFile)
        else:
            self.__ProcessAudios()

    def __ProcessAudios(self):
        soundProcessor = SoundProcessing()
        soundProcessor.ProcessSound()
        self.audiodata = soundProcessor.audiodata
    
    def GetTrainingData(self):
        X = self.audiodata["soundarray"]
        X = np.array((X-np.min(X))/(np.max(X)-np.min(X)))
        X = X/np.std(X)
        y = self.audiodata["speciesarray"]
        print(len(X))
        print(len(y))
        print("X e Y definidos")
        

        #Split twice to get the validation set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=123)
        #Print the shapes
        print(str(X_train.shape), str(X_test.shape), str(X_val.shape), str(len(y_train)), str(len(y_test)), str(len(y_val)))
        print("Conjuntos definidos")

        return X_train, X_test, X_val, y_train, y_test, y_val


class VocatoadRNN():
    def __init__(self):
        self.DefineRNN()
        self.ReloadModel()

    def ReloadModel(self):
        if modelWeightsFile.exists():
            self.model.load_weights(modelWeightsFile)

    def DefineRNN(self):
        input_shape=(128,1000)
        model = keras.Sequential()
        model.add(LSTM(128,input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(2048, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(192, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(96, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(48, activation='relu'))
        model.add(Dropout(0.4))
        model.add(Dense(5000, activation='softmax'))
        model.summary()

        model.compile(optimizer='adam',loss='SparseCategoricalCrossentropy',metrics=['acc'])

        self.model = model

    def Train(self):
        X_train, X_test, X_val, y_train, y_test, y_val = VocatoadTrainingData().GetTrainingData()

        # define the checkpoint
        cp1= ModelCheckpoint(filepath=modelWeightsFile, save_weights_only=True, monitor='loss', save_best_only=True, verbose=1, mode='min')
        callbacks_list = [cp1]

        history = self.model.fit(X_train, y_train, epochs=200, batch_size=72, 
                    validation_data=(X_val, y_val), shuffle=False, callbacks=callbacks_list)

        history_dict=history.history
        loss_values=history_dict['loss']
        acc_values=history_dict['acc']
        val_loss_values = history_dict['val_loss']
        val_acc_values=history_dict['val_acc']
        epochs=range(1,51)
        fig,(ax1,ax2)=plt.subplots(1,2,figsize=(15,5))
        ax1.plot(epochs,loss_values,'co',label='Training Loss')
        ax1.plot(epochs,val_loss_values,'m', label='Validation Loss')
        ax1.set_title('Training and validation loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax2.plot(epochs,acc_values,'co', label='Training accuracy')
        ax2.plot(epochs,val_acc_values,'m',label='Validation accuracy')
        ax2.set_title('Training and validation accuracy')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        plt.show()

        TrainLoss, Trainacc = self.model.evaluate(X_train,y_train)
        TestLoss, Testacc = self.model.evaluate(X_test, y_test)
        y_pred=self.Predict(X_test)
        print('Confusion_matrix: ',tf.math.confusion_matrix(y_test, np.argmax(y_pred,axis=1)))
    
    def Predict(self,sounds=[]):
        if not sounds:
            return None
        return self.model.predict(sounds)
    
    def IdentifySound(self,soundfile):
        data, baseLength = IndividualSound(soundfile).getSoundFeatures()
        return self.Predict([data])

    def IdentifySounds(self,soundlist):
        features = []
        baseLength = None
        for soundfile in soundlist:      
            data, baseLength = IndividualSound(soundfile,baseLength).getSoundFeatures()
        features.append(data)
        return self.Predict(np.concatenate(features,axis=0))



if __name__ == "__main__":
    vocatoad = VocatoadRNN()
    vocatoad.Train()
