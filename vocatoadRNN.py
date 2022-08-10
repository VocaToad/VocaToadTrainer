import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import LSTM, Dense, Dropout
from keras.callbacks import ModelCheckpoint

from paths import *
from vocatoadTrainingData import VocatoadTrainingData
from soundProcessing import IndividualSound


class VocatoadRNN():
    history = None

    def __init__(self):
        self.modelFile = savedModelsPath.joinpath(type(self).__name__).absolute()
        self.savePath = speciesModelWeightsFile
        self.historyFile = speciesModelPath.joinpath("history").absolute()
        self.labels = None
        self.loss = "loss"
        self.accuracy = "acc"
        self.validationLoss = "val_loss"
        self.validationAccuracy = "val_acc"
        self.DefineRNN()
        self.ReloadModel()

    def ReloadModel(self):
        if self.modelFile.exists():
            print("Reloading Model at "+str(self.modelFile))
            self.model = keras.models.load_model(self.modelFile)
            return
        if self.savePath.exists():
            print("Reloading Model at "+str(self.savePath))
            self.model.load_weights(self.savePath)
            self.CompileModel()
    
    def CompileModel(self):
        self.model.compile(optimizer='adam',loss='SparseCategoricalCrossentropy',metrics=['acc'])
        self.model.summary()


    def DefineRNN(self):
        input_shape=(128,1000)
        model = keras.Sequential()
        model.add(LSTM(6400,input_shape=input_shape))
        #model.add(Dropout(0.2))
        model.add(Dense(3200, activation='relu'))
        model.add(Dense(1600, activation='relu'))
        model.add(Dense(800, activation='softmax'))
        model.summary()

        self.model = model

        self.CompileModel()
       

    def Train(self, trainingData=None,labels=None):
        if not labels:
            labels = self.labels
        if not trainingData:
            trainingData = VocatoadTrainingData(labels)
        X_train, X_test, X_val, y_train, y_test, y_val = trainingData.GetTrainingData()

        # define the checkpoint
        cp1= ModelCheckpoint(filepath=self.savePath, save_weights_only=False, monitor='loss', save_best_only=True, verbose=1, mode='min')
        callbacks_list = [cp1]

        self.history = self.model.fit(X_train, y_train, epochs=50, batch_size=72, 
                    validation_data=(X_val, y_val), shuffle=False, callbacks=callbacks_list)

        self.SaveHistory()
        self.SaveModel()

        history_dict=self.history.history
        loss_values=history_dict[self.loss]
        acc_values=history_dict[self.accuracy]
        val_loss_values = history_dict[self.validationLoss]
        val_acc_values=history_dict[self.validationAccuracy]
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
        if not len(sounds):
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

    def SaveHistory(self): 
        hist_json_file = str(self.historyFile)+datetime.now().strftime("%d-%m-%Y-%H-%M-%S")+".json"   
        hist_df = pd.DataFrame(self.history.history) 
        with open(hist_json_file, mode='w') as f:
            hist_df.to_json(f)

    def SaveModel(self):
        self.model.save(self.modelFile)


if __name__ == "__main__":
    vocatoad = VocatoadRNN()
    vocatoad.Train()
