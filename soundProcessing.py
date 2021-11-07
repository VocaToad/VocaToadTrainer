import pandas as pd
import numpy as np
import librosa
import librosa.display
import warnings
warnings.filterwarnings('ignore')

from paths import *
from csvPopulator import csvPopulator

hop_length = 512 #the default spacing between frames
n_fft = 255 #number of samples

class IndividualSound():
    def __init__(self,filename,baseLength=None):
        self.filename = filename
        self.baseLength = baseLength

    def getSoundFeatures(self):
        y, sr = librosa.load(self.filename,sr=28000)
        data = np.array([padding(librosa.feature.mfcc(y, n_fft=n_fft,hop_length=hop_length,n_mfcc=128),1,1000)])
        print ("data len: "+str(len(data[0])))
        fixedData = [[]]
        if not self.baseLength:
            self.baseLength = len(data[0][0])
        if len(data[0][0])!= self.baseLength:
            print("current length diverges: "+str(len(data[0][0])))
            for i in data[0]:
                fixedData[0].append(i[0:self.baseLength])
            print("After Fixing:\n Length: "+str(len(fixedData[0])) + " \n length Internal: "+ str(len(fixedData[0][0])))
            return fixedData, self.baseLength
        else:
            return data, self.baseLength

class SoundProcessing:
    def __init__(self):
        if not recordingsFile.exists():
            self.__PopulateCsv()
        self.df = pd.read_csv(recordingsFile)
        self.df.head()
        self.audiodata = {}

    def __PopulateCsv(self):
        csvpopulator = csvPopulator()
        csvpopulator.CreateAnuraCsv()

    def __SaveAudioData(self):
        np.savez_compressed(audioDataFile, soundarray=self.audiodata["soundarray"], speciesarray=self.audiodata["speciesarray"])
            

    def ProcessSound(self):
        self.audiodata["soundarray"], self.audiodata["speciesarray"] = get_features(self.df) 
        self.__SaveAudioData()


def get_features(df_in):
    features=[] #list to save features
    labels=[] #list to save labels
    baseLength = None
    for index in range(0,len(df_in)):      
        filename = df_in.iloc[index]['audio']
        #save labels
        species_id = np.array(df_in.iloc[index]['species'])
        #load the file 
        data, baseLength = IndividualSound(filename,baseLength).getSoundFeatures()
        features.append(data)
        labels.append(species_id)
        print("Index: "+str(index) + " of "+ str(len(df_in)))
    output=np.concatenate(features,axis=0)
    return(np.array(output), labels)


def padding(array, xx, yy):
    """
    :param array: numpy array
    :param xx: desired height
    :param yy: desirex width
    :return: padded array
    """

    print("Inputs: xx: "+str(xx)+" yy: "+str(yy))
    h = array.shape[0]
    w = array.shape[1]

    print("Array: h: "+str(h)+" w: "+str(w))

    a = np.absolute(xx - h) // 2
    aa = np.absolute(xx - a - h)

    print("A: a: "+str(a)+" w: "+str(aa))

    b = np.absolute(yy - w) // 2
    bb = np.absolute(yy - b - w)

    print("B: b: "+str(b)+" w: "+str(bb))

    print("Array: length: "+str(len(array)) + " length lv1: "+str(len(array[0])))

    return np.pad(array, pad_width=((0, 0), (b, bb)), mode='constant')

if __name__ == "__main__":
    soundProcessor = SoundProcessing()
    soundProcessor.ProcessSound()