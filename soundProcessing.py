import pathlib
import time
import pandas as pd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import warnings
warnings.filterwarnings('ignore')

from paths import *
from csvPopulator import csvPopulator

hop_length = 512 #the default spacing between frames
n_fft = 255 #number of samples
max_size = 1000

class IndividualSound():
    def __init__(self,filename,baseLength=None, outputFolder=None):
        self.filename = filename
        self.saveFile = outputPath.joinpath(self.filename.split("/")[-1].replace(".mp3",".npz")).absolute()
        self.baseLength = baseLength
        self.outputFolder = pathlib.Path(outputFolder) if outputFolder else None
        self.sr = None
        self.features = None
        self.image = []
        if not self.saveFile.exists():
            self.extractSoundFeatures()
            self.extractAllFeaturesImages()
        else:
            self.Reload()

    def Save(self):
        filename = self.saveFile
        np.savez_compressed(filename,\
             features=self.features,\
                 images=self.image)
        return filename
    
    def Reload(self):
        audioFeatures = np.load(self.saveFile)
        self.features = audioFeatures["features"]
        self.image = audioFeatures["images"]

    def extractSoundFeatures(self):
        y, self.sr = librosa.load(self.filename,sr=28000)
        data = np.array([padding(librosa.feature.mfcc(y, n_fft=n_fft,hop_length=hop_length,n_mfcc=128),1,1000)])
        print ("data len: "+str(len(data[0])))
        if not self.baseLength:
            self.baseLength = len(data[0][0])
        self.features  = fixLength(data, self.baseLength)
        print("Features: "+str(len(self.features)))
            
    
    def extractAllFeaturesImages(self):
        max_size=1000 #my max audio file feature width
        y, self.sr = librosa.load(self.filename,sr=28000)
        stft = padding(np.abs(librosa.stft(y, n_fft=255, hop_length= 512)), 128, max_size)
        MFCCs = padding(librosa.feature.mfcc(y, n_fft=n_fft, hop_length=hop_length,n_mfcc=128),128,max_size)
        spec_centroid = librosa.feature.spectral_centroid(y=y, sr=self.sr)
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=self.sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=self.sr)
        #Now the padding part
        image = np.array(fixLength([padding(normalize(spec_bw),1, max_size)], self.baseLength)).reshape(1,max_size)
        image = np.append(image,fixLength([padding(normalize(spec_centroid),1, max_size)],self.baseLength)[0], axis=0) 
        #repeat the padded spec_bw,spec_centroid and chroma stft until they are stft and MFCC-sized
        for i in range(0,9):
            image = np.append(image, fixLength([padding(normalize(spec_bw),1, max_size)],self.baseLength)[0], axis=0)
            image = np.append(image, fixLength([padding(normalize(spec_centroid),1, max_size)],self.baseLength)[0], axis=0)
            image = np.append(image, fixLength([padding(normalize(chroma_stft),12, max_size)],self.baseLength)[0], axis=0)
        image=np.dstack((image,np.abs(fixLength([stft],self.baseLength)[0])))
        image=np.dstack((image,fixLength([MFCCs],self.baseLength)[0]))
        self.image = image
        print("Imagens: "+str(len(self.image)))
    
    def getSoundFeatures(self):
        return self.features, self.baseLength
    
    def getSoundImages(self):
        return self.image, self.baseLength

    def Waveplot(self):
        y,sr=librosa.load(self.filename)
        librosa.display.waveplot(y,sr=sr, x_axis='time', color='purple',offset=0.0)

    def PlotMFCC(self):
        y, self.sr = librosa.load(self.filename,sr=28000)
        fig, ax = plt.subplots(figsize=(20,7))
        MFCCs = np.absolute(librosa.feature.mfcc(y, n_fft=n_fft, hop_length=hop_length,n_mfcc=128))
        librosa.display.specshow(MFCCs,sr=self.sr, cmap='jet',hop_length=hop_length, x_axis='time', y_axis='mel')
        ax.set_xlabel('Tempo', fontsize=15)
        ax.set_title('Espectrograma de Mel (MFCC)', size=20)
        plt.colorbar().set_label('dB', rotation=270)
        plt.savefig(outputPath.joinpath(self.filename.split("/")[-1].replace(".mp3","-MFCC.png")).absolute())
        if self.outputFolder:
            plt.savefig(self.outputFolder.joinpath(self.filename.split("/")[-1].replace(".mp3","-MFCC.png")).absolute())
        time.sleep(0.5)

    def PlotSTFT(self):
        y, self.sr = librosa.load(self.filename,sr=28000)
        stft = librosa.stft(y, n_fft=255, hop_length= 512)
        fig, ax = plt.subplots(figsize=(20,7))
        librosa.display.specshow(stft,sr=self.sr, cmap='jet',hop_length=hop_length, x_axis='time', y_axis='hz')
        ax.set_ylabel('Hz', fontsize=15)
        ax.set_xlabel('Tempo', fontsize=15)
        ax.set_title('Transformada de Fourier de Curto Prazo (STFT)', size=20)
        plt.colorbar().set_label('dB', rotation=270)
        plt.savefig(outputPath.joinpath(self.filename.split("/")[-1].replace(".mp3","-STFT.png")).absolute())
        if self.outputFolder:
            plt.savefig(self.outputFolder.joinpath(self.filename.split("/")[-1].replace(".mp3","-STFT.png")).absolute())
        time.sleep(0.5)

    def PlotCentroid(self):
        y, self.sr = librosa.load(self.filename,sr=28000)
        spec_centroid = librosa.feature.spectral_centroid(y=y, sr=self.sr)
        times = librosa.times_like(spec_centroid)
        fig, ax = plt.subplots(figsize=(20,7))
        ax.semilogy(times, spec_centroid[0])
        ax.set(ylabel='Hz', xticks=[], xlim=[times.min(), times.max()])
        ax.set_xlabel('Tempo', fontsize=15)
        ax.set_title('Centróide Espectral (SC)', size=20)
        plt.savefig(outputPath.joinpath(self.filename.split("/")[-1].replace(".mp3","-SC.png")).absolute())
        if self.outputFolder:
            plt.savefig(self.outputFolder.joinpath(self.filename.split("/")[-1].replace(".mp3","-SC.png")).absolute())
        time.sleep(0.5)
    
    def PlotChromaSTFT(self):
        y, self.sr = librosa.load(self.filename,sr=28000)
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=self.sr)
        fig, ax = plt.subplots(figsize=(20,7))
        librosa.display.specshow(chroma_stft,sr=self.sr, cmap='jet',hop_length=hop_length, x_axis='time', y_axis='hz')
        ax.set_xlabel('Tempo', fontsize=15)
        ax.set_title('Cromagrama(Chroma-STFT)', size=20)
        plt.colorbar()
        plt.savefig(outputPath.joinpath(self.filename.split("/")[-1].replace(".mp3","-CSTFT.png")).absolute())
        if self.outputFolder:
            plt.savefig(self.outputFolder.joinpath(self.filename.split("/")[-1].replace(".mp3","-CSTFT.png")).absolute())
        time.sleep(0.5)
    
    def PlotSBW(self):
        y, self.sr = librosa.load(self.filename,sr=28000)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=self.sr)
        fig, ax = plt.subplots(figsize=(20,7))
        times = librosa.times_like(spec_bw)
        ax.semilogy(times, spec_bw[0])
        ax.set(ylabel='Hz', xticks=[], xlim=[times.min(), times.max()])
        ax.set_xlabel('Tempo', fontsize=15)
        ax.set_title('Largura de Banda Espectral (SBW)', size=20)
        plt.savefig(outputPath.joinpath(self.filename.split("/")[-1].replace(".mp3","-SBW.png")).absolute())
        print(self.outputFolder)
        if self.outputFolder:
            plt.savefig(self.outputFolder.joinpath(self.filename.split("/")[-1].replace(".mp3","-SBW.png")).absolute())
        time.sleep(0.5)

    def Plot(self):
        self.PlotMFCC()
        self.PlotCentroid()
        self.PlotSTFT()
        self.PlotChromaSTFT()
        self.PlotSBW()


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
        np.savez_compressed(audioDataFile,\
             soundarray=self.audiodata["soundarray"],\
                 soundimages=self.audiodata["soundimages"],\
                 speciesarray=self.audiodata["labels"]["species"],\
                     familiesarray=self.audiodata["labels"]["family"],\
                         gendersarray=self.audiodata["labels"]["gender"])
            
    
    def ProcessSound(self):
        self.ExtractSoundFeatures()
        self.GetSoundFeatures() 

    def GetSoundFeatures(self):
        images = []
        features=[]
        labels={
        "family":[],
        "gender": [],
        "species": []
        }
        for index in range(0,len(self.df)):      
            filename = self.df.iloc[index]['audio']
            #save labels
            species_id = np.array(self.df.iloc[index]['species'])
            family_id = np.array(self.df.iloc[index]['family'])
            gender_id = np.array(self.df.iloc[index]['gender'])
            #load the file 
            individual =  IndividualSound(filename,1000)
            imageData, baseLength = individual.getSoundImages()
            images.append([imageData])
            soundData, baseLength = individual.getSoundFeatures()
            features.append(soundData)
            labels["species"].append(species_id)
            labels["family"].append(family_id)
            labels["gender"].append(gender_id)
            print("Index: "+str(index) + " of "+ str(len(self.df)))
        outputImages=np.concatenate(images,axis=0)
        outputFeatures = np.concatenate(features,axis=0)
        self.audiodata["soundarray"] = outputFeatures
        print("Número de amostras de áudio: "+str(len(self.audiodata["soundarray"])))
        self.audiodata["soundimages"] = outputImages
        print("Número de amostras de imagens de áudio: "+str(len(self.audiodata["soundimages"])))
        self.audiodata["labels"] = labels
        self.__SaveAudioData()
        return(outputFeatures, outputImages, labels)

    def ExtractSoundFeatures(self):
        baseLength = 1000
        for index in range(0,len(self.df)):      
            filename = self.df.iloc[index]['audio']
            #load the file 
            IndividualSound(filename,baseLength).Save()
            print("Index: "+str(index) + " of "+ str(len(self.df)))
            
        
def fixLength(data, baseLength):
    fixedData = [[]]
    if len(data[0][0])!= baseLength:
        print("current length diverges: "+str(len(data[0][0])))
        for i in data[0]:
            fixedData[0].append(i[0:baseLength])
        print("After Fixing:\n Length: "+str(len(fixedData[0])) + " \n length Internal: "+ str(len(fixedData[0][0])))
        return fixedData
    else:
        return data

def get_features(df_in):
    features=[] #list to save features
    labels={
        "family":[],
        "gender": [],
        "species": []
        }
    baseLength = None
    for index in range(0,len(df_in)):      
        filename = df_in.iloc[index]['audio']
        #save labels
        species_id = np.array(df_in.iloc[index]['species'])
        family_id = np.array(df_in.iloc[index]['family'])
        gender_id = np.array(df_in.iloc[index]['gender'])
        #load the file 
        data, baseLength = IndividualSound(filename,baseLength).getSoundFeatures()
        features.append(data)
        labels["species"].append(species_id)
        labels["family"].append(family_id)
        labels["gender"].append(gender_id)
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