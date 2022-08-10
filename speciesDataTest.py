import pandas as pd
import numpy as np

from soundProcessing import SoundProcessing, IndividualSound
from vocatoadSpeciesCNN import VocatoadSpeciesCNN
from paths import *

class speciesDataTest:
    def __init__(self) -> None:
        self.LoadFamiliesCsv()
        self.LoadGendersCsv()
        self.LoadSpeciesCsv()
        self.LoadRecordingsCsv()
        self.AddRecordingCountColumn()
        self.AddAccuracyColumn()
        self.RecordingsBySpecies = {}
        self.genderAccuracySum = {}
        self.familyAccuracySum = {}
        self.vocatoad = VocatoadSpeciesCNN()

    def LoadSpeciesCsv(self):
        self.species = pd.read_csv(speciesFile)
        self.species.head()
    
    def LoadFamiliesCsv(self):
        self.families = pd.read_csv(familiesFile)
        self.families.head()
    
    def LoadGendersCsv(self):
        self.genders = pd.read_csv(gendersFile)
        self.genders.head()
    
    def LoadRecordingsCsv(self):
        self.recordings = pd.read_csv(recordingsFile)
        self.recordings.head()

    def AddRecordingCountColumn(self):
        self.species["recordings"] = 0
        self.families["recordings"] = 0
        self.genders["recordings"] = 0
    
    def AddAccuracyColumn(self):
        self.species["accuracy"] = 0.
        self.families["accuracy"] = 0.
        self.genders["accuracy"] = 0.

    def CountRecordings(self):
        for index in range(0,len(self.recordings)):
            species_id = self.recordings.iloc[index]['species']
            family_id = self.recordings.iloc[index]['family']
            gender_id = self.recordings.iloc[index]['gender']
            filename = self.recordings.iloc[index]['audio']
            self.species.at[species_id, 'recordings']  += 1
            self.families.at[family_id, 'recordings'] += 1
            self.genders.at[gender_id, 'recordings'] += 1
            if self.species.at[species_id, 'recordings'] == 1:
                self.RecordingsBySpecies[species_id] = {
                    "gender": gender_id,
                    "family": family_id,
                    "recordings":[filename]}
            else:
                self.RecordingsBySpecies[species_id]["recordings"].append(filename)
            
            if self.genders.at[gender_id, 'recordings'] == 1:
                self.genderAccuracySum[gender_id] = 0.
            if self.families.at[family_id, 'recordings'] == 1:
                self.familyAccuracySum[family_id] = 0.
            # if index==0:
            #     individual =  IndividualSound(filename,1000)
            #     individual.Plot()
            #     result = self.vocatoad.IdentifySoundImage(filename)
            #     print(result[0][species_id])
            #     print(GetBestResults(result,10))

    def CalculateAccuracy(self):
        self.CalculateAccuracyForSpecies()
        self.CalculateAccuracyForGenders()
        self.CalculateAccuracyForFamilies()
    
    def CalculateAccuracyForSpecies(self):
        print("Calculating Accuracy For Species")
        for species_id, value in self.RecordingsBySpecies.items():
            accuracySum = 0.
            results = self.vocatoad.IdentifySoundImages(value["recordings"])
            bestResults = GetBestResults(results,1)
            for result in bestResults:
                if species_id in result:
                    accuracySum += result[species_id]
            self.species.at[species_id, 'accuracy'] = accuracySum/self.species.at[species_id, 'recordings']
            print("Specie "+str(species_id)+": "+str(self.species.at[species_id, 'accuracy']))
            self.genderAccuracySum[value["gender"]] += accuracySum
            self.familyAccuracySum[value["family"]] += accuracySum
    
    def CalculateAccuracyForGenders(self):
        for gender_id, accuracySum in self.genderAccuracySum.items():
            self.genders.at[gender_id, 'accuracy'] = accuracySum/self.genders.at[gender_id, 'recordings']

    def CalculateAccuracyForFamilies(self):
        for family_id, accuracySum in self.familyAccuracySum.items():
            self.families.at[family_id, 'accuracy'] = accuracySum/self.families.at[family_id, 'recordings']
    
    def CreateSpeciesReport(self):
        self.species.to_csv(speciesReport, index=False)
    
    def CreateGendersReport(self):
        self.genders.to_csv(gendersReport, index=False)
    
    def CreateFamiliesReport(self):
        self.families.to_csv(familiesReport, index=False)

    def CreateReports(self):
        self.CountRecordings()
        self.CalculateAccuracy()
        self.CreateSpeciesReport()
        self.CreateGendersReport()
        self.CreateFamiliesReport()
    
def RemoveNearZeroes(array):
    zeroedArray = []
    for result in array:
        zeroedOptions = []
        for option in result:
            if option < 0.01:
                zeroedOptions.append(0)
            else:
                zeroedOptions.append(option)
        zeroedArray.append(zeroedOptions)
    return zeroedArray

def GetBestResults(array, x=10):
    topArray = []
    for result in array:
        resultDictionary = ConvertResultToDictionary(result)
        topDictionary = GetTopXOptions(resultDictionary, x)
        topArray.append(topDictionary)
    return topArray

def GetTopXOptions(dictionary, x=10):
    topDictionary = {}
    for i in range(0,x):
            max_key, value = GetMostProbableOption(dictionary)
            topDictionary.update({max_key:value})
            dictionary.pop(max_key)
    return topDictionary

        
def ConvertResultToDictionary(result):
    resultDictionary = {}
    for index in range(0,len(result)):
        resultDictionary[index] = result[index]
    return resultDictionary

def GetMostProbableOption(dictionary):
    max_key = max(dictionary, key=dictionary.get)
    return max_key, dictionary[max_key]


if __name__ == "__main__":
    speciesDataTest = speciesDataTest()
    speciesDataTest.CreateReports()
