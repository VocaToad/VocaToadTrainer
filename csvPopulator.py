import yaml
import logging
import pathlib
import json
import csv
from paths import *

class csvPopulator:
    def __init__(self):
        self.__loadConfigFile()
        self.recordings = []
        self.species = []
        self.families = []
        self.genders = []
        self.speciesNextId = 0
        
        
    def __loadConfigFile(self):
        logging.debug("loadConfigFile started")
        try:
            with open(fnjvAnuraPath, "r") as fnjvAnuraFile:
                self.fnjvAnura = yaml.load(fnjvAnuraFile)
            logging.info("Loaded fnjvAnura Configuration: "+fnjvAnuraPath._str)
        except:
            logging.critical("Error loading Database Configuration:"+fnjvAnuraPath._str)
            logging.exception('')
        logging.debug("loadConfigFile finished")
    
    def __LoadFromFnjvAnuraFile(self):
        self.fnjvAnuraFilePath = pathlib.Path(self.fnjvAnura["folder"]).absolute()
        referenceFile = self.fnjvAnuraFilePath.joinpath(self.fnjvAnura["mainFile"]).absolute()
        with open(str(referenceFile)) as f:
            mainData = json.load(f)
        return mainData
        
    def __PopulateFromFnjvAnuraFile(self):
        mainData = self.__LoadFromFnjvAnuraFile()
        successfulRecordings = 0
        failedRecordings = 0
        for recording in mainData["animals"]:
            if self.__PopulateRecording(recording):
                successfulRecordings += 1
            else:
                failedRecordings += 1
        logging.info("Recordings added to database: "+str(successfulRecordings))
        if failedRecordings > 0:
            logging.warning(" Failed Recordings: "+str(failedRecordings))

    def __LoadRecordingFile(self,filename):
        recordingFile = self.fnjvAnuraFilePath.joinpath(filename).absolute()
        with open(str(recordingFile)) as f:
            recordingData = json.load(f)
        return recordingData

    def __PopulateRecording(self,recording):
        try:
            recordingData = self.__LoadRecordingFile(recording["filename"])
            animal = self.__SpeciesExist(recordingData)
            if not animal:
                if not self.__AddSpecies(recordingData):
                    return False
            
            if self.__RecordingExist(recordingData):
                return True
            
            if not self.__AddRecording(recordingData):
                return False
            return True
        except:
            logging.exception('')
            return False

    def __SpeciesExist(self, recordingData):
        try:
            for specie in self.species:
                if specie["gender"] == recordingData["gender"] and \
                    specie["species"] == recordingData["species"]:
                    return specie
            return None
        except:
            logging.exception('')
            return None
    
    def __FamilyExist(self, recordingData):
        try:
            for family in self.families:
                if family["family"] == recordingData["family"]:
                    return family
            return None
        except:
            logging.exception('')
            return None
    
    def __GenderExist(self, recordingData):
        try:
            for gender in self.genders:
                if gender["gender"] == recordingData["gender"]:
                    return gender
            return None
        except:
            logging.exception('')
            return None

    def __AddSpecies(self,recordingData):
        try:
            if not self.__FamilyExist(recordingData):
                if not self.__AddFamily(recordingData):
                    return False
            if not self.__GenderExist(recordingData):
                if not self.__AddGender(recordingData):
                    return False
            self.species.append({"id": self.speciesNextId,\
                "class": recordingData["class"],\
                "family": recordingData["family"],\
                    "gender": recordingData["gender"],\
                        "species": recordingData["species"],\
                            "popularName": recordingData["popularName"]})
            self.speciesNextId += 1
            return True
        except:
            logging.critical("Unable to Create Species: "+recordingData["gender"]+" "+recordingData["species"])
            logging.exception('')
            return False

    def __AddFamily(self,recordingData):
        try:
            self.families.append(
                {
                "id": len(self.families),
                "family": recordingData["family"]
                }
                )
            return True
        except:
            logging.critical("Unable to Create Family: "+recordingData["family"])
            logging.exception('')
            return False

    def __AddGender(self,recordingData):
        try:
            self.genders.append(
                {
                    "id": len(self.genders),\
                        "gender": recordingData["gender"]
                        }
                        )
            return True
        except:
            logging.critical("Unable to Create Gender: "+recordingData["gender"])
            logging.exception('')
            return False
        
    def __RecordingExist(self,recordingData):
        try:
            for recording in self.recordings:
                if recording["id"] == recordingData["number"]:
                    return recording
            return None
        except:
            logging.exception('')
            return None

    def __AddRecording(self,recordingData):
        try:
            animal = self.__SpeciesExist(recordingData)
            family = self.__FamilyExist(recordingData)
            gender = self.__GenderExist(recordingData)
            self.recordings.append({
                "id": recordingData["number"],
                "family": family["id"],
                "gender": gender["id"],
                "species": animal["id"],
                "audio": self.fnjvAnuraFilePath.joinpath(recordingData["individualData"]["audio"]["audio"]["file"].split("/")[-1]).absolute()
            })
            return True
        except:
            logging.critical("Unable to Create Recording: "+recordingData["number"]+" "+recordingData["family"]+" "+recordingData["gender"]+" "+recordingData["species"])
            logging.exception('')
            return False
    
    def CreateAnuraCsv(self):
        self.__PopulateFromFnjvAnuraFile()
        self.__CreateSpeciesCsv()
        self.__CreateFamilyCsv()
        self.__CreateGenderCsv()
        self.__CreateRecordingsCsv()
    
    def __CreateSpeciesCsv(self):
        with open(speciesFile, 'w', encoding='utf8', newline='') as output_file:
            fc = csv.DictWriter(output_file, fieldnames=self.species[0].keys(),)
            fc.writeheader()
            fc.writerows(self.species)

    def __CreateFamilyCsv(self):
        with open(familiesFile, 'w', encoding='utf8', newline='') as output_file:
            fc = csv.DictWriter(output_file, fieldnames=self.families[0].keys(),)
            fc.writeheader()
            fc.writerows(self.families)

    def __CreateGenderCsv(self):
        with open(gendersFile, 'w', encoding='utf8', newline='') as output_file:
            fc = csv.DictWriter(output_file, fieldnames=self.genders[0].keys(),)
            fc.writeheader()
            fc.writerows(self.genders)
    
    def __CreateRecordingsCsv(self):
        with open(recordingsFile, 'w', encoding='utf8', newline='') as output_file:
            fc = csv.DictWriter(output_file, fieldnames=self.recordings[0].keys(),)
            fc.writeheader()
            fc.writerows(self.recordings)

if __name__ == "__main__":
    csvpopulator = csvPopulator()
    csvpopulator.CreateAnuraCsv()