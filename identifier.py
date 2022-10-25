import pathlib
import pandas as pd
import numpy as np
from vocatoadSpeciesCNN import VocatoadSpeciesCNN
from soundProcessing import IndividualSound

import paths


class ToadIdentifier:
    def __init__(self):
        self.speciesReference = pd.read_csv(paths.speciesFile)
        self.gendersReference = pd.read_csv(paths.gendersFile)
        self.familiesReference = pd.read_csv(paths.familiesFile)

    def GetSoundImages(self, audio, baseLength=None, outputFolder=None):
        soundData = IndividualSound(audio, baseLength, outputFolder)
        soundData.Plot()

    def IdentifyToadSpecieWithCNN(self, audio, baseLength=None, outputFolder=None):
        self.GetSoundImages(audio, baseLength, outputFolder)
        prediction = VocatoadSpeciesCNN().IdentifySoundImage(audio, baseLength, outputFolder)
        max_value = prediction.max()
        index = np.where(prediction == max_value)[0]
        entry = self.speciesReference.iloc[index].values.tolist()[0]
        
        return {
            "order": entry[1],
            "family": entry[2],
            "gender": entry[3],
            "specie": entry[4],
            "name": entry[5] if not "nan" else "",
            "certainty": "{:.2f}%".format(max_value*100)
        }
        


if __name__ == "__main__":
    testFolder= pathlib.Path("Teste").absolute()
    testFile = testFolder.joinpath("FNJV_0032775.mp3").absolute()
    print(str(testFile))
    identifier = ToadIdentifier()
    print(identifier.IdentifyToadSpecieWithCNN(str(testFile),None, str(testFolder)))
