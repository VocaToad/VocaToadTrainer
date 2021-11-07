import pathlib

projectPath = pathlib.Path(__file__).parent.absolute()
configPath = projectPath.joinpath("config").absolute()
fnjvAnuraPath = configPath.joinpath("FNJV_Anura.yaml").absolute()
outputPath = projectPath.joinpath("output").absolute()
speciesFile = outputPath.joinpath("species.csv").absolute()
recordingsFile = outputPath.joinpath("recordings.csv").absolute()
audioDataFile = outputPath.joinpath("audioData.npz").absolute()
checkpointPath = projectPath.joinpath("checkpoint").absolute()
modelWeightsFile = checkpointPath.joinpath("vocatoadCheckpoint").absolute()

outputPath.mkdir(exist_ok=True)
checkpointPath.mkdir(exist_ok=True)