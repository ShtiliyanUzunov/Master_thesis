import configparser
import os

from CommandLineInteface import CommandLineInterface
from Data import Data
from EmotionModel import EmotionModel


cp = configparser.ConfigParser()
cp.read(os.path.join(os.path.dirname(__file__), "Config.ini"))

trainingPath = cp.get("ImagePaths", "TrainingPath")
trainingImages = 1000

print("Loading data...")
data = Data()

print("Preparing traing data...")
#prepareLandmarkTrainingData = PrepareLandmarkTrainingData(data, trainingImages, trainingPath)
#prepareLandmarkTrainingData.prepare()

print("Creating emotion model...")
emotionModel = EmotionModel(data)

cli = CommandLineInterface(data, emotionModel)
cli.start()

#data.subjects["S045"].sessions["002"].getPeakPhoto().show()
#data.subjects["S132"].sessions["003"].show()