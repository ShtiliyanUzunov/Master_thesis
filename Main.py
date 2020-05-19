import configparser
import os

from CommandLineInteface import CommandLineInterface
from DataLoader import DataLoader
from EmotionModel import EmotionModel

cp = configparser.ConfigParser()
cp.read(os.path.join(os.path.dirname(__file__), "resources/Config.ini"))


data = DataLoader()

print("Creating emotion model...")
emotionModel = EmotionModel(data)

cli = CommandLineInterface(data, emotionModel)
cli.start()

#data.subjects["S045"].sessions["002"].getPeakPhoto().show()
#data.subjects["S132"].sessions["003"].show()