from Data import Data
from EmotionModel import EmotionModel
import matplotlib
from CommandLineInteface import CommandLineInterface



matplotlib.rcParams.update({'font.size': 8})

print "Loading data..."
data = Data()

print "Creating emotion model..."
emotionModel = EmotionModel(data)

cli = CommandLineInterface(data, emotionModel)
cli.start()

#emotionModel.printScores()

#data.subjects["S045"].sessions["002"].getPeakPhoto().show()
#data.printData()
#print(data.subjects["S132"].sessions["003"].photos[3].landmarks)
#data.subjects["S132"].sessions["003"].show()