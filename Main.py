from Data import Data
from EmotionModel import EmotionModel
import matplotlib

matplotlib.rcParams.update({'font.size': 8})

data = Data()
#emotionModel = EmotionModel(data)
#emotionModel.printScores()

data.subjects["S045"].sessions["002"].getPeakPhoto().show()
#data.printData()
#print(data.subjects["S132"].sessions["003"].photos[3].landmarks)
#data.subjects["S132"].sessions["003"].show()