from DataLoader import DataLoader
import matplotlib

matplotlib.rcParams.update({'font.size': 8})

dl = DataLoader()
dl.subjects["S045"].sessions["002"].getPeakPhoto().show()
dl.printData()
print(dl.subjects["S132"].sessions["003"].photos[3].landmarks)
dl.subjects["S132"].sessions["003"].show()