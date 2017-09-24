from scipy import ndimage
import matplotlib.pyplot as plt

class Photo:
    def __init__(self, name, data, landmarks, path):
        self.name = name
        self.data = data
        self.landmarks = landmarks
        self.path = path
    
    def loadData(self):
        if self.data == None:
            self.data = ndimage.imread(self.path, False, 'L')
        
    def unloadData(self):
        self.data = None
    
    def show(self, showLandmarks = True):
        self.loadData()
        plt.imshow(self.data, interpolation='none', cmap='Greys_r')
        for index in range(len(self.landmarks)):
            landmark = self.landmarks[index]
            ax = plt.gcf().gca()
            circle = plt.Circle((landmark[0], landmark[1]), 1, color='red')
            ax.add_artist(circle)
            #text = plt.Text(x = landmark[0], y = landmark[1], text=index, color='red')
            #ax.add_artist(text)
        plt.show()