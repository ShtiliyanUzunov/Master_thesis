from scipy import ndimage
import matplotlib.pyplot as plt
import cv2

class Photo:
    def __init__(self, name, data, landmarks, path):
        self.name = name
        self.data = data
        self.landmarks = landmarks
        self.path = path
    
    def loadData(self):
        if self.data is None:
            self.data = cv2.imread(self.path, False)
        
    def unloadData(self):
        self.data = None
    
    def show(self, showLandmarks = True, overlap_landmarks=None):
        self.loadData()
        plt.imshow(self.data, interpolation='none', cmap='Greys_r')
        for index in range(len(self.landmarks)):
            landmark = self.landmarks[index]
            ax = plt.gcf().gca()
            circle = plt.Circle((landmark[0], landmark[1]), 0.5, color='red')
            ax.add_artist(circle)
            #text = plt.Text(x = landmark[0], y = landmark[1], text=index, color='red')
            #ax.add_artist(text)

        if overlap_landmarks is not None:
            for index in range(len(overlap_landmarks)):
                landmark = overlap_landmarks[index]
                ax = plt.gcf().gca()
                circle = plt.Circle((landmark[0], landmark[1]), 0.5, color='green')
                ax.add_artist(circle)

        plt.show()