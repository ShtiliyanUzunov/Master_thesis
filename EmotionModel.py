from Data import Data
from sklearn.svm import LinearSVC
#from sklearn.svm import NuSVC
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
import sys

class EmotionModel:
    def __init__(self, data):
        self._data = data
        self.buildModel()
        
    def setData(self, data):
        self._data = data
        
    def buildModel(self):
        x = self._aggregateEmotionFeatureData()
        y = self._aggregateEmotionClasses()
        
        #TODO: Is SVM the best model ?
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5)
        
        self._clf = SVC(kernel='linear')
        self._clf.fit(x_train, y_train)
        
        #self._clf = LinearSVC()
        #self._clf.fit(x_train, y_train)

        self._trainScore = self._clf.score(x_train, y_train)
        self._testScore = self._clf.score(x_test, y_test)

        #TODO: Refactor this
        def showNormFeature (index):
            print len(x)
            x1 = x[index][0::2]
            y1 = x[index][1::2]
            print max(x1)
            print min(x1)
            print max(y1)
            print min(y1)
            
            plt.scatter(x1,y1)
            plt.gca().invert_yaxis()
            plt.show()
            
        #showNormFeature(2)
    
    def printScores(self):
        print "Training: " + `self._trainScore`
        print "Test: " + `self._testScore`
    
    def _aggregateEmotionFeatureData(self):
        features = []
        
        def aggregate(subject, session):
            oSession = self._data.subjects[subject].sessions[session]
            if oSession.emotion is not None:
                peakPhoto = oSession.getPeakPhoto()
                flatLandmarks = []
                for landmark in peakPhoto.landmarks:
                    flatLandmarks.append(landmark[0])
                    flatLandmarks.append(landmark[1])
                features.append(flatLandmarks)
    
        self._data.visitAllSessions(aggregate)
    
        return self._normalizeFeatures(features)

    def _aggregateEmotionClasses(self):
        labels = []
        
        def aggregate(subject, session):
            oSession = self._data.subjects[subject].sessions[session]
            if oSession.emotion is not None:
                labels.append(oSession.emotion)
        
        self._data.visitAllSessions(aggregate)
        return labels
    
    #TODO: Is this the best normalization? 
    def _normalizeFeatures(self, features):
        for feature in features:
            lowestX = sys.maxint
            lowestY = sys.maxint
            highestX = 0
            highestY = 0
            for x in feature[0::2]:
                if x < lowestX:
                    lowestX = x
                if x > highestX:
                    highestX = x
                    
            for y in feature[1::2]:
                if y < lowestY:
                    lowestY = y
                if y > highestY:
                    highestY = y
            
            xDiff = highestX - lowestX
            yDiff = highestY - lowestY
            
            feature[0::2] = [ 400*(x1 - lowestX) / xDiff for x1 in feature[0::2] ]
            feature[1::2] = [ 300*(y1 - lowestY) / yDiff for y1 in feature[1::2] ]
            
        return features