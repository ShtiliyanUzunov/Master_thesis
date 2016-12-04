'''
Created on 7.11.2016
@author: Shtiliyan
'''

from DataLoader import DataLoader
from sklearn.svm import LinearSVC
from sklearn.svm import NuSVC
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
import sys

def iterateSessions (fn):
    for subject in dl.subjects:
        for session in dl.subjects[subject].sessions:
            fn(dl.subjects[subject].sessions[session])

def normalizeFeatures(features):
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
        
        feature[0::2] = [ (x1 - lowestX) / xDiff for x1 in feature[0::2] ]
        feature[1::2] = [ (y1 - lowestY) / yDiff for y1 in feature[1::2] ]
        
    return features

def aggregateEmotionFeatureData():
    features = []
    
    def aggregate(session):
        if session.emotion is not None:
            peakPhoto = session.getPeakPhoto()
            flatLandmarks = []
            for landmark in peakPhoto.landmarks:
                flatLandmarks.append(landmark[0])
                flatLandmarks.append(landmark[1])
            features.append(flatLandmarks)

    iterateSessions(aggregate)

    return normalizeFeatures(features)

def aggregateEmotionClasses():
    labels = []
    
    def aggregate(session):
        if session.emotion is not None:
            labels.append(session.emotion)
    
    iterateSessions(aggregate)
    return labels
            
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
            
dl = DataLoader()

x = aggregateEmotionFeatureData()
y = aggregateEmotionClasses()

#showNormFeature(35)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)

print len(x_train[0])
clf = SVC(kernel='linear')
clf.fit(x_train, y_train)

clf2 = LinearSVC()
clf2.fit(x_train, y_train)


#for index in range(0, len(x_test)):
    #print clf.predict(x_test[index]), y_test[index]
    
print clf.score(x_train, y_train)
print clf.score(x_test, y_test)

print clf2.score(x_train, y_train)
print clf2.score(x_test, y_test)

#print x_train
#print x_test
