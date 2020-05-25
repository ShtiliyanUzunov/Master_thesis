from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import sys

class EmotionModel:
    def __init__(self, data):
        self._data = data
        self.buildModel()


    def buildModel(self):
        x, y = self._extractEmotionFeatureData()

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

        self._clf = SVC(C=100, kernel='linear')
        self._clf.fit(x_train, y_train)
        
        #self._clf = LinearSVC()
        #self._clf.fit(x_train, y_train)

        self._trainScore = self._clf.score(x_train, y_train)
        self._testScore = self._clf.score(x_test, y_test)
        print("Training score {}. Test score {}.".format(self._trainScore, self._testScore))

        #TODO: Refactor this
        def showNormFeature (index):
            print(len(x))
            x1 = x[index][0::2]
            y1 = x[index][1::2]
            print(max(x1))
            print(min(x1))
            print(max(y1))
            print(min(y1))
            
            plt.scatter(x1,y1)
            plt.gca().invert_yaxis()
            plt.show()
            
        #showNormFeature(2)

    def predict(self, session):
        peakPhoto = session.getPeakPhoto()
        return self.predict_by_landmarks(peakPhoto.landmarks)

    def predict_by_landmarks(self, landmarks):
        flat_landmarks = []
        for landmark in landmarks:
            flat_landmarks.append(landmark[0])
            flat_landmarks.append(landmark[1])

        flat_landmarks = [flat_landmarks]
        flat_landmarks = self._normalizeFeatures(flat_landmarks)
        return self._clf.predict(flat_landmarks)[0]

    def getScores(self):
        return {
            "Train": self._trainScore,
            "Test": self._testScore
        }
    
    def _extractEmotionFeatureData(self):
        features = []
        emotionLabels = []
        
        def extract(subject, session):
            oSession = self._data.subjects[subject].sessions[session]
            if oSession.emotion is not None:
                last_n_photos = oSession.get_last_n_photos(2)
                for peakPhoto in last_n_photos:
                    flatLandmarks = []
                    for landmark in peakPhoto.landmarks:
                        flatLandmarks.append(landmark[0])
                        flatLandmarks.append(landmark[1])
                    emotionLabels.append(oSession.emotion)
                    features.append(flatLandmarks)
    
        self._data.visit_all_sessions(extract)
    
        return self._normalizeFeatures(features), emotionLabels


    def _normalizeFeatures(self, features):
        for feature in features:
            lowestX = sys.maxsize
            lowestY = sys.maxsize
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