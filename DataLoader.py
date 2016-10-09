import os

class Photo:
    def __init__(self):
        self.data = None
        self.landmarks = None

class PhotoSession:
    def __init__(self, name):
        self.name = name
        self.emotion = None
        self.facs = None
        self.photos = None

class Subject:
    def __init__(self, name):
        self.name = name
        self.sessions = {}

class DataLoader:
    Images = "D:\Programming\Cohn-Kanade\cohn-kanade-images"
    FACS = "D:\Programming\Cohn-Kanade\Emotion"
    Landmarks = "D:\Programming\Cohn-Kanade\FACS"
    Emotions = "D:\Programming\Cohn-Kanade\Landmarks"
    
    def __init__(self):
        self.loadSubjects()
        
        self.loadImages()
        self.loadFACS()
        self.loadLandmarks()
        self.loadEmotions()
    
    def loadSubjects(self):
        self.subjects = {}
        directory = DataLoader.Images
        
        subjects = os.listdir(directory)
        for name in subjects:
            if name in self.subjects:
                continue
            oSubject = Subject(name)
            self.subjects[name] = oSubject
            subjectDir = os.path.join(directory, name)
            sessions = os.listdir(subjectDir)
            for sessionName in sessions:
                oSubject.sessions[sessionName] = PhotoSession(sessionName)
        return
    
    
    def loadImages(self):
        return
    
    def loadFACS(self):
        return
    
    def loadLandmarks(self):
        return
    
    def loadEmotions(self):
        return
    
dl = DataLoader()
print dl.subjects
    