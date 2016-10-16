from scipy import ndimage
import matplotlib.pyplot as plt
import os

photoCounter = 0

class Photo:
    def __init__(self, name, data, landmarks):
        self.name = name
        self.data = data
        self.landmarks = landmarks

class PhotoSession:
    def __init__(self, name):
        self.name = name
        self.emotion = None
        self.facs = None
        self.photos = []

class FACSLabel:
    def __init__(self, code, intensity):
        self.code = code
        self.intensity = intensity

class Subject:
    def __init__(self, name):
        self.name = name
        self.sessions = {}

class DataLoader:
    Images = "D:\Programming\Cohn-Kanade\cohn-kanade-images"
    FACS = "D:\Programming\Cohn-Kanade\FACS"
    Landmarks = "D:\Programming\Cohn-Kanade\Landmarks"
    Emotions = "D:\Programming\Cohn-Kanade\Emotion"
    
    def __init__(self):
        self.loadSubjects()
        self.visitAllSessions(self.loadPhotos)
        self.visitAllSessions(self.loadFACS)
        self.visitAllSessions(self.loadEmotions)
    
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
                #.DS_Store are files for the MacOS that store data for the views
                #Those fiels are not needed
                if sessionName.endswith(".DS_Store"):
                    continue
                oSubject.sessions[sessionName] = PhotoSession(sessionName)
        return
       
    def loadPhotos(self, subject, session):
        global photoCounter
        photoFolder = os.path.join(DataLoader.Images, subject, session)
        landmarkFolder = os.path.join(DataLoader.Landmarks, subject, session)
        for photo in os.listdir(photoFolder):
            if photo.endswith(".DS_Store"):
                continue
            photoPath = os.path.join(photoFolder, photo)
            photoCounter += 1
            photoData = ndimage.imread(photoPath, True)
            self.subjects[subject].sessions[session].photos.append(Photo(photo, photoData, None))
        return
    
    def visitAllSessions(self, callback):
        for subject in self.subjects:
            sessions = self.subjects[subject].sessions
            for session in sessions:
                callback(subject, session)
      
    def loadFACS(self, subject, session):
        sessionPath = os.path.join(DataLoader.FACS, subject, session)
        facsList = os.listdir(sessionPath)
        for listItem in facsList:
            facsLiPath = os.path.join(sessionPath, listItem)
            f = open(facsLiPath)
            lines = f.readlines()
            for line in lines:
                if line == "\n":
                    continue
                facs = line.strip().split("   ")
                self.subjects[subject].sessions[session].facs = FACSLabel(float(facs[0]), float(facs[1]))
            f.close()
        return
     
    #TODO: Maybe emotion should equal -1 when we don't have any data ?       
    def loadEmotions(self, subject, session):
        sessions = self.subjects[subject].sessions
        emotionFolder = os.path.join(DataLoader.Emotions, subject, session)
        if not os.path.exists(emotionFolder):
            sessions[session].emotion = None
            return
        
        fileList = os.listdir(emotionFolder)
        if len(fileList) == 0:
            sessions[session].emotion = None
            return
        
        f = open(os.path.join(emotionFolder, fileList[0]))
        sessions[session].emotion = float(f.readline().strip())
        f.close()
    
    #For debug purposes, and to get an idea of how the data looks loaded in memory
    def printData(self):
        for subjectName in self.subjects:
            print subjectName
            sessions = self.subjects[subjectName].sessions
            for sessionName in sessions:
                session = sessions[sessionName]
                print "\t" + sessionName
                print "\t\t emotion: %s"%(repr(session.emotion))
                print "\t\t facsCode: %s intensity: %s"%(session.facs.code, session.facs.intensity)
        return

dl = DataLoader()
dl.printData()
print photoCounter