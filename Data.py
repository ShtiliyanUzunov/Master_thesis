import configparser
import os

from pth3.DataModel.FACSLabel import FACSLabel
from pth3.DataModel.PhotoSession import PhotoSession
from pth3.DataModel.Subject import Subject

from DataModel.Photo import Photo

cp = configparser.ConfigParser()
cp.read(os.path.join(os.path.dirname(__file__), "Config.ini"))

class Data:
    Images = cp.get("ImagePaths", "Images")
    FACS = cp.get("ImagePaths", "FACS")
    Landmarks = cp.get("ImagePaths", "Landmarks")
    Emotions =  cp.get("ImagePaths", "Emotions")
    
    def __init__(self):
        self.loadSubjects()
        self.visitAllSessions(self.loadPhotos)
        self.visitAllSessions(self.loadFACS)
        self.visitAllSessions(self.loadEmotions)
    
    def loadSubjects(self):
        self.subjects = {}
        directory = Data.Images
        
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

    def visitAllSessions(self, callback):
        for subject in self.subjects:
            sessions = self.subjects[subject].sessions
            for session in sessions:
                callback(subject, session)
       
    def loadPhotos(self, subject, session):
        photoFolder = os.path.join(Data.Images, subject, session)
        landmarkFolder = os.path.join(Data.Landmarks, subject, session)
        for photo in os.listdir(photoFolder):
            if photo.endswith(".DS_Store"):
                continue

            landmark = photo.replace('.png', '_landmarks.txt')
            landmarkPath = os.path.join(landmarkFolder, landmark)
            landmarks = []
            f = open(landmarkPath)
            lines = f.readlines()
            for line in lines:
                if line == "\n":
                    continue
                strLm = line.strip().split("   ")
                arrLm = [float(strLm[0]), float(strLm[1])]
                landmarks.append(arrLm)
            f.close()

            photoPath = os.path.join(photoFolder, photo)
            oPhoto = Photo(photo, None, landmarks, photoPath)
            self.subjects[subject].sessions[session].photos.append(oPhoto)
        return
      
    def loadFACS(self, subject, session):
        sessionPath = os.path.join(Data.FACS, subject, session)
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
        emotionFolder = os.path.join(Data.Emotions, subject, session)
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
            print(subjectName)
            sessions = self.subjects[subjectName].sessions
            for sessionName in sessions:
                session = sessions[sessionName]
                print("\t" + sessionName)
                print("\t\t emotion: %s"%(repr(session.emotion)))
                print("\t\t facsCode: %s intensity: %s"%(session.facs.code, session.facs.intensity))
        return