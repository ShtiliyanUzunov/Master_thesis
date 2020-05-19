import configparser
import os

from data_model.FACS_label import FACS_label
from data_model.photo import Photo
from data_model.photo_session import PhotoSession
from data_model.subject import Subject

cp = configparser.ConfigParser()
cp.read(os.path.join(os.path.dirname(__file__), "resources/Config.ini"))

class DataLoader:
    Images = cp.get("ImagePaths", "Images")
    FACS = cp.get("ImagePaths", "FACS")
    Landmarks = cp.get("ImagePaths", "Landmarks")
    Emotions =  cp.get("ImagePaths", "Emotions")
    
    def __init__(self):
        print("Loading data...")
        self._load_subjects()
        self.visit_all_sessions(self._load_photos)
        self.visit_all_sessions(self._load_FACS)
        self.visit_all_sessions(self._load_emotions)
    
    def _load_subjects(self):
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

    def visit_all_sessions(self, callback):
        for subject in self.subjects:
            sessions = self.subjects[subject].sessions
            for session in sessions:
                callback(subject, session)
       
    def _load_photos(self, subject, session):
        photoFolder = os.path.join(DataLoader.Images, subject, session)
        landmarkFolder = os.path.join(DataLoader.Landmarks, subject, session)
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
      
    def _load_FACS(self, subject, session):
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
                self.subjects[subject].sessions[session].facs = FACS_label(float(facs[0]), float(facs[1]))
            f.close()
        return

    def _load_emotions(self, subject, session):
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
    def _print_data(self):
        for subjectName in self.subjects:
            print(subjectName)
            sessions = self.subjects[subjectName].sessions
            for sessionName in sessions:
                session = sessions[sessionName]
                print("\t" + sessionName)
                print("\t\t emotion: %s"%(repr(session.emotion)))
                print("\t\t facsCode: %s intensity: %s"%(session.facs.code, session.facs.intensity))
        return