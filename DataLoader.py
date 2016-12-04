from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib
import os

photoCounter = 0

matplotlib.rcParams.update({'font.size': 8})

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

class PhotoSession:
    def __init__(self, name):
        self.name = name
        self.emotion = None
        self.facs = None
        self.photos = []
        
    def show(self):
        for photo in self.photos:
            photo.show()
        
    def loadSession(self):
        for photo in self.photos:
            photo.loadData()
            
    def unloadSession(self):
        for photo in self.photos:
            photo.unloadData()
            
    def getPeakPhoto(self):
        return self.photos[len(self.photos) - 1]

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
            photoCounter = photoCounter+1
            oPhoto = Photo(photo, None, landmarks, photoPath)
            self.subjects[subject].sessions[session].photos.append(oPhoto)
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

#dl = DataLoader()
#dl.subjects["S045"].sessions["002"].getPeakPhoto().show()
#dl.printData()
#print(dl.subjects["S132"].sessions["003"].photos[3].landmarks)
#dl.subjects["S132"].sessions["003"].show()

