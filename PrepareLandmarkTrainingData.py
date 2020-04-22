
'''
    The class takes the data, number of images and the location as a parameter
    and checks if there are 2 folders - Images, and Points, 
    both should have the same number of files inside them - equal to the numberOfImages parameter
'''
import os
import random
import win32security
import ntsecuritycon as con
import getpass
from shutil import copyfile

class PrepareLandmarkTrainingData:
    points = "POINTS"
    images = "IMAGES"
    
    def __init__(self, data, numberOfImages, pathToImages):
        self._data = data
        self._numberOfImages = numberOfImages
        self._pathToImages = pathToImages
        self._imagesPath = os.path.join(self._pathToImages, PrepareLandmarkTrainingData.images)
        self._pointsPath = os.path.join(self._pathToImages, PrepareLandmarkTrainingData.points)
    
    def _fileCountMatch(self):
        if not os.path.exists(self._imagesPath) or not os.path.exists(self._pointsPath):
            return False
        
        lenImages = len([name for name in os.listdir(self._imagesPath) if os.path.isfile(name)])
        lenPoints = len([name for name in os.listdir(self._pointsPath) if os.path.isfile(name)])
        return lenImages == self._numberOfImages and lenPoints == self._numberOfImages
    
    def _cleanDirs(self):
        if os.path.exists(self._imagesPath):        
            os.rmdir(self._imagesPath)
            
        if os.path.exists(self._pointsPath):   
            os.rmdir(self._pointsPath)
        
        os.makedirs(self._imagesPath)
        os.chmod(self._imagesPath, 0o777)
        os.makedirs(self._pointsPath)
        os.chmod(self._pointsPath, 0o777)
    
    def prepare(self):
        if self._fileCountMatch():
            return

        self._cleanDirs()

        allImagePaths = {}
        
        def _visitSession(subject, session):
            oSession = self._data.subjects[subject].sessions[session]
            for photo in oSession.photos:
                allImagePaths[photo.path] = photo.landmarks
            
        self._data.visitAllSessions(_visitSession)
        
        trainImagePaths = {}
        
        for i in range(0, self._numberOfImages):
            imgsLen = len(allImagePaths.items())
            randomImgIndex = random.randint(0, imgsLen - 1)
            randomImgName = allImagePaths.items()[randomImgIndex][0]
            
            trainImagePaths[randomImgName] = allImagePaths[randomImgName]
            del allImagePaths[randomImgName]
            
        for imgPath in trainImagePaths:
            copyfile(imgPath, self._imagesPath)
            imgName = imgPath[imgPath.rfind("\\"):] + ".pts"
            
            pointsFilePath = os.path.join(self._pointsPath, imgName)
            
            f = open(pointsFilePath, 'w')
            f.write(trainImagePaths[imgPath])
            f.close()