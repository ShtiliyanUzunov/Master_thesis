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

    def get_last_n_photos(self, n):
        return self.photos[len(self.photos) - n:]