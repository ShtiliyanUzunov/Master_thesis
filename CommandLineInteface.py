'''
Created on Sep 24, 2017

@author: Shtiliyan
'''

from .EmotionMap import EmotionMap
from dask.array.learn import predict

class CommandLineInterface:
        
    def __init__(self, data, emotionModel):
        self._data = data
        self._emotionModel = emotionModel
        self._history = []
        pass

    def printMenu(self):
        print("")
        print("Input options")
        for x in list(self.commandNames.keys()):
            print(" - " + x + " - " + self.commandNames[x]["description"])
        
    def start(self):
        self.printMenu()
        self.listenForCommand()
        
    def listenForCommand(self):
        inp = input("")
        self.processCommand(inp)
        
    def processCommand(self, inp):
        inp = inp.split(" ")
        
        if not inp[0] in self.commandNames:
            if len(inp[0]) > 0:
                print("Command not found")
            self.listenForCommand()
            return
        
        self.commandNames[inp[0]]["execute"](self, inp)
        self.listenForCommand()
    
    def printInvalidCommandUsage(self, syntax):
            print("Invalid command usage")
            print("Correct syntax:")
            print(syntax)
        
    def _subjNotFound(self, subjName):
        if subjName not in self._data.subjects:
            print("Subject not found")
            return True
        return False
    
    def _sessionNotFound(self, subjName, sessionName):
        if sessionName not in self._data.subjects[subjName].sessions:
            print("Session not found")
            return True
        return False
    
    def executePrintScores(self, inp):
        scores = self._emotionModel.getScores()
        print("Train " + str(scores["Train"]))
        print("Test " + str(scores["Test"]))

    def executeExit(self, inp):
        exit()
    
    def executeListSubjects(self, inp):
        for subject in self._data.subjects:
            print(subject)
        
        print("Total subjects count:" + str(len(list(self._data.subjects.keys()))))
    
    def executeHelp(self, inp):
        self.printMenu()
    
    def executeShowSession(self, inp):
        if (len(inp) != 3 or
            not inp[1].startswith("-subj=") or
            not inp[2].startswith("-sess=")):
            self.printInvalidCommandUsage("showsession -subj=<subject name> -sess=<session>")
            return
        
        subjName = inp[1][inp[1].index("=") + 1 : ]
        sessionName = inp[2][inp[2].index("=") + 1 : ]
        
        if (self._subjNotFound(subjName) or 
            self._sessionNotFound(subjName, sessionName)):
            return
        
        self._data.subjects[subjName].sessions[sessionName].show()
    
    def executeListSessions(self, inp):
        if len(inp) != 2 or not inp[1].startswith("-subj="):
            self.printInvalidCommandUsage("listsessions -subj=<subject name>")
            return
        
        subjName = inp[1][inp[1].index("=") + 1 : ]
        if self._subjNotFound(subjName):
            return

        for sessionName in self._data.subjects[subjName].sessions:
            session = self._data.subjects[subjName].sessions[sessionName]
            print("\t" + sessionName)
            print("\t\t emotion: %s, %s" % (str(session.emotion), EmotionMap[str(session.emotion)]))
            print("\t\t facsCode: %s intensity: %s" % (session.facs.code, session.facs.intensity))
            
    def executeGetEmotion(self, inp):
        if (len(inp) != 3 or
            not inp[1].startswith("-subj=") or
            not inp[2].startswith("-sess=")):
            self.printInvalidCommandUsage("getemotion -subj=<subject name> -sess=<session>")
            return
        
        subjName = inp[1][inp[1].index("=") + 1 : ]
        sessionName = inp[2][inp[2].index("=") + 1 : ]
        
        if (self._subjNotFound(subjName) or 
            self._sessionNotFound(subjName, sessionName)):
            return
        
        print(self._data.subjects[subjName].sessions[sessionName].emotion)

    def executePredictEmotion(self, inp):
        if (len(inp) != 3 or
            not inp[1].startswith("-subj=") or
            not inp[2].startswith("-sess=")):
            self.printInvalidCommandUsage("predictemotion -subj=<subject name> -sess=<session>")
            return
        
        subjName = inp[1][inp[1].index("=") + 1 : ]
        sessionName = inp[2][inp[2].index("=") + 1 : ]
        
        if (self._subjNotFound(subjName) or 
            self._sessionNotFound(subjName, sessionName)):
            return

        session = self._data.subjects[subjName].sessions[sessionName]
        emotionNumber =  str(self._emotionModel.predict(session))
        print("Predicted emotion: " + emotionNumber + " " + EmotionMap[emotionNumber])

    def executeShowPeak(self, inp):
        if (len(inp) != 3 or
            not inp[1].startswith("-subj=") or
            not inp[2].startswith("-sess=")):
            self.printInvalidCommandUsage("showpeak -subj=<subject name> -sess=<session>")
            return
        
        subjName = inp[1][inp[1].index("=") + 1 : ]
        sessionName = inp[2][inp[2].index("=") + 1 : ]
        
        if (self._subjNotFound(subjName) or 
            self._sessionNotFound(subjName, sessionName)):
            return
        
        session = self._data.subjects[subjName].sessions[sessionName]
        session.getPeakPhoto().show()
    
    def executeListFalsePredictions(self, inp):
        def predictSession(subject, session):
            oSession = self._data.subjects[subject].sessions[session]
            actualEmotion = oSession.emotion
            predictedEmotion = self._emotionModel.predict(oSession)
            
            if (actualEmotion != predictedEmotion) and (actualEmotion is not None):
                print("Prediction difference: " + subject + " " + session)
        
        self._data.visitAllSessions(predictSession)

CommandLineInterface.commandNames = {
    "printScores": {
        "execute": CommandLineInterface.executePrintScores,
        "description": "Prints the scores of the emotion model after the training"
    },
    "exit": {
        "execute": CommandLineInterface.executeExit,
        "description": "Exits the application"
    },
    "listSubjects": {
        "execute": CommandLineInterface.executeListSubjects,
        "description": "Lists all subjects in the data set"
    },
    "listSessions": {
        "execute": CommandLineInterface.executeListSessions,
        "description": "Lists all sessions for the given subject"
    },
    "help": {
        "execute": CommandLineInterface.executeHelp,
        "description": "Lists all available commands"
    },
    "getEmotion": {
        "execute": CommandLineInterface.executeGetEmotion,
        "description": "Prints the emotion for a given session as described in the dataset"
    },
    "predictEmotion": {
        "execute": CommandLineInterface.executePredictEmotion,
        "description": "Predicts the emotion of the session using the computed emotion model"
    },
    "showSession": {
        "execute": CommandLineInterface.executeShowSession,
        "description": "Shows a sequence of images for the given session"
    },
    "showPeak": {
        "execute": CommandLineInterface.executeShowPeak,
        "description": "Shows the peak image in a given session"
    }, 
    "listFalsePredictions": {
        "execute": CommandLineInterface.executeListFalsePredictions,
        "description": "Lists all the false positive sessions of the model"
    }
}