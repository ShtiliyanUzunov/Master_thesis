'''
Created on Sep 24, 2017

@author: Shtiliyan
'''

class CommandLineInterface:
        
    def __init__(self, data, emotionModel):
        self._data = data
        self._emotionModel = emotionModel
        self._history = []
        pass

    def printMenu(self):
        print ""
        print "Input options"
        for x in self.commandNames.keys():
            print " - " + x
        
    def start(self):
        self.printMenu()
        self.listenForCommand()
        
    def listenForCommand(self):
        inp = raw_input("")
        self.processCommand(inp)
        
    def processCommand(self, inp):
        inp = inp.split(" ")
        
        if not inp[0] in self.commandNames:
            if len(inp[0]) > 0:
                print "Command not found"
            self.listenForCommand()
            return
        
        self.commandNames[inp[0]]["execute"](self, inp)
        self.listenForCommand()
    
    def printInvalidCommandUsage(self, syntax):
            print "Invalid command usage"
            print "Correct syntax:"
            print syntax
        
    def _subjNotFound(self, subjName):
        if subjName not in self._data.subjects:
            print "Subject not found"
            return True
        return False
    
    def _sessionNotFound(self, subjName, sessionName):
        if sessionName not in self._data.subjects[subjName].sessions:
            print "Session not found"
            return True
        return False
    
    def executePrintScores(self, inp):
        self._emotionModel.printScores()

    def executeExit(self, inp):
        exit()
    
    def executeListSubjects(self, inp):
        for subject in self._data.subjects:
            print subject
        
        print "Total subjects count:" + str(len(self._data.subjects.keys()))
    
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
        
        for sessionKey in self._data.subjects[subjName].sessions:
            print sessionKey
            
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
        
        print self._data.subjects[subjName].sessions[sessionName].emotion
        
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
        
        print self._data.subjects[subjName].sessions[sessionName].emotion
CommandLineInterface.commandNames = {
        "showscores": {
            "execute": CommandLineInterface.executePrintScores
        },
        "exit": {
            "execute": CommandLineInterface.executeExit
        },
        "listsubjects": {
            "execute": CommandLineInterface.executeListSubjects
        },
        "listsessions": {
            "execute": CommandLineInterface.executeListSessions
        },
        "help": {
            "execute": CommandLineInterface.executeHelp
        },
        "getemotion": {
            "execute": CommandLineInterface.executeGetEmotion
        },
        "predictemotion": {
            "execute": CommandLineInterface.executePredictEmotion
        },
        "showsession": {
            "execute": CommandLineInterface.executeShowSession
        }
    }
    
