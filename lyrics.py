from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
import nltk
from scipy.special import erf
from math import tanh, atanh
import os.path
import csv
class word(object):
    #data object that contains the emotion vector, and part of speach for a lyrical word
    def __init__(self, word ='' , valence = 0, arousal = 0, POS = 'FW'):
        self.word = word
        self.valence = valence
        self.arousal = arousal
        self.POS = POS

def loadSong(lyricFile, dictionary={}):
    #This takes an imput file and loads the song as an aray of word arrays (each being a line)
    #The parsing will lemmatize get parts of speach and as needed stem the words using snow stemmer
    newSong = []#song()
    songPhrase =[];
    LyricFile = open(lyricFile, 'r')
    RawLyricLines = LyricFile.readlines()
    lemmatizer = WordNetLemmatizer()
    snowStem = SnowballStemmer("english")
    for r in RawLyricLines: # parses line by line and put these into seperate phrases so song can be analyzed phrase by phrase

#        print("newLyricLine")
        newPhrase = word_tokenize(r)
        POS = nltk.pos_tag(newPhrase)
        for p in POS:
            w = p[0];
            currentWord = lemmatizer.lemmatize(w).lower()
            if not dictionary.has_key(currentWord):
                currentWord = snowStem.stem(currentWord)
            if dictionary.has_key(currentWord):
                dictWord = dictionary[currentWord]
                valence = dictWord[0]
                arousal = dictWord[1]
                newWord = word(currentWord,valence,arousal,p[1])
            else:
                newWord = word(currentWord,0,0,p[1])
            songPhrase.append(newWord)
        if len(songPhrase)>0:
            newSong.append(songPhrase)
        songPhrase =[]
    return newSong
def loadTrainingSong(lyricFile, dictionary={}):
    #This takes an imput file and loads the song as an aray of word arrays (each being a line)
    #The parsing will lemmatize get parts of speach and as needed stem the words using snow stemmer
    newSong = []#song()
    LyricFile = open(lyricFile, 'r')
    RawLyricLines = LyricFile.readlines()
    lemmatizer = WordNetLemmatizer()
    snowStem = SnowballStemmer("english")
    for r in RawLyricLines: # parses line by line and put these into seperate phrases so song can be analyzed phrase by phrase

#        print("newLyricLine")
        newPhrase = word_tokenize(r)
        POS = nltk.pos_tag(newPhrase)
        for p in POS:
            w = p[0]
            currentWord = lemmatizer.lemmatize(w).lower()
            if not dictionary.has_key(currentWord):
                currentWord = snowStem.stem(currentWord)
            if dictionary.has_key(currentWord):
                dictWord = dictionary[currentWord]
                valence = dictWord[0]
                arousal = dictWord[1]
                newWord = word(currentWord,valence,arousal,p[1])
            else:
                newWord = word(currentWord,0,0,p[1])
            newSong.append(newWord)
    return newSong
def loadTrainingData(vectorFile = 'lyrics/lyricVectors.csv', dictionary ={}):
    songs = []
    TrainingVectorFile = open(vectorFile, 'r')
    reader = csv.reader(TrainingVectorFile)
    vectors = []
    for r in reader:
        lyricFile='lyrics/' + str(r[0]) +'.txt'
        if os.path.isfile(lyricFile):
            songs.append(loadTrainingSong(lyricFile,dictionary))
            print(lyricFile + ' loaded')
            vector = [float(r[1]), float(r[2])]
            vectors.append(vector)
    return [songs, vectors]


class emotionState (object):
    #container that has stores the current state and prvious states to be used by genetic algorethem
    def __init__(self, currentPhraseLength =0,currentVector = [0,0], currentPOS ='FW',nextWord = word(), previousVector = [0,0],previousPOS = 'FW', previous2Vector = [0,0], previous2POS = 'FW'):
        self.currentPOS = currentPOS
        self.previousPOS = previousPOS
        self.previous2POS = previous2POS
        self.currentVector = currentVector
        self.previousVector = previousVector
        self.previous2Vector = previous2Vector
        self.currentPhraseLength = currentPhraseLength
        self.nextWord = nextWord
    def initNewES(self,  previousES, w = word()):
        #this lets you initilize an emotion state with a previous state and the current input word
        self.currentPOS = w.POS
        self.previousPOS = previousES.currentPOS
        self.previous2POS = previousES.previousPOS
        self.previousVector = previousES.currentVector
        self.previous2Vector = previousES.previousVector
        self.currentPhraseLength = previousES.currentPhraseLength+1
        self.nextWord = w

    def setCurrentVector(self,CV):
        self.currentVector = CV


class emotionStateMechine(object):
    #and emotion state mechine handles finding what the next emotion state preidction is base on input vlaues and previous staes
    def __init__(self, type='naive'):
        self.type = type
    def getNextEState(self,newWord = word(),  currentEState = emotionState()):
        #function to get next emaotino state
        newES = emotionState()
        newES.initNewES(currentEState, newWord)
        nextV = [currentEState.previousVector[0] + newWord.valence,currentEState.previousVector[1] + newWord.arousal]
        newES.setCurrentVector(nextV)
        return newES

    def getPhraseStates(self, phrase = [] , currentState = emotionState()):
        #retuns a vecor of the states if you wante to follw the eotion trajectory
        eStates =[]
        CES = currentState
        for w in phrase:
            CES = self.getNextEState(w, CES)
            eStates.append(CES)
        return eStates
    def getPhraseOutcomeState(self,phrase = [] , currentState = emotionState()):
        #gets the last staet from a phrase
        states = self.getPhraseStates(phrase,currentState)
        return states.pop()
    def getPhraseOutcomeVector(self, phrase=[], currentState=emotionState()):
        #returns the vecotr from the last state of a phrase
        return self.getPhraseOutcomeState(phrase, currentState).currentVector

class bFunction(object):
    #this help to keep values in the desired bounds
    def __init__(self, type = "complex", const=1):
        self.type =type
        self.k = const
    def evaluate(self, CValue, newValue,decayC =1.00):
        if (self.type == "complex"): #this is a normilizeng function desined to have decay that helps keep it from
            #getting stuck at 5 to -5. This is desinged to have the behvior we felt was fitting for the emption of
            #lyrics
            if abs(CValue)>= 5:
                currentX = 4.9
            else :currentX = atanh(CValue/5)
            currentX = currentX
            nextVal = 5 * (tanh((1+self.k) * (currentX  + newValue/5)) +tanh((1-self.k) * (currentX  + newValue/5)))/ (2*decayC)
        elif (self.type =="exp"):
            nextVal = 5*erf(self.k*(CValue+newValue))
        elif (self,type == "tanh"):
            if abs(CValue)>= 5:
                currentX = 4.9
            else: currentX = self.k*atanh( CValue/5)
            currentX = currentX
            nextVal = tanh(self.k * (currentX + newValue))
        return nextVal;
class boundedESM(emotionStateMechine): #this normalizes the emotions but other wise simply adds values together
    def __init__(self, boundFunction):
        self.binder=boundFunction;
    def getNextEState(self,newWord = word(),  currentEState = emotionState()):
        newES = emotionState()
        newES.initNewES(currentEState, newWord)
        nextValence = self.binder.evaluate(currentEState.previousVector[0], newWord.valence)
        nextArousal = self.binder.evaluate(currentEState.previousVector[1], newWord.arousal)
        nextV = [nextValence,nextArousal]
        newES.setCurrentVector(nextV)
        return newES

class lyricAnalyiser(object):
    #tihis is a class that analyzes a song after a Emotion stat mechin is set
    def __init__(self, ESM = emotionStateMechine()):
        self.ESM = ESM
    def quickOutcome(self, newSong = [], printValues =False):
        currentState = emotionState()
        for p in newSong:
            currentState = self.ESM.getPhraseOutcomeState(p,currentState)
            currentState = self.ESM.getPhraseOutcomeState(p, currentState)
            if printValues: print (currentState.currentVector)
        return currentState

import cPickle as pickle
def analyzeLyricalEmotionContent(LyricFileName = 'lyric.txt', dictionaryFile = 'dictionary.p', EmotionStateMechineFile = 'ESM.p'):
## this is a function that returns the emotion vector given a emostion state mechin, ditionary and song
    d = pickle.load(open(dictionaryFile, "rb"))
    ESM = pickle.load(open(EmotionStateMechineFile,"rb"))
    al = lyricAnalyiser(ESM)
    return al.quickOutcome(loadSong(LyricFileName,d))









