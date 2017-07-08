from random import randint
from random import random
from lyrics import emotionState
from lyrics import emotionStateMechine
from lyrics import word
from lyrics import bFunction
import numpy

class gene (object):
    def __init__(self,maxW=.01):
        self.conditionObject = randint(0, 9)  # 0 is two previous, 1 is previous, 2 is current
        self.condition = randint(0,8)
        self.weight = (random()-.5)*2*maxW
        self.targetVector = randint(0,1)
   #     self.operator = randint(0,1) # maybe use this if need more complexity
    def getVector(self, ES =emotionState()):
        output = [0,0]
        #if self.operator == 0:
        output[self.targetVector] = self.weight
  #      else:
  #          output[self.targetVector] =
        return output
    def run(self, ES = emotionState()):
        if self.conditionObject==0:
            if self.condition<4:
                if ES.currentVector[0]<0 and abs(ES.currentVector[0])>self.condition:
                    return self.getVector()
            if self.condition ==4:
                if abs(ES.currentVector[0])<=self.condition:
                    return self.getVector()
            else:
                if ES.currentVector[0]>(self.condition-4):
                    return self.getVector()
        elif self.conditionObject == 1:
            if self.condition < 4:
                if ES.currentVector[1] < 0 and abs(ES.currentVector[1]) >  self.condition:
                    return self.getVector()
            if self.condition == 4:
                if abs(ES.currentVector[1]) <=  self.condition:
                    return self.getVector()
            else:
                if ES.currentVector[1] >  (self.condition - 4):
                    return self.getVector()
        elif self.conditionObject==2:
            if self.condition == 0 and ES.currentPOS.startswith('NN'): #nounn
                return self.getVector()
            if self.condition == 1 and ES.currentPOS.startswith('V'): #verb
                return self.getVector()
            if self.condition == 2 and ES.currentPOS.startswith('RB'):
                return self.getVector()
            if self.condition == 3 and ES.currentPOS.startswith('JJ'):
                return self.getVector()
            if self.condition == 4  and ES.currentPOS.startswith('W'):
                return self.getVector()
            if self.condition == 5  and ES.currentPOS.startswith('P'):
                return self.getVector()
            if self.condition == 6 and ES.currentPOS.startswith('C'):
                return self.getVector()
            if self.condition == 7 and ES.currentPOS.startswith('DT'):
                return self.getVector()
            if self.condition == 8:
                return self.getVector()
        elif self.conditionObject==3:
            if self.condition<4:
                if ES.previousVector[0]<0 and abs(ES.previousVector[0])>self.condition:
                    return self.getVector()
            if self.condition ==4:
                if abs(ES.previousVector[0])<=self.condition:
                    return self.getVector()
            else:
                if ES.previousVector[0]>(self.condition-4):
                    return self.getVector()
        elif self.conditionObject == 4:
            if self.condition < 4:
                if ES.previousVector[1] < 0 and abs(ES.previousVector[1]) > self.condition:
                    return self.getVector()
            if self.condition == 4:
                if abs(ES.previousVector[1]) <= self.condition:
                    return self.getVector()
            else:
                if ES.previousVector[1] > (self.condition - 4):
                    return self.getVector()
        elif self.conditionObject==5:
            if self.condition == 0 and ES.previousPOS.startswith('NN'): #nounn
                return self.getVector()
            if self.condition == 1 and ES.previousPOS.startswith('V'): #verb
                return self.getVector()
            if self.condition == 2 and ES.previousPOS.startswith('RB'):
                return self.getVector()
            if self.condition == 3 and ES.previousPOS.startswith('JJ'):
                return self.getVector()
            if self.condition == 4  and ES.previousPOS.startswith('W'):
                return self.getVector()
            if self.condition == 5  and ES.previousPOS.startswith('P'):
                return self.getVector()
            if self.condition == 6 and ES.previousPOS.startswith('C'):
                return self.getVector()
            if self.condition == 7 and ES.previousPOS.startswith('DT'):
                return self.getVector()
            if self.condition == 8:
                return self.getVector()
        elif self.conditionObject==6:
            if self.condition<4:
                if ES.previous2Vector[0]<0 and abs(ES.previous2Vector[0])>self.condition:
                    return self.getVector()
            if self.condition ==4:
                if abs(ES.previous2Vector[0])<=self.condition:
                    return self.getVector()
            else:
                if ES.previous2Vector[0]>(self.condition-4):
                    return self.getVector()
        elif self.conditionObject == 7:
            if self.condition < 4:
                if ES.previous2Vector[1] < 0 and abs(ES.previous2Vector[1]) >  self.condition:
                    return self.getVector()
            if self.condition == 4:
                if abs(ES.previous2Vector[1]) <= self.condition:
                    return self.getVector()
            else:
                if ES.previous2Vector[1] > (self.condition - 4):
                    return self.getVector()
        elif self.conditionObject==8:
            if self.condition == 0 and ES.previous2POS.startswith('NN'): #nounn
                return self.getVector()
            if self.condition == 1 and ES.previous2POS.startswith('V'): #verb
                return self.getVector()
            if self.condition == 2 and ES.previous2POS.startswith('RB'):
                return self.getVector()
            if self.condition == 3 and ES.previous2POS.startswith('JJ'):
                return self.getVector()
            if self.condition == 4  and ES.previous2POS.startswith('W'):
                return self.getVector()
            if self.condition == 5  and ES.previous2POS.startswith('P'):
                return self.getVector()
            if self.condition == 6 and ES.previous2POS.startswith('C'):
                return self.getVector()
            if self.condition == 7 and ES.previous2POS.startswith('DT'):
                return self.getVector()
            if self.condition == 8:
                return self.getVector()
        elif self.conditionObject == 9:#size of phrase
            if self.condition == 0 and ES.currentPhraseLength > self.condition:
                return self.getVector()
        return [0,0]


class chromozone (object):
    def __int__(self,maxSize=100, geneList =[]):
        self.genes = geneList
        self.maxSize=maxSize
    def makeRandomCh(self, n=1, maxW=.01):
        self.genes=[]
        for i in range(0,n):
            self.genes.append(gene(maxW))
    def spawnCh(self, mom,dad,mutationC,chSize):
        self.genes=[]
        self.maxSize = chSize
        geneCount = 0
        momL = len(mom.genes)
        dadL = len(dad.genes)
        if momL+dadL < (1-mutationC)*chSize:
            self.genes.append(mom.getAllGenes)
            self.genes.append(dad.getAllGenes)
            geneCount = momL+dadL
        else:
            numFromMom = int(random()*(1-mutationC)*chSize)
            numFromDad = int((1-mutationC)*chSize-numFromMom)
            for i in range(numFromMom): self.genes.append(mom.genes[i])
            for i in range(numFromDad): self.genes.append(dad.genes[i])
#            if numFromMom>0: self.genes.append(mom.getGenes(numFromMom))
#            if numFromDad>0: self.genes.append(dad.getGenes(numFromDad))
            geneCount = len(self.genes)
        while (geneCount<chSize):
            self.genes.append(gene())
            geneCount+=1

    def getGenes(self, n=0):
        outputGenes=[]
        oldGenes = self.genes
        i=0
        while i<n and 0<len(oldGenes):
            newIndex = randint(0,len(oldGenes)-1)
            outputGenes.append(oldGenes.pop(newIndex))
        return outputGenes
    def getAllGenes(self):
        return self.genes
    def evaluate(self,ES=emotionState()):
        vector=[0,0]
        for g in self.genes:
            newV = g.run(ES)
            vector = [vector[0] +newV[0],vector[1] +newV[1]]
        return vector



class geneticESM(emotionStateMechine):
    #emotion state mechine that uses words and word around it to get prediction of the next emotion state
    def __init__(self,binder = bFunction(),chCount = 10, geneCount=100, geneMaxW = .01,chromozones=[] ):
        if chromozones==[]:
            self.chromozones = []
            for i in range(chCount):
                newCh = chromozone()
                newCh.makeRandomCh(geneCount,geneMaxW)
                self.chromozones.append(newCh)
        else:
            self.chromozones=chromozones
        self.numCh = chCount
        self.binder = binder
        self.geneCount = geneCount
    def getNextEState(self,newWord = word(),  currentEState = emotionState(), chIndex =0):
        newES = emotionState()
        fitness=[]
        newES.initNewES(currentEState, newWord)
        c = self.chromozones[chIndex]
        newChEvaluation = c.evaluate(currentEState)
        newValence = newChEvaluation[0] + newWord.valence #this could be adjusted a lot posibly
        if newValence >=4.99:
            newValence = 4.99
        newArousal = newChEvaluation[1] + newWord.arousal
        if newArousal >= 4.99:
            newArousal = 4.99
        nextValence = self.binder.evaluate(currentEState.previousVector[0], newValence)
        nextArousal = self.binder.evaluate(currentEState.previousVector[1], newArousal)

        nextV = [nextValence,nextArousal]
        newES.setCurrentVector(nextV)
        return newES

    def getPhraseOutcomeState(self,phrase = [] , currentState = emotionState(),i=0):
        eStates =[]
        CES = currentState
        for w in phrase:
            CES = self.getNextEState(w, CES,i)
            eStates.append(CES)
        return eStates.pop()

    def getPhraseOutcomeVector(self, phrase=[], currentState=emotionState()):
        chIndependentOutComeVectors =[self.getPhraseOutcomeState(phrase, currentState, i).currentVector for i in range (self.numCh)]

        return [sum(chIndependentOutComeVectors[:][i])/len(self.chromozones) for i in range(2)]

    def getFitness(self,phrase=[], expectedOutcomeVector=[0,0], startingState=emotionState(),):
        chIndependentOutComeVectors = [self.getPhraseOutcomeState(phrase, startingState, i).currentVector for i in range(self.numCh)]
        return [10-numpy.sqrt((chIndependentOutComeVectors[i][0] - expectedOutcomeVector[0])**2 + (chIndependentOutComeVectors[i][1] - expectedOutcomeVector[1])**2) for i in range(self.numCh)]
    def evolve(self,numSurvivors=0, mutationC=0):
        newChrArry=[]
        for i in range(numSurvivors):newChrArry.append(self.chromozones[i])
        if numSurvivors > 1:
            while len(newChrArry)<self.numCh:
                newCh = chromozone()
                mom = newChrArry[randint(0, numSurvivors-1)]
                dad = newChrArry[randint(0, numSurvivors-1)]

                newCh.spawnCh(mom, dad, mutationC, self.geneCount)
                newChrArry.append(newCh)
        else:
            while len(newChrArry)<self.numCh:
                newCh = chromozone()
                newCh.makeRandomCh()
                newChrArry.append(newCh)

        self.chromozones=newChrArry

class geneticESM2(emotionStateMechine):
    # this si a class of genetic emotion state mechine that gets it's emotion state by ignoring all non emotion words
    def __init__(self,binder = bFunction(),chCount = 10, geneCount=100, geneMaxW = .01,chromozones=[] ):
        if chromozones==[]:
            self.chromozones = []
            for i in range(chCount):
                newCh = chromozone()
                newCh.makeRandomCh(geneCount,geneMaxW)
                self.chromozones.append(newCh)
        else:
            self.chromozones=chromozones
        self.numCh = chCount
        self.binder = binder
        self.geneCount = geneCount
    def getNextEState(self,newWord = word(),  currentEState = emotionState(), chIndex =0):
        newES = emotionState()
        fitness=[]
        newES.initNewES(currentEState, newWord)
        c = self.chromozones[chIndex]
        newChEvaluation = c.evaluate(currentEState)
        newValence = newChEvaluation[0] + newWord.valence #this could be adjusted a lot posibly
        if newValence >=4.99:
            newValence = 4.99
        newArousal = newChEvaluation[1] + newWord.arousal
        if newArousal >= 4.99:
            newArousal = 4.99
        nextValence = self.binder.evaluate(currentEState.previousVector[0], newValence)
        nextArousal = self.binder.evaluate(currentEState.previousVector[1], newArousal)

        nextV = [nextValence,nextArousal]
        newES.setCurrentVector(nextV)
        return newES

    def getPhraseOutcomeState(self,phrase = [] , currentState = emotionState(),i=0):
        eStates =[]
        CES = currentState
        for w in phrase:
            if w.valence != 0 or w.arousal !=0:
                CES = self.getNextEState(w, CES,i)
                eStates.append(CES)
        return eStates.pop()

    def getPhraseOutcomeVector(self, phrase=[], currentState=emotionState()):
        chIndependentOutComeVectors =[self.getPhraseOutcomeState(phrase, currentState, i).currentVector for i in range (self.numCh)]

        return [sum(chIndependentOutComeVectors[:][i])/len(self.chromozones) for i in range(2)]

    def getFitness(self,phrase=[], expectedOutcomeVector=[0,0], startingState=emotionState(),):
        chIndependentOutComeVectors = [self.getPhraseOutcomeState(phrase, startingState, i).currentVector for i in range(self.numCh)]
        return [10-numpy.sqrt((chIndependentOutComeVectors[i][0] - expectedOutcomeVector[0])**2 + (chIndependentOutComeVectors[i][1] - expectedOutcomeVector[1])**2) for i in range(self.numCh)]
    def evolve(self,numSurvivors=0, mutationC=0):
        newChrArry=[]
        for i in range(numSurvivors):newChrArry.append(self.chromozones[i])
        if numSurvivors > 1:
            while len(newChrArry)<self.numCh:
                newCh = chromozone()
                mom = newChrArry[randint(0, numSurvivors-1)]
                dad = newChrArry[randint(0, numSurvivors-1)]

                newCh.spawnCh(mom, dad, mutationC, self.geneCount)
                newChrArry.append(newCh)
        else:
            while len(newChrArry)<self.numCh:
                newCh = chromozone()
                newCh.makeRandomCh()
                newChrArry.append(newCh)

        self.chromozones=newChrArry

import cPickle as pickle
class chromTrainer(object):
    def __init__(self, geneESM = geneticESM()):
        self.trainingPhrases = []
        self.trainingOutputVectors = []
        self.ESM = geneESM
        self.chNum = geneESM.numCh
        self.testPhrases =[]
        self.testVectors = []
    def setChromozones(self,ch):
        self.Chromozones = ch
    def randomelyPopulateCh(self):
        self.Chromozones=[]
        i=0
        while i< self.chNum:
            ch = chromozone()
            ch.makeRandomCh()
            self.Chromozones.append(ch)
            i+=1
    def setTrainingData(self, phrases, outComeVectors):
        self.trainingPhrases = phrases
        self.trainingOutputVectors = outComeVectors
    def splitTrain_Test(self, testPercent = .1):
        testNum = int(testPercent*len(self.trainingPhrases))
        for i in range (testNum):
            randNumb = randint(0, len(self.trainingPhrases))
            self.testPhrases.append(self.trainingPhrases.pop(randNumb))
            self.testVectors.append(self.trainingOutputVectors.pop(randNumb))

    def test(self):
        for i in range(len(self.testPhrases)):
            [v,a] = self.ESM.getPhraseOutcomeVector(self.testPhrases[i])
            crntTestVector = self.testVectors[i]
            l_2norm = numpy.sqrt((v+crntTestVector[0])**2+(a+crntTestVector[1])**2)
        return l_2norm/float(len(self.testPhrases))
    def runEpoch(self, mutationC=.1, survivorRatio=.1):
        n = len(self.trainingPhrases)
        fitness =[]
        avgFit=[]
        for i in range(n):
            fitness.append(self.ESM.getFitness(self.trainingPhrases[i],self.trainingOutputVectors[i]))
        for j in range(self.chNum):
            chromSum=0

            for i in range(n):
                chromSum += fitness[i][j]

            avgFit.append(chromSum/n)

        #avgFit = [numpy.average(fitness[:,i])for i in range(self.chNum)]
        fitIndexOrder = sorted(range(self.chNum),key=lambda i:avgFit[i], reverse=True)
        self.ESM.chromozones= [self.ESM.chromozones[i] for i in fitIndexOrder]

        self.ESM.evolve(int(survivorRatio*self.chNum),mutationC)
        return numpy.average(avgFit)

    def train(self, epochCount=1, MutationC = .3, SurvivalRatio = .5):
        for epoch in range(epochCount):
            print(self.runEpoch(mutationC=MutationC,survivorRatio=SurvivalRatio))
    def trainAnnealing(self, epochCount=1, MutationC = .3, SurvivalRatio = .5, EndMutationC = .01, EndSurvivalRatio = .99):
        MC = MutationC
        mStep = (MutationC-EndMutationC)/(epochCount-1)
        SR =SurvivalRatio
        sStep = (SurvivalRatio-EndSurvivalRatio)/(epochCount-1)
        extError = []
        intError = []
        c =0
        for epoch in range(epochCount):
            print(epoch)
            intError.append(10-self.runEpoch(mutationC=MC,survivorRatio=SR))
            extError.append(self.test())
            MC+=mStep
            SR+=sStep
            c+=1
            if c >10:
                pickle.dump(self.ESM, open("ESM3.p", "wb"))
                pickle.dump(extError, open("extError3.p", "wb"))
                pickle.dump(intError, open("intError3.p", "wb"))
                c=0
        print (extError)
        print (intError)
