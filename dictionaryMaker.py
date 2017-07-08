from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
import csv
import nltk
#This takes the dicitionary file and loads it to a dictionary with valence and arousal
def loadDictionary(dictionaryFileName = 'wordVectorDictionary.csv'):
    lemmatizer = WordNetLemmatizer()
    snowStem = SnowballStemmer('english')
    DictFile = open(dictionaryFileName)
    reader = csv.reader(DictFile)
    Dict = {}
    for w in reader:
        #parses the data from the dictionary into words and their associated vector
        currentWord = w[1].lower()
        valence = float(w[2])-5
        arousal = float(w[3])-5
        Dict[lemmatizer.lemmatize(currentWord)] = [valence,arousal]
        snowWord = snowStem.stem(currentWord)
        #take care of words that are steamed differently
        #The dictionary was made with words that would be stemmed into each other so I first try the original word then
        #I try the stemmed version
        if not Dict.has_key(snowWord):
            Dict[snowWord] =  [valence,arousal]
    return Dict