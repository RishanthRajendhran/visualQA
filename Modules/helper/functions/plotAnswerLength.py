from Modules.helper.functions.plotPie import plotPie
import logging
import pandas as pd
import nltk

def plotAnswerLength(dataset, datasetType, fileName, plotTitle):
    answerLengths = []
    answerAvgLen = 0
    vocab = set()
    for d in dataset: 
        wordsInAnswer = nltk.tokenize.word_tokenize(d["goldLabel"])
        vocab.update(wordsInAnswer)
        answerLength = len(wordsInAnswer)
        answerAvgLen += answerLength
        answerLengths.append(answerLength)
    answerAvgLen /= len(dataset)
    logging.info(f"{datasetType}:")
    logging.info(f"\tAverage answer length: {answerAvgLen}")
    logging.info(f"\tSize of vocabulary: {len(vocab)}")
    answerLengths = pd.DataFrame(answerLengths, columns=["answerLength"])
    plotPie(answerLengths, "answerLength", fileName, plotTitle, otherPercent=0)