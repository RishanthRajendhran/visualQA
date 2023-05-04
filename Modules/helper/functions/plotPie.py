import matplotlib.pyplot as plt 
import pandas as pd
import os

def plotPie(dataFrame, column, fileName, plotTitle, otherPercent=0.015):
    colVals = dataFrame[column].unique()
    colCounts = []
    for colVal in colVals:
        colCounts.append(dataFrame[dataFrame[column]==colVal].shape[0])

    colCounts = pd.DataFrame(zip(colVals, colCounts), columns=[column, "count"])
    colCounts.loc[colCounts["count"]<=otherPercent*len(dataFrame), column] = "other"
    colCounts = colCounts.groupby(column)['count'].sum().reset_index()

    plt.pie(colCounts['count'], labels=colCounts[column], autopct=customAutopct)
    plt.title(plotTitle)
    isExist = os.path.exists("./plots")
    if not isExist:
        os.makedirs("./plots")
    plt.savefig(f"./plots/{fileName}.png")
    plt.clf()

def customAutopct(pct):
    return '{p:.2f}%'.format(p=pct)