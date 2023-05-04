from Modules.helper.imports.functionImports import plotPie, plotAnswerLength, checkFile
from Modules.helper.imports.packageImports import argparse, logging, Path, exists, pickle, np, load_dataset, pd, plt, os, nltk, mpl
mpl.rcParams['font.size'] = 6.0

parser = argparse.ArgumentParser()

parser.add_argument(
    "-debug",
    action="store_true",
    help="Boolean flag to enable debug mode"
)

parser.add_argument(
    "-log",
    type=str,
    help="Path to file to print logging information",
    default=None
)

parser.add_argument(
    "-cacheDir",
    help="Path to cache location for Huggingface datasets",
    default="./"
    # default="/scratch/general/vast/u1419542/huggingface_cache/datasets/"
)

parser.add_argument(
    "-valTestSplit",
    type=float,
    help="Percentage split between validation and test sets as a fraction",
    default=0.8
)

parser.add_argument(
    "-maxSamples",
    type=int,
    help="Maximum no. of samples to be used in train/validation/test sets (Memory constraints)",
    default=45000
)

args =  parser.parse_args()

debug = args.debug
logFile = args.log
cacheDir = args.cacheDir
valTestSplit = args.valTestSplit 
maxSamples = args.maxSamples

if logFile:
    checkFile(logFile)
    logging.basicConfig(filename=logFile, filemode='w', level=logging.INFO)
elif debug:
    logging.basicConfig(filemode='w', level=logging.DEBUG)
else:
    logging.basicConfig(filemode='w', level=logging.INFO)

if valTestSplit <= 0 or valTestSplit >= 1:
    raise ValueError(f"valTestSplit should be a floating value between 0 and 1!")


ds = load_dataset("Graphcore/vqa", cache_dir=cacheDir)

oriTrain, oriValid = ds["train"], ds["validation"]
train, valid, test = [], [], []

#Discard questions with missing answers
for t in oriTrain:
    if len(t["label"]["weights"]):
        t["goldLabel"] = t["label"]["ids"][np.argmax(t["label"]["weights"])]
        train.append(t)
    if len(train) >= maxSamples:
        break

for v in oriValid:
    if len(v["label"]["weights"]):
        v["goldLabel"] = v["label"]["ids"][np.argmax(v["label"]["weights"])]
        valid.append(v)
    if max(valTestSplit, 1-valTestSplit)*len(valid) >= maxSamples:
        break

test = valid[int(len(valid)*valTestSplit):]
valid = valid[:int(len(valid)*valTestSplit)]

trainDF = pd.DataFrame(train)
validDF = pd.DataFrame(valid)
testDF = pd.DataFrame(test)

plotPie(trainDF, "question_type", "trainQuestionTypes", "Question Types (Train set)")
plotPie(trainDF, "answer_type", "trainAnswerTypes", "Answer Types (Train set)")

plotPie(validDF, "question_type", "validQuestionTypes", "Question Types (Validation set)")
plotPie(validDF, "answer_type", "validAnswerTypes", "Answer Types (Validation set)")

plotPie(testDF, "question_type", "testQuestionTypes", "Question Types (Test set)")
plotPie(testDF, "answer_type", "testAnswerTypes", "Answer Types (Test set)")

plotAnswerLength(train, "Train set", "trainAnswerLength", "Answer Length (Train set)")
plotAnswerLength(valid, "Validation set", "validAnswerLength", "Answer Length (Validation set)")
plotAnswerLength(test, "Test set", "testAnswerLength", "Answer Length (Test set)")