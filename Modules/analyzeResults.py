
from Modules.helper.imports.functionImports import checkFile
from Modules.helper.imports.packageImports import argparse, logging, Path, exists, os, pickle, np, plt

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
    "-predictions",
    help="Path to file containing predictions to visualize",
    default="testPredictions.pkl"
)

parser.add_argument(
    "-numSamples",
    type=int,
    help="No. of predictions to visualize",
    default=100
)

parser.add_argument(
    "-vocab",
    help="Path to file containing Decoder vocabulary",
    default="vocab.pkl"
)

args = parser.parse_args()
debug = args.debug
logFile = args.log
predictionsFile = args.predictions 
numSamples = args.numSamples
vocabPath = args.vocab

checkFile(vocabPath, ".pkl")
with open(vocabPath, "rb") as f:
    vocab = pickle.load(f)

inverseVocab = {}
for k, v in vocab.items():
    if v in inverseVocab.keys():
        raise ValueError(f"Invalid vocabulary: two words assigned the same index {v}!")
    inverseVocab[v] = k

if logFile:
    checkFile(logFile)
    logging.basicConfig(filename=logFile, filemode='w', level=logging.INFO)
elif debug:
    logging.basicConfig(filemode='w', level=logging.DEBUG)
else:
    logging.basicConfig(filemode='w', level=logging.INFO)

checkFile(predictionsFile, ".pkl")
with open(predictionsFile, "rb") as f:
    predictions = pickle.load(f)

images = predictions["images"]
questions = predictions["questions"]
labels = predictions["labels"]
targets = predictions["targets"]
questionTypes = predictions["questionTypes"]
answerTypes = predictions["answerTypes"]

if numSamples <= 0:
    logging.warning("numSamples has to be a positive number!")
    numSamples = 1
elif numSamples > len(questions):
    logging.warning("numSamples cannot be greater than no. of predictions!")
    numSamples = len(questions)

chosenIndices = np.random.choice(len(questions), numSamples).tolist()
chosenIndices.sort()
accuracy = {
    "questionType": {},
    "answerType": {},
    "overall": {
        "score": 0,
        "numExamples": 0
    }
}

isExist = os.path.exists("./plots")
if not isExist:
    os.makedirs("./plots")

isExist = os.path.exists("./plots/predictions")
if not isExist:
    os.makedirs("./plots/predictions")

for i in range(len(questions)):
    if questionTypes[i] not in accuracy["questionType"]:
        accuracy["questionType"][questionTypes[i]] = {
            "score": 0,
            "numExamples": 0
        }
    if answerTypes[i] not in accuracy["answerType"]:
        accuracy["answerType"][answerTypes[i]] = {
            "score": 0,
            "numExamples": 0
        }
    accuracy["overall"]["numExamples"] += 1
    accuracy["questionType"][questionTypes[i]]["numExamples"] += 1
    accuracy["answerType"][answerTypes[i]]["numExamples"] += 1

    if i in chosenIndices:
        logging.info("{}/{}".format(chosenIndices.index(i), len(chosenIndices)))
        logging.info(f"Question: {questions[i]}")
        plt.imshow(images[i].permute(1,2,0))
        plt.title(f"Test Prediction Image {chosenIndices.index(i)}")
        plt.savefig(f"./plots/predictions/testPredictionsImage_{chosenIndices.index(i)}.png")
    label = ""
    target = ""
    for j in range(len(labels[i])):
        if inverseVocab[int(labels[i][j])] == "[SEP]":
            break
        label += inverseVocab[int(labels[i][j])] + " "
    label = label.strip()
    for j in range(len(targets[i])):
        if inverseVocab[int(targets[i][j])] == "[SEP]":
            break
        target += inverseVocab[int(targets[i][j])] + " "
    target = target.strip()
    if i in chosenIndices:
        if label == target:
            logging.info("Performance: Correct")
        else: 
            logging.info("Performance: Incorrect")
        logging.info(f"Label      : {label}")
        logging.info(f"Prediction : {target}")
        logging.info("-"*20)
    if target == label:
        accuracy["overall"]["score"] += 1
        accuracy["questionType"][questionTypes[i]]["score"] += 1
        accuracy["answerType"][answerTypes[i]]["score"] += 1
logging.info("*"*40)
logging.info("Results:")
logging.info("\tAccuracy:")
logging.info("\t\tOverall: {:0.2f}%".format((accuracy["overall"]["score"]/accuracy["overall"]["numExamples"])*100))
logging.info("\t\tBy Question Type:")
for questionType in accuracy["questionType"].keys():
    logging.info("\t\t\t{}: {:0.2f}%".format(questionType, (accuracy["questionType"][questionType]["score"]/accuracy["questionType"][questionType]["numExamples"])*100))
logging.info("\t\tBy Answer Type:")
for answerType in accuracy["answerType"].keys():
    logging.info("\t\t\t{}: {:0.2f}%".format(answerType, (accuracy["answerType"][answerType]["score"]/accuracy["answerType"][answerType]["numExamples"])*100))

with open("testAccuracy.pkl","wb") as f:
    pickle.dump(accuracy, f)