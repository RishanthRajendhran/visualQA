from Modules.helper.imports.functionImports import checkFile, createDataLoader, testModel
from Modules.helper.imports.packageImports import load_dataset, pd, np, torch, cv2, torchvision, transformers, logging, argparse, pickle

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
    default="/scratch/general/vast/u1419542/huggingface_cache/datasets/"
)

parser.add_argument(
    "-valTestSplit",
    type=float,
    help="Percentage split between validation and test sets as a fraction",
    default=0.9
)

parser.add_argument(
    "-batchSize",
    type=int,
    help="Batch size of dataloader",
    default=64
)

parser.add_argument(
    "-maxSamples",
    type=int,
    help="Maximum no. of samples to be used in train/validation/test sets (Memory constraints)",
    default=45000
)

parser.add_argument(
    "-maxLen",
    type=int,
    help="Maximum length of question sequence input to BERT model",
    default=128
)

parser.add_argument(
    "-preTrainedModel",
    type=str,
    help="Pretrained BERT model to use from transformers package",
    default="bert-base-cased"
)

parser.add_argument(
    "-imageSize",
    type=int,
    help="Target Size of image [(3, imageSize, imageSize)]",
    default=224
)

parser.add_argument(
    "-vocab",
    help="Path to file containing Decoder vocabulary",
    default="vocab.pkl"
)

parser.add_argument(
    "-load",
    help="Path to file containing model to load",
    default="fullModel.pt"
)

args =  parser.parse_args()

debug = args.debug
logFile = args.log
cacheDir = args.cacheDir
valTestSplit = args.valTestSplit 
batchSize = args.batchSize
maxSamples = args.maxSamples
maxLen = args.maxLen
preTrainedModel = args.preTrainedModel 
imageSize = args.imageSize
vocabPath = args.vocab
loadModel = args.load

checkFile(vocabPath, ".pkl")
with open(vocabPath, "rb") as f:
    vocab = pickle.load(f)

if logFile:
    checkFile(logFile)
    logging.basicConfig(filename=logFile, filemode='w', level=logging.INFO)
elif debug:
    logging.basicConfig(filemode='w', level=logging.DEBUG)
else:
    logging.basicConfig(filemode='w', level=logging.INFO)

if valTestSplit <= 0 or valTestSplit >= 1:
    raise ValueError(f"valTestSplit should be a floating value between 0 and 1!")
if batchSize <= 0:
    raise ValueError("Batch Size has to be a positive number!")


imgToTensor = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Resize(size=(imageSize, imageSize))
])

ds = load_dataset("Graphcore/vqa", cache_dir=cacheDir)

oriValid = ds["validation"]
valid, test = [], []

for v in oriValid:
    if len(v["label"]["weights"]):
        image = cv2.imread(v["image_id"])
        v["image"] = imgToTensor(image)
        valid.append(v)
    if max(valTestSplit, 1-valTestSplit)*len(valid) >= maxSamples:
        break

test = valid[int(len(valid)*valTestSplit):]
testDF = pd.DataFrame(test)

if torch.cuda.is_available():
  device = torch.device("cuda")
else:
  device = torch.device("cpu")

checkFile(loadModel, ".pt")
model = torch.load(loadModel)
model = model.to(device)

tokenizer = transformers.BertTokenizer.from_pretrained(preTrainedModel)

#Perform Data Augementations
#Not suitable for this dataset
testDataLoader = createDataLoader(testDF, vocab, tokenizer, maxLen, batchSize, device)

logging.info(f"Data Statistics:")
logging.info(f"\tTest: {len(testDF)} examples")
logging.info("*"*15)

allImages, allQuestions, allLabels, allTargets, testAcc = testModel(
    model, 
    testDataLoader, 
    device, 
    len(testDF), 
    vocab
)

logging.info(f"\tTest Accuracy: {testAcc}")

predictions = {
    "images": allImages,
    "questions": allQuestions,
    "labels": allLabels,
    "targets": allTargets
}

with open("testPredictions.pkl","wb") as f:
    pickle.dump(predictions, f)