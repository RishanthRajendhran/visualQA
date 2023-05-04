from Modules.helper.imports.classImports import VisualQADataset, VQAModel
from Modules.helper.imports.functionImports import checkFile, createDataLoader, trainModel, evaluateModel, testModel
from Modules.helper.imports.packageImports import load_dataset, pd, torch, cv2, torchvision, transformers, logging, argparse, pickle, nltk

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
    "-numEpochs",
    type=int,
    help="Number of epochs to train model for",
    default=5
)

parser.add_argument(
    "-learningRate",
    type=float,
    help="Learning rate for optimizer",
    default=0.01
)

parser.add_argument(
    "-weightDecay",
    type=float,
    help="Weight Decay for optimizer",
    default=0.01
)

parser.add_argument(
    "-imageSize",
    type=int,
    help="Target Size of image [(3, imageSize, imageSize)]",
    default=224
)

parser.add_argument(
    "-numAttnHeads",
    type=int,
    help="No. of attention heads in the decoder transformer layer",
    default=2
)

parser.add_argument(
    "-numLayers",
    type=int,
    help="No. of decoder transformer layers in the decoder block",
    default=4
)

parser.add_argument(
    "-vocab",
    help="Path to file containing Decoder vocabulary",
    default="vocab.pkl"
)

parser.add_argument(
    "-generateVocab",
    action="store_true",
    help="Boolean flag to enable generation of Decoder vocabulary from train set labels"
)

parser.add_argument(
    "-load",
    help="Path to file containing model to load",
    default=None
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
numEpochs = args.numEpochs
learningRate = args.learningRate
weightDecay = args.weightDecay
imageSize = args.imageSize
numAttnHeads = args.numAttnHeads
numLayers = args.numLayers
vocabPath = args.vocab
generateVocab = args.generateVocab
loadModel = args.load

if generateVocab:
    vocab = {}
    allWords = set()
else:
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

oriTrain, oriValid = ds["train"], ds["validation"]
train, valid, test = [], [], []

#Discard questions with missing answers
for t in oriTrain:
    if len(t["label"]["weights"]):
        image = cv2.imread(t["image_id"])
        t["image"] = imgToTensor(image)
        train.append(t)
        if generateVocab:
            newWords = set()
            for wLabel in t["label"]["ids"]:
                newWords.update(set(nltk.tokenize.word_tokenize(wLabel)))
            allWords.update(newWords)
    if len(train) >= maxSamples:
        break

if generateVocab:
    for i, w in enumerate(allWords):
        vocab[w] = i
    if "[PAD]" not in vocab.keys():
        vocab["[PAD]"] = len(vocab)
    if "[CLS]" not in vocab.keys():
        vocab["[CLS]"] = len(vocab)
    if "[SEP]" not in vocab.keys():
        vocab["[SEP]"] = len(vocab)
    if "[UNK]" not in vocab.keys():
        vocab["[UNK]"] = len(vocab)
    with open(vocabPath,"wb") as f:
        pickle.dump(vocab, f)

for v in oriValid:
    if len(v["label"]["weights"]):
        image = cv2.imread(v["image_id"])
        v["image"] = imgToTensor(image)
        valid.append(v)
    if max(valTestSplit, 1-valTestSplit)*len(valid) >= maxSamples:
        break

test = valid[int(len(valid)*valTestSplit):]
valid = valid[:int(len(valid)*valTestSplit)]

trainDF = pd.DataFrame(train)
validDF = pd.DataFrame(valid)
testDF = pd.DataFrame(test)

if torch.cuda.is_available():
  device = torch.device("cuda")
else:
  device = torch.device("cpu")

if loadModel:
    checkFile(loadModel, ".pt")
    model = torch.load(loadModel)
else:
    model = VQAModel(preTrainedModel, maxLen, numAttnHeads, numLayers, len(vocab))
model = model.to(device)

tokenizer = transformers.BertTokenizer.from_pretrained(preTrainedModel)

#Perform Data Augementations
#Not suitable for this dataset

trainDataLoader = createDataLoader(trainDF, vocab, tokenizer, maxLen, batchSize, device)
validDataLoader = createDataLoader(validDF, vocab, tokenizer, maxLen, batchSize, device)
testDataLoader = createDataLoader(testDF, vocab, tokenizer, maxLen, batchSize, device)

# optimizer = transformers.AdamW(
#     model.parameters(), 
#     lr=learningRate, 
#     correct_bias=False, 
#     weight_decay=weightDecay
# )
# optimizer = torch.optim.Adam(
#     model.parameters(), 
#     lr=learningRate, 
#     weight_decay=weightDecay
# )
optimizer = torch.optim.SGD(
    model.parameters(), 
    lr=learningRate, 
)
totalSteps = numEpochs
scheduler = transformers.get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=totalSteps
)

#Do not reward/punish model for padding, starting and ending tokens
vocabWeights = torch.ones(len(vocab))
for w in ["[PAD]", "[CLS]", "[SEP]"]:
    if w in vocab.keys():
        vocabWeights[vocab[w]] = 0

lossFunction = torch.nn.CrossEntropyLoss(weight=vocabWeights).to(device)

history = {
    "trainAcc": [],
    "trainLoss": [],
    "validAcc": [],
    "validLoss": [],
    "testAcc": [],
    "testLoss": []
}
bestAcc = 0 

logging.info(f"Data Statistics:")
logging.info(f"\tTrain: {len(trainDF)} examples")
logging.info(f"\tValidation: {len(validDF)} examples")
logging.info(f"\tTest: {len(testDF)} examples")
logging.info("*"*15)

for epoch in range(numEpochs):
    logging.info(f"Epoch {epoch+1}/{numEpochs}")

    trainAcc, trainLoss = trainModel(
        model,
        trainDataLoader,
        lossFunction,
        optimizer, 
        device, 
        scheduler,
        len(trainDF)
    )

    logging.info(f"\tTrain Accuracy: {trainAcc}, Train Loss: {trainLoss}")

    validAcc, validLoss = evaluateModel(
        model,
        validDataLoader,
        lossFunction,
        device, 
        len(validDF)
    )

    logging.info(f"\tValidation Accuracy: {validAcc}, Validation Loss: {validLoss}")

    # testAcc = testModel(
    #     model, 
    #     testDataLoader, 
    #     device, 
    #     len(testDF), 
    #     vocab
    # )

    # logging.info(f"\tTest Accuracy: {testAcc}")

    history["trainAcc"].append(trainAcc)
    history["trainLoss"].append(trainLoss)

    history["validAcc"].append(validAcc)
    history["validLoss"].append(validLoss)

    # history["testAcc"].append(validAcc)

    if validAcc > bestAcc: 
        torch.save(model, "fullModel.pt")
        torch.save(model.state_dict(), "modelStateDict.pt")
        bestAcc = validAcc

with open("history.pkl","wb") as f:
    pickle.dump(history, f)

# {
#     'question': 'What is this photo taken looking through?', 
#     'question_type': 'what is this', 
#     'question_id': 458752000, 
#     'image_id': '/scratch/general/vast/u1419542/huggingface_cache/datasets/downloads/extracted/e630cabf88ea136437d313fce4007ab4e63b456f32041d24fc4f5a8527edf10a/train2014/COCO_train2014_000000458752.jpg', 
#     'answer_type': 'other', 
#     'label': 
#         {   
#             'ids': ['net'], 
#             'weights': [1.0]
#         }
# }

#Test set has no gold answers
#Need to do cross-validation