import torch 
import numpy as np

def testModel(model, dataLoader, device, numExamples, vocab):
    model = model.eval()
    maxLen = model.getMaxLen()

    if "[PAD]" not in vocab.keys():
        raise RuntimeError("Could not find [PAD] in vocab!")
    if "[CLS]" not in vocab.keys():
            raise RuntimeError("Could not find [CLS] in vocab!")
    if "[SEP]" not in vocab.keys():
        raise RuntimeError("Could not find [SEP] in vocab!")

    corrPreds = 0 
    batchInd = 0

    allImages = []
    allQuestions = []
    allLabels = []
    allTargets = []
    allQuestionTypes = []
    allAnswerTypes = []

    with torch.no_grad():
        for d in dataLoader:
            batchInd += 1
            # logging.info(f"Batch {batchInd}/{len(dataLoader)}")
            images = []
            questions = []
            labels = []
            targets = []
            questionTypes = []
            answerTypes = []
            for i in range(len(d)):
                # logging.info(f"\tSample {i+1}/{len(d)}")
                images.append(d[i]["image"])
                questions.append(d[i]["question"])
                labels.append(d[i]["label"])
                questionTypes.append(d[i]["questionType"])
                answerTypes.append(d[i]["answerType"])
                target = torch.tensor([vocab["[CLS]"]], dtype=torch.long, device=device)
                for _ in range(maxLen):
                    output = model(
                        torch.stack([d[i]["image"]]),
                        torch.tensor([d[i]["queInputIDs"]]),
                        torch.tensor([d[i]["queAttentionMasks"]]),
                        torch.stack([target]),
                    )

                    nextTargetToken = output[-1].topk(1)[1].view(-1)[-1].item()
                    nextTargetToken = torch.tensor([nextTargetToken], dtype=torch.long, device=device)
                    target = torch.cat((target, nextTargetToken), dim=0)

                    if nextTargetToken.view(-1).item() == vocab["[SEP]"]:
                        break 
                padTarget = torch.tensor([vocab["[PAD]"]]*(maxLen-len(target)+1), dtype=torch.long, device=device)
                target = torch.cat((target[1:], padTarget))
                targets.append(target)

            allImages.extend(images)
            allQuestions.extend(questions)
            allLabels.extend(labels)
            allTargets.extend(targets) 
            allQuestionTypes.extend(questionTypes)
            allAnswerTypes.extend(answerTypes)

            labels = torch.stack(labels).to(device)
            targets = torch.stack(targets).to(device)
            # #No reward for getting the padding tokens right
            # targets[targets==vocab["[PAD]"]] = len(vocab)+1
            # corrPreds += torch.sum(targets == labels)
            corrPreds += torch.sum((targets==labels).all(dim=1))
    return allImages, allQuestions, allLabels, allTargets, allQuestionTypes, allAnswerTypes, corrPreds.double()/numExamples