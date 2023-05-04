import torch
import numpy as np 

def evaluateModel(model, dataLoader, lossFunction, device, numExamples):
    model = model.eval()

    losses = []
    corrPreds = 0 

    with torch.no_grad():
        for d in dataLoader:
            curImages = []
            queInputIDs = []
            queAttentionMasks = []
            labels = []
            targets = []
            targetMasks = []
            targetPaddingMasks = []
            for i in range(len(d)):
                curImages.append(d[i]["image"])
                queInputIDs.append(d[i]["queInputIDs"])
                queAttentionMasks.append(d[i]["queAttentionMasks"])
                labels.append(d[i]["label"])
                targets.append(d[i]["target"])
                targetMasks.append(d[i]["targetMask"])
                targetPaddingMasks.append(d[i]["targetPaddingMask"])
            labels = torch.stack(labels).to(device)

            outputs = model(
                torch.stack(curImages),
                torch.tensor(queInputIDs),
                torch.tensor(queAttentionMasks),
                torch.stack(targets),
                torch.stack(targetMasks),
                torch.stack(targetPaddingMasks),
            )

            _, preds = torch.max(outputs, dim=-1)

            loss = lossFunction(torch.flatten(outputs,0,1), torch.flatten(labels,0,1))

            corrPreds += torch.sum((preds==labels).all(dim=1))
            # corrPreds += torch.sum(preds == labels)
            losses.append(loss.item())
    return corrPreds.double()/numExamples, np.mean(losses)