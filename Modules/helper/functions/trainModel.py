import torch 
import numpy as np

def trainModel(model, dataLoader, lossFunction, optimizer, device, scheduler, numExamples):
    model = model.train()

    losses = []
    corrPreds = 0
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

        # corrPreds += torch.sum(preds == labels)
        corrPreds += torch.sum((preds==labels).all(dim=1))
        losses.append(loss.item())
        #Zero out gradients from previous batches
        optimizer.zero_grad()
        #Backwardpropagate the losses
        loss.backward()
        #Avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        #Perform a step of optimization
        optimizer.step()
    scheduler.step()
    return corrPreds.double()/numExamples, np.mean(losses)