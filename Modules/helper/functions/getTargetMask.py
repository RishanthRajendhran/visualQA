import torch

def getTargetMask(maxLen):
    targetMask = torch.tril(torch.ones(maxLen, maxLen)==1)
    targetMask = targetMask.float()
    targetMask = targetMask.masked_fill(targetMask==0, float("-inf"))
    targetMask = targetMask.masked_fill(targetMask==1, float(0))
    return targetMask