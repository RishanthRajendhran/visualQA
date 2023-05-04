def getTargetPaddingMask(vocab, target):
    if "[PAD]" not in vocab.keys():
        raise RuntimeError("Could not find [PAD] in vocab!")
    return (target == vocab["[PAD]"])