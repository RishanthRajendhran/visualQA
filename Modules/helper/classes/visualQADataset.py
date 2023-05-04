import torch 
import nltk
import numpy as np
from Modules.helper.functions.getTargetMask import getTargetMask
from Modules.helper.functions.getTargetPaddingMask import getTargetPaddingMask

class VisualQADataset:
    def __init__(self, questions, questionTypes, questionIDs, imageIDs, images, answerTypes, labels, vocab, tokenizer, maxLen, device):
        self.questions = questions
        self.questionTypes = questionTypes 
        self.questionIDs = questionIDs 
        self.imageIDs = imageIDs 
        self.images = images
        self.answerTypes = answerTypes 
        self.labels = labels
        self.vocab = vocab
        self.tokenizer = tokenizer 
        self.maxLen = maxLen
        self.device = device
    
    def __len__(self):
        return len(self.questionIDs)

    def __getitem__(self, item):
        curQuestion = self.questions[item]
        curQuestionType = self.questionTypes[item]
        curQuestionID = self.questionIDs[item]
        curImageID = self.imageIDs[item]
        curImage = self.images[item]
        curAnswerType = self.answerTypes[item]
        curLabel = self.labels[item]

        if "[PAD]" not in self.vocab.keys():
            raise RuntimeError("Could not find [PAD] in vocab!")
        if "[UNK]" not in self.vocab.keys():
            raise RuntimeError("Could not find [UNK] in vocab!")
        if "[CLS]" not in self.vocab.keys():
            raise RuntimeError("Could not find [CLS] in vocab!")
        if "[SEP]" not in self.vocab.keys():
            raise RuntimeError("Could not find [SEP] in vocab!")
        newLabel = [self.vocab["[PAD]"]]*self.maxLen
        newTarget = [self.vocab["[PAD]"]]*self.maxLen
        newTarget[0] = self.vocab["[CLS]"]
        lastI = -1
        for i, word in enumerate(nltk.tokenize.word_tokenize(curLabel["ids"][np.argmax(curLabel["weights"])])):
            lastI = i
            if i >= self.maxLen:
                break
            if word in self.vocab.keys():
                newLabel[i] = self.vocab[word]
                newTarget[i+1] = self.vocab[word]
            elif word.lower() in self.vocab.keys():
                newLabel[i] = self.vocab[word.lower()]
                newTarget[i+1] = self.vocab[word.lower()]
            else: 
                newLabel[i] = self.vocab["[UNK]"]
                newTarget[i+1] = self.vocab["[UNK]"]
        newLabel[min(self.maxLen-1, lastI+1)] = self.vocab["[SEP]"]
        newTarget[min(self.maxLen-1, (lastI+1)+1)] = self.vocab["[SEP]"]
        newLabel = torch.tensor(newLabel, dtype=torch.long)
        newTarget = torch.tensor(newTarget, dtype=torch.long)

        targetMask = getTargetMask(self.maxLen)
        targetPaddingMask = getTargetPaddingMask(self.vocab, newTarget)
                                               

        encoding = self.tokenizer.encode_plus(
            curQuestion,
            max_length=self.maxLen,
            add_special_tokens=True,
            padding="max_length",
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors="pt",
            truncation=True
        )
        queInputIDs = encoding["input_ids"].reshape(-1,).tolist()
        queAttentionMasks = encoding["attention_mask"].reshape(-1,).tolist()
        return {
            "question": curQuestion,
            "questionType": curQuestionType,
            "questionID": curQuestionID,
            "imageID": curImageID,
            "image": curImage,
            "answerType": curAnswerType,
            "queInputIDs": queInputIDs,
            "queAttentionMasks": queAttentionMasks,
            "label": newLabel,
            "target": newTarget,
            "targetMask": targetMask,
            "targetPaddingMask": targetPaddingMask
        }