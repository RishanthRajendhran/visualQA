from Modules.helper.classes.visualQADataset import VisualQADataset
import torch

def createDataLoader(df, vocab, tokenizer, maxLen, batchSize, device="cpu"):
    ds = VisualQADataset(
        questions = df["question"].to_numpy(), 
        questionTypes = df["question_type"].to_numpy(), 
        questionIDs = df["question_id"].to_numpy(), 
        imageIDs = df["image_id"].to_numpy(), 
        images = df["image"].to_numpy(),
        answerTypes = df["answer_type"].to_numpy(), 
        labels = df["label"].tolist(), 
        vocab = vocab,
        tokenizer=tokenizer,
        maxLen=maxLen,
        device = device
    )

    return torch.utils.data.DataLoader(
        ds,
        batch_size=batchSize,
        num_workers=0,
        shuffle=True,
        collate_fn=lambda batch: batch,
    )