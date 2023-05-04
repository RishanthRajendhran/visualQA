import torch
import torchvision 
import transformers
import numpy as np

class VQAModel(torch.nn.Module):
    def __init__(self, preTrainedModel, maxLen, numAttnHeads, numLayers, vocabSize):
        super(VQAModel, self).__init__()
        self.numAttnHeads = numAttnHeads
        self.numLayers = numLayers
        self.vocabSize = vocabSize
        self.preTrainedModel = preTrainedModel
        self.maxLen = maxLen 
        self.imageEncoder = torchvision.models.resnet18()
        self.imageEncoder.fc = torch.nn.Identity()
        self.imageBatchNorm = torch.nn.BatchNorm1d(num_features=512)
        self.questionEncoder = transformers.BertModel.from_pretrained(self.preTrainedModel)
        self.questionBatchNorm = torch.nn.BatchNorm1d(num_features=self.questionEncoder.config.hidden_size)
        self.concatenatedReprSize = 512 + self.questionEncoder.config.hidden_size
        self.decoderEmbedding = torch.nn.Embedding(num_embeddings=self.vocabSize, embedding_dim=self.concatenatedReprSize)
        self.decoderLayer = torch.nn.TransformerDecoderLayer(d_model=self.concatenatedReprSize, nhead=self.numAttnHeads, batch_first=True)
        self.transformerDecoder = torch.nn.TransformerDecoder(self.decoderLayer, num_layers=self.numLayers)
        self.outLayer = torch.nn.Sequential(
            torch.nn.Linear(self.concatenatedReprSize, self.vocabSize),
            # torch.nn.Softmax(dim=1)
        )
        self.device = "cpu"

    def forward(self, image, queInputID, queAttentionMask, target, targetMask=None, targetPaddingMask=None):
        image = image.to(self.device)
        queInputID = queInputID.to(self.device)
        queAttentionMask = queAttentionMask.to(self.device)
        target = target.to(self.device)
        if targetMask != None and targetPaddingMask != None:
            targetMask = targetMask.to(self.device)
            targetPaddingMask = targetPaddingMask.to(self.device)

        imageEmbed = self.imageEncoder(image)
        imageEmbed = self.imageBatchNorm(imageEmbed)
        questionEmbed = self.questionEncoder(
            input_ids=queInputID,
            attention_mask=queAttentionMask
        )
        questionEmbed["last_hidden_state"] = self.questionBatchNorm(questionEmbed["last_hidden_state"].permute([0,2,1])).permute([0,2,1])
        imageEmbed = imageEmbed.unsqueeze(1).repeat(1,self.maxLen,1)
        concatenatedRepr = torch.cat((imageEmbed, questionEmbed["last_hidden_state"]),dim=-1)
        tgt = self.decoderEmbedding(target)*np.sqrt(self.concatenatedReprSize)
        
        if targetMask != None and targetPaddingMask != None:
            if self.training:
                #Target mask needs to be provided to all attention heads
                targetMask = targetMask.repeat(self.numAttnHeads, 1, 1)

                # logging.info("*"*30)
                # logging.info(f"Shape of imageEmbed: {imageEmbed.shape}")
                # logging.info(f"Shape of concatenatedRepr: {concatenatedRepr.shape}")
                # logging.info(f"Shape of tgt: {tgt.shape}")
                # logging.info(f"Shape of targetMask: {targetMask.shape}")
                # logging.info(f"Shape of targetPaddingMask: {targetPaddingMask.shape}")
                # logging.info("*"*30)

                decoderOut = self.transformerDecoder(
                    tgt=tgt, 
                    memory=concatenatedRepr, 
                    tgt_mask=targetMask,
                    tgt_key_padding_mask=targetPaddingMask
                )
            else:
                decoderOut = self.transformerDecoder(
                tgt=tgt, 
                memory=concatenatedRepr, 
                # tgt_is_causal=True,
                # tgt_mask=targetMask,
                tgt_key_padding_mask=targetPaddingMask
            )
        else:
            decoderOut = self.transformerDecoder(
                tgt=tgt, 
                memory=concatenatedRepr, 
            )
        out = self.outLayer(decoderOut)
        return out
    
    def to(self, device):
        self.device = device 
        self = super().to(device)
        return self
    
    def getMaxLen(self):
        return self.maxLen