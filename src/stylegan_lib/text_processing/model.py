import torch
import torch.nn as nn
import numpy as np
import torchvision.models as models
from transformers import  DistilBertModel, AlbertModel, RobertaModel, BertModel, AdamW, DistilBertConfig, BertConfig, AlbertConfig, RobertaConfig

class BertRegressor(nn.Module):    
    def __init__(self, model_type, from_pretrained = True):
        super(BertRegressor,self).__init__()

        if from_pretrained:
            if model_type == 'bert-base-uncased':
                self.bert = BertModel.from_pretrained('bert-base-uncased')

            if model_type == 'distilbert-base-uncased':
                self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')

            if model_type == 'albert-base-v2':
                self.bert = AlbertModel.from_pretrained('albert-base-v2')

            if model_type == 'roberta-base':
                self.bert = RobertaModel.from_pretrained('roberta-base') 

        else:
            if model_type == 'bert-base-uncased':
                self.bert = BertModel(BertConfig())

            if model_type == 'distilbert-base-uncased':
                self.bert = DistilBertModel(DistilBertConfig())

            if model_type == 'albert-base-v2':
                self.bert = AlbertModel(AlbertConfig())

            if model_type == 'roberta-base':
                self.bert = RobertaModel(RobertaConfig())
        

        self.regressor = nn.Linear(768, 34, bias=True)
        self.dropout = nn.Dropout(0.1)
                
    def forward(self, input_ids, attention_mask, train = True):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output,_ = torch.max(outputs['last_hidden_state'], 1)
        if train:
            pooled_output = self.dropout(pooled_output)
        logits_reg = self.regressor(pooled_output)
        return logits_reg
