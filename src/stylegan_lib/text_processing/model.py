import torch
import torch.nn as nn
import numpy as np
import torchvision.models as models
from transformers import  DistilBertModel, AlbertModel, RobertaModel, BertModel, AdamW

class BertRegressor(nn.Module):    
    def __init__(self, model_type):
        super(BertRegressor,self).__init__()

        if model_type == 'bert-base-uncased':
            self.bert = BertModel.from_pretrained('bert-base-uncased')

        if model_type == 'distilbert-base-uncased':
            self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')

        if model_type == 'albert-base-v2':
            self.bert = AlbertModel.from_pretrained('albert-base-v2')

        if model_type == 'roberta-base':
            self.bert = RobertaModel.from_pretrained('roberta-base') 

        self.regressor = nn.Linear(768, 34, bias=True)
        self.dropout = nn.Dropout(0.1)
                
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output,_ = torch.max(outputs['last_hidden_state'], 1)
        pooled_output = self.dropout(pooled_output)
        logits_reg = self.regressor(pooled_output)
        return logits_reg
