import torch
from torch.utils.data import Dataset
import pandas as pd
from skimage import io
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, AlbertTokenizer, RobertaTokenizer, BertTokenizer
'''
# training loop
for epoch in range(num_epochs):
    # loop over all batches
    for i in range(total_batches):
        batch_x, batch_y = ...
'''
def do_data(csv_file, model_type, train_split_ratio = 0.8):
    # read the data
    dataset = pd.read_csv(csv_file)

    # split dataset to train-val splits
    text = dataset['Description']
    labels = dataset.drop(columns=['Description'])
    train_texts, val_texts, train_labels, val_labels = train_test_split(text, labels, train_size = train_split_ratio)

    # encode the text
    if model_type == 'bert-base-uncased':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    if model_type == 'distilbert-base-uncased':
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    if model_type == 'albert-base-v2':
        tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')

    if model_type == 'roberta-base':
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base') 
    
    train_encodings = tokenizer(list(train_texts), truncation=True, padding=True)
    val_encodings = tokenizer(list(val_texts), truncation=True, padding=True)

    return train_encodings, train_labels, val_encodings, val_labels



class TextToAttrDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(list(self.labels.iloc[idx]))
        return item

    def __len__(self):
        return len(self.labels)
