import torch
import pickle
import numpy as np
from transformers import DistilBertTokenizer, AlbertTokenizer, RobertaTokenizer, BertTokenizer
from .model import BertRegressor
class TextProcessor():
    def __init__(self, architecture, checkpoint_path = None):
        # device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # model
        if checkpoint_path is None:
            checkpoint_path = 'Bert/checkpoints/' + architecture + '.pth'

        self.model = BertRegressor(architecture).to(self.device)
        self.model.load_state_dict(torch.load(checkpoint_path, self.device)) 
        # model.load_state_dict(copy.deepcopy(torch.load("model_state.pth",device)))
        # tokenizer
        if architecture == 'bert-base-uncased':
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        if architecture == 'distilbert-base-uncased':
            self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

        if architecture == 'albert-base-v2':
            self.tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')

        if architecture == 'roberta-base':
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base') 

        # normalization post processing details
        with open('Bert/attributes_max.pkl', 'rb') as f:
            self.attributes_max_values = pickle.load(f)
        self.zero_start_attributes = [
            'Arched_Eyebrows',
            'Bushy_Eyebrows',
            'Straight_Hair',
            'Mustache',
            'Beard',
            'Skin_Color',
            'Chubby',
            'Male',
            'Old',
            'Wide_Eyes',
            'Big_Lips',
            'Big_Nose',
            'Big_Ears',
            'Wearing_Lipstick'
        ]
        self.nonzero_start_attributes = [
            'Black_Hair',
            'Blond_Hair',
            'Brown_Hair',
            'Gray_Hair',
            'Red_Hair',
            'Receding_Hairline',
            'Bald',
            'Bangs',
            'Hair_Length',
            'Goatee',
            'Sideburns',
            'Asian',
            'Bags_Under_Eyes',
            'Black_Eyes',
            'Green_Eyes',
            'Blue_Eyes',
            'Brown_Eyes',
            'Double_Chin',
            'High_Cheekbones',
            'Pointy_Nose',
            'Rosy_Cheeks',
            'Heavy_Makeup',
            'Wearing_SightGlasses',
            'Wearing_SunGlasses'
        ]

    
    def make_logits(self, logits):
        all_logits_mod_list = []
        for log in logits:
            attributes = list(self.attributes_max_values.keys())

            logits_mod = {attributes[i]: log[i] for i in range(len(attributes))} 

            # print()
            # print()
            # for key in logits_mod.keys():
            #     print(key, ':', logits_mod[key])

            # round on glasses
            logits_mod['Wearing_SightGlasses'] = np.round(logits_mod['Wearing_SightGlasses'])
            logits_mod['Wearing_SunGlasses'] = np.round(logits_mod['Wearing_SunGlasses'])

            # option 1 - zero start
            for zs_attr in self.zero_start_attributes:
                # not mentioned
                if logits_mod[zs_attr] < 0.5:
                    if zs_attr != 'Male':
                        logits_mod[zs_attr] = -1 
                    else:
                        logits_mod[zs_attr] = 0
                # nothing else should be negative
                elif logits_mod[zs_attr] <= 1:
                    if zs_attr != 'Male':
                        logits_mod[zs_attr] = 0
                    else:
                        logits_mod[zs_attr] = 1
                # mentioned -> scale
                else:
                    if zs_attr == 'Male':
                        logits_mod[zs_attr] = 1
                    else:
                        logits_mod[zs_attr] -= 1
                        logits_mod[zs_attr] = logits_mod[zs_attr] / (self.attributes_max_values[zs_attr]-2)
                
            # option 2 - non-zero start 
            for nzs_attr in self.nonzero_start_attributes:
                # not mentioned
                if logits_mod[nzs_attr] < 0.5:
                    logits_mod[nzs_attr] = -1 
                # mentioned -> scale
                else:
                    logits_mod[nzs_attr] = logits_mod[nzs_attr] / (self.attributes_max_values[nzs_attr] - 1)

            # clip on 1 max
            for key in logits_mod.keys():
                if logits_mod[key] > 1:
                    logits_mod[key] = 1

            logits_mod_list = list(logits_mod.values())
            all_logits_mod_list.append(logits_mod_list)

        # print()
        # print()
        # for key in logits_mod.keys():
        #     print(key, ':', logits_mod[key])

        return all_logits_mod_list

    
            
    def predict(self, sentence):
        # encode the sentence
        encodings = self.tokenizer([sentence], truncation=True, padding=True)
        input_ids = torch.tensor(encodings['input_ids']).to(self.device)
        attention_mask = torch.tensor(encodings['attention_mask']).to(self.device)
        logits = self.model(input_ids, attention_mask=attention_mask).cpu().data.numpy()
        logits = self.make_logits(logits)
        return logits

# processor = TextProcessor('./checkpoints/distilbert-base-uncased.pkl', 'distilbert-base-uncased')
# processor.predict('a guy with long hair and sunglasses.')



