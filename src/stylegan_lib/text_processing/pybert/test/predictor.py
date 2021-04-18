#encoding:utf-8
import torch
import numpy as np
import pickle
from ..common.tools import model_device
from ..callback.progressbar import ProgressBar

class Predictor(object):
    def __init__(self,model,path_to_max_attributes,n_gpu):
        self.model = model
        self.model, self.device = model_device(n_gpu= n_gpu, model=self.model)
        with open(path_to_max_attributes, 'rb') as f:
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

    def predict(self,data):
        pbar = ProgressBar(n_total=len(data),desc='Testing')
        all_logits = None
        for step, batch in enumerate(data):
            self.model.eval()
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                input_ids, input_mask, segment_ids, label_ids = batch
                logits = self.model(input_ids, segment_ids, input_mask).detach().cpu().numpy()

                logits = self.make_logits(logits)


            if all_logits is None:
                all_logits = logits
            else:
                all_logits = np.concatenate([all_logits,logits],axis = 0)
            pbar(step=step)
        print()
        if 'cuda' in str(self.device):
            torch.cuda.empty_cache()
        all_logits = np.array(all_logits)
        return all_logits






