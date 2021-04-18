from stylegan2_networks import SynthesisNetwork
from text_processing.inference import BERTMultiLabelClassifier
from utils import generate_seed, manipulate_latent, postprocess_text_logits

import torch
import numpy as np


class StyleGANServer:
    """
    StyleGAN2 Server
    ...

    Main StyleGAN2 API for :
    1) Face Generation from text / values.
    2) Face Refinement.
    ...
    """
    def __init__(self):
        # initialize server parameters
        # define CUDA device
        self.device = torch.device('cuda')
        # set latent manipulation parameters
        self.axes_range = 4.0
        self.recalc_logits = True
        # load BERT multi-label classifier
        self.bert_model = BERTMultiLabelClassifier()
        # load StyleGAN2 synthesis network
        self.stylegan2_generator = SynthesisNetwork(
            w_dim=512, img_resolution=1024, img_channels=3
        )
        self.stylegan2_generator.load_state_dict(
            torch.load("models/stgan2_model.pt")
        )
        self.stylegan2_generator.to(self.device)
        # load feature directions
        self.attributes_dir = np.load("directions/attributes_directions.npy")
        self.morph_dir = np.load("directions/morph_directions.npy")
        # load seed latent vectors
        self.latent_seed = np.load("models/initial_seed.npy")

    def process_text(self, sent):
        # extract attributes values from text
        # run BERT inference on text
        sent_pred = self.bert_model.predict(sent)
        # re-scale output values from text
        text_logits = postprocess_text_logits(sent_pred, self.axes_range)
        # return extracted logits
        return text_logits

    def generate_face(self, values):
        # generate face from attributes values
        # generate initial latent seed
        latent_vector, image_logits = generate_seed(
            self.latent_seed, self.attributes_dir
        )
        # perform latent manipulation
        target_latent = manipulate_latent(
            latent_vector, image_logits, values,
            self.attributes_dir, recalculate=self.recalc_logits
        )
        target_latent = np.expand_dims(target_latent, axis=0)
        # convert resultant latent vector into torch tensor
        w_tensor = torch.tensor(target_latent, device=self.device)
        # run StyleGAN2 synthesis network on final tensor
        face_image = self.stylegan2_generator(w_tensor, noise_mode='const')
        # post-process final face image
        face_image = face_image.permute(0, 2, 3, 1).cpu().detach().numpy()[0]
        face_image[face_image < -1.0] = -1.0
        face_image[face_image > 1.0] = 1.0
        face_image = (face_image + 1.0) * 127.5
        face_image = face_image.astype(np.uint8)
        # return final face image
        return face_image

    def refine_face(self, offsets):
        # refine generated face using given attributes offsets
        pass
