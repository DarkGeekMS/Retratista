from .stylegan2_networks import SynthesisNetwork
from .text_processing.inference import TextProcessor
from .utils import calculate_feature_components, generate_seed, manipulate_latent, postprocess_text_logits

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
        self.bert_model = TextProcessor('distilbert-base-uncased')
        # load StyleGAN2 synthesis network
        self.stylegan2_generator = SynthesisNetwork(
            w_dim=512, img_resolution=1024, img_channels=3
        )
        self.stylegan2_generator.load_state_dict(
            torch.load("src/stylegan_lib/models/stgan2_model.pt")
        )
        self.stylegan2_generator.to(self.device)
        # load feature directions
        self.gen_dir = np.load("src/stylegan_lib/directions/gen_directions.npy")
        self.refine_dir = np.load("src/stylegan_lib/directions/refine_directions.npy")
        self.pose_dir = np.expand_dims(
            np.load("src/stylegan_lib/directions/pose_directions.npy"), axis=0
        )
        # load seed latent vectors
        self.latent_seed = np.load("src/stylegan_lib/models/initial_seed.npy")
        # initialize latent vector store
        self.stored_latent, self.stored_values = generate_seed(
            self.latent_seed, self.refine_dir
        )

    def process_text(self, sent):
        # extract attributes values from text
        # run BERT inference on text
        sent_pred = self.bert_model.predict(sent)
        # re-scale output values from text
        text_logits = postprocess_text_logits(sent_pred, self.axes_range)
        # return extracted logits
        return text_logits

    def generate_face(self, values, rescale=False):
        # generate face from attributes values
        # re-scale input attributes values
        if rescale:
            values = np.array(
                [
                    (logit * self.axes_range * 2.0) - self.axes_range 
                    if logit != -1.0 else -100.0 for logit in values
                ]
            )
        # generate initial latent seed
        latent_vector, image_logits = generate_seed(
            self.latent_seed, self.gen_dir
        )
        # perform latent manipulation
        self.stored_latent = manipulate_latent(
            latent_vector, image_logits, values,
            self.gen_dir, recalculate=self.recalc_logits
        )
        w_vector = np.expand_dims(self.stored_latent, axis=0)
        # convert resultant latent vector into torch tensor
        w_tensor = torch.tensor(w_vector, device=self.device)
        # run StyleGAN2 synthesis network on final tensor
        face_image = self.stylegan2_generator(w_tensor, noise_mode='const')
        # post-process final face image
        face_image = face_image.permute(0, 2, 3, 1).cpu().detach().numpy()[0]
        face_image[face_image < -1.0] = -1.0
        face_image[face_image > 1.0] = 1.0
        face_image = (face_image + 1.0) * 127.5
        face_image = face_image.astype(np.uint8)
        # re-calculate final components for all features through projection
        self.stored_values = calculate_feature_components(self.stored_latent, self.refine_dir)
        # re-scale the final attributes values between 0 and 1
        values = (self.stored_values + self.axes_range) / (2.0 * self.axes_range) 
        # return final face image and corresponding attributes values
        return face_image, values

    def refine_face(self, values):
        # refine generated face using given attributes (or morph) offsets
        # re-scale the attribute offset
        values = np.array(
                [
                    (logit * self.axes_range * 2.0) - self.axes_range 
                    if logit != -1.0 else -100.0 for logit in values
                ]
            )
        # re-adjust the stored latent vector
        self.stored_latent = manipulate_latent(
            self.stored_latent, self.stored_values, values,
            self.refine_dir, recalculate=self.recalc_logits
        )
        w_vector = np.expand_dims(self.stored_latent, axis=0)
        # convert resultant latent vector into torch tensor
        w_tensor = torch.tensor(w_vector, device=self.device)
        # run StyleGAN2 synthesis network on final tensor
        face_image = self.stylegan2_generator(w_tensor, noise_mode='const')
        # post-process refined face image
        face_image = face_image.permute(0, 2, 3, 1).cpu().detach().numpy()[0]
        face_image[face_image < -1.0] = -1.0
        face_image[face_image > 1.0] = 1.0
        face_image = (face_image + 1.0) * 127.5
        face_image = face_image.astype(np.uint8)
        # re-calculate final components for all features through projection
        self.stored_values = calculate_feature_components(self.stored_latent, self.refine_dir)
        # re-scale the final attributes values between 0 and 1
        values = (self.stored_values + self.axes_range) / (2.0 * self.axes_range) 
        # return refined face image and corresponding attributes values
        return face_image, values

    def rotate_face(self, angle):
        # rotate generated face using given angle
        # re-scale given angle to range [-4, 4]
        angle = angle / (90.0 / 4.0)
        # get current face angle
        curr_angle = calculate_feature_components(self.stored_latent, self.pose_dir)[0]
        # differentiate face angle
        diff_angle = angle - curr_angle
        # move latent vector along pose direction
        w_vector = self.stored_latent + diff_angle * self.pose_dir[0]
        w_vector = np.expand_dims(w_vector, axis=0)
        # convert resultant latent vector into torch tensor
        w_tensor = torch.tensor(w_vector, device=self.device)
        # run StyleGAN2 synthesis network on final tensor
        face_image = self.stylegan2_generator(w_tensor, noise_mode='const')
        # post-process refined face image
        face_image = face_image.permute(0, 2, 3, 1).cpu().detach().numpy()[0]
        face_image[face_image < -1.0] = -1.0
        face_image[face_image > 1.0] = 1.0
        face_image = (face_image + 1.0) * 127.5
        face_image = face_image.astype(np.uint8)
        # return rotated face image
        return face_image
