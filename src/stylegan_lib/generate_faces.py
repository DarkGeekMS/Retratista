from stylegan2_networks import SynthesisNetwork
from text_processing.inference import BERTMultiLabelClassifier
from utils.seed_generation import generate_seed
from utils.latent_manipulation import manipulate_latent
from utils.text_postprocessing import postprocess_text_logits

import torch
import numpy as np
import skimage.io as io
import coloredlogs
import logging
import argparse
import json
import os


def generate_faces(config):
    # generate face images from text descriptions using directions-based latent navigation
    # initialize CUDA device
    device = torch.device('cuda')

    # initialize debug logger
    logger = logging.getLogger(__name__)
    coloredlogs.install(level='DEBUG', logger=logger)

    # read text descriptions
    print('Reading required text descriptions ...\n')
    sent_list = []
    with open(config['text_desc']) as f:
        for line in f:
            line = line.strip()
            sent_list.append(line)
    
    # initialize BERT multi-label classifier model
    print('Initializing BERT classifier ...\n')
    bert_model = BERTMultiLabelClassifier()

    # initialize StyleGAN2 generator
    print('Initializing StyleGAN2 generator ...\n')
    stylegan2_generator = SynthesisNetwork(w_dim=512, img_resolution=1024, img_channels=3)
    stylegan2_generator.load_state_dict(torch.load(config['stylegan_pt']))
    stylegan2_generator.to(device)

    # read pre-defined feature directions
    print('Reading feature directions ...\n')
    feature_directions = np.load(config['directions_npy'])

    # read initial seed latent vectors
    print('Reading initial seed latent vectors ...\n')
    seed_latent_vectors = np.load(config['initial_seed_npy'])

    # loop over each text description to generate the corresponding face image
    print('Starting face generation from text ...\n')
    for idx, sent in enumerate(sent_list):
        # print face id
        print(f'Face ID : {idx}')

        # extract logits from text
        print('Extracting attributes values from text ...')
        sent_pred = bert_model.predict(sent)
        text_logits = postprocess_text_logits(sent_pred, config['axes_range'])
        # DEBUG : print debug features
        if config['debug_mode']:
            logger.debug(f'Sent[{idx}] :')
            for logit_idx in config['debug_features']:
                logger.debug(f'[{logit_idx}] : {text_logits[logit_idx]}')
        
        # generate a random seed of extended latent vector and corresponding logits
        print('Generating initial seed ...')
        latent_vector, image_logits = generate_seed(seed_latent_vectors, feature_directions)
        # DEBUG : print debug features and save initial face image
        if config['debug_mode']:
            logger.debug(f'Random vector[{idx}] :')
            for logit_idx in config['debug_features']:
                logger.debug(f'[{logit_idx}] : {image_logits[logit_idx]}')
            logger.debug('Saving initial face image ...')
            w_tensor = torch.tensor(np.expand_dims(latent_vector, axis=0), device=device)
            initial_face_image = stylegan2_generator(w_tensor, noise_mode='const')
            initial_face_image = initial_face_image.permute(0, 2, 3, 1).cpu().detach().numpy()[0]
            initial_face_image[initial_face_image < -1.0] = -1.0
            initial_face_image[initial_face_image > 1.0] = 1.0
            initial_face_image = (initial_face_image + 1.0) * 127.5
            initial_face_image = initial_face_image.astype(np.uint8)
            io.imsave(f'results/{idx}_init.png', initial_face_image)
        
        # manipulate latent space to get the target latent vector
        print('Performing latent manipulation ...')
        target_latent = manipulate_latent(
            latent_vector, image_logits, text_logits, feature_directions, recalculate=config["recalculate_logits"]
        )
        target_latent = np.expand_dims(target_latent, axis=0)
        # DEBUG : print debug features
        if config['debug_mode']:
            logger.debug(f'Final vector[{idx}] :')
            for logit_idx in config['debug_features']:
                components = []
                for layer_idx in range(feature_directions[logit_idx].shape[0]):
                    unit_direction = np.divide(
                        feature_directions[logit_idx][layer_idx], np.sqrt(np.dot(feature_directions[logit_idx][layer_idx],
                        feature_directions[logit_idx][layer_idx]))
                    )
                    components.append(np.dot(target_latent[0][layer_idx], unit_direction))
                logger.debug(f'[{logit_idx}] : {sum(components) / len(components)}')
        
        # generate the required face image
        print('Performing face generation ...')
        w_tensor = torch.tensor(target_latent, device=device)
        face_image = stylegan2_generator(w_tensor, noise_mode='const')
        face_image = face_image.permute(0, 2, 3, 1).cpu().detach().numpy()[0]
        face_image[face_image < -1.0] = -1.0
        face_image[face_image > 1.0] = 1.0
        face_image = (face_image + 1.0) * 127.5
        face_image = face_image.astype(np.uint8)

        # save the generated face image
        print('Saving output face image ...')
        io.imsave(f'results/{idx}.png', face_image)
        print('\n-------------------------------------------------------------\n')


if __name__ == "__main__":
    # arguments parsing
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-cfg', '--config', type=str, help='path to config file of pipeline parameters', default='configs/generation_config.json'
    )

    args = argparser.parse_args()

    # read JSON config
    with open(args.config) as f:
        config = json.load(f)

    # folders creation
    if not os.path.isdir('results'):
        os.mkdir('results')
    
    # run face generation function
    print('################################################################\n')
    print('########## DIRECTIONS-BASED FACE GENERATION FROM TEXT ##########\n')
    print('################################################################\n')
    generate_faces(config)
