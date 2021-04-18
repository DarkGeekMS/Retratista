from utils.seed_generation import calculate_feature_components

import numpy as np
import copy

def differentiate_logits(l_text, l_img):
    # differentiate between text and image labels, while keeping unspecified labels as zero
    l_diff = np.array([l1 - l2 if l1 != -100.0 else 0.0 for l1, l2 in zip(l_text, l_img)])
    # return differentiated logits
    return l_diff

def manipulate_latent(seed_latent_vec, seed_logits, text_logits, feature_directions, recalculate=False):
    # manipulate random latent vector based on predicted features
    # differentiate predicted logits
    logits_diff = differentiate_logits(text_logits, seed_logits)
    # loop over each feature and navigate the latent space
    final_latent_vec = copy.deepcopy(seed_latent_vec)
    for axis_idx in range(len(text_logits)):
        # navigate using differentiated logits
        latent_shift = feature_directions[axis_idx] * logits_diff[axis_idx]
        # apply latent shift of single feature
        final_latent_vec += latent_shift
        # check whether to recalculate feature components or not
        if recalculate:
            # re-calculate components for all features through projection
            seed_logits = calculate_feature_components(final_latent_vec, feature_directions)
            # re-differentiate predicted logits
            logits_diff = differentiate_logits(text_logits, seed_logits)
    return final_latent_vec
