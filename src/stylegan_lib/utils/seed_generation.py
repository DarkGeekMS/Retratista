import numpy as np

def calculate_feature_components(latent_vector, feature_directions):
    # calculate components for all features through projection
    # project latent vector onto all feature directions
    logits = []
    # loop over all feature directions
    for direction in feature_directions:
        components = []
        # project each layer independently
        for idx in range(direction.shape[0]):
            # get unit vector of direction for each layer (512)
            unit_direction = np.divide(direction[idx], np.sqrt(np.dot(direction[idx], direction[idx])))
            # project latent vector onto unit direction vector
            components.append(np.dot(latent_vector[idx], unit_direction))
        # average projected components of all layers
        avg_component = sum(components) / len(components)
        logits.append(avg_component)
    # return feature components of latent vector
    return np.array(logits)

def generate_seed(seed_latent_vectors, feature_directions):
    # generate a random seed of extended latent vector and corresponding logits
    # pick random vector from initial seed vectors
    w = seed_latent_vectors[np.random.randint(seed_latent_vectors.shape[0])][0]
    # calculate components for all features through projection
    logits = calculate_feature_components(w, feature_directions)
    # return random latent vector and its logits
    return w, logits
