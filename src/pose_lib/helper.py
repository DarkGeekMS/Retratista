import numpy as np
from skimage import transform as trans
import cv2

def affine_align(img, landmark=None, **kwargs):
    M = None
    src = np.array([
     [38.2946, 51.6963],
     [73.5318, 51.5014],
     [56.0252, 71.7366],
     [41.5493, 92.3655],
     [70.7299, 92.2041] ], dtype=np.float32 )
    src=src * 224 / 112

    dst = landmark.astype(np.float32)
    tform = trans.SimilarityTransform()
    tform.estimate(dst, src)
    M = tform.params[0:2,:]
    warped = cv2.warpAffine(img, M, (224, 224), borderValue = 0.0)
    return warped


def landmark_68_to_5(t68):
    le = t68[36:42, :].mean(axis=0, keepdims=True)
    re = t68[42:48, :].mean(axis=0, keepdims=True)
    no = t68[31:32, :]
    lm = t68[48:49, :]
    rm = t68[54:55, :]
    t5 = np.concatenate([le, re, no, lm, rm], axis=0)
    t5 = t5.reshape(10)
    return t5

def create_path(a_path, b_path):
    name_id_path = os.path.join(a_path, b_path)
    if not os.path.exists(name_id_path):
        os.makedirs(name_id_path)
    return name_id_path