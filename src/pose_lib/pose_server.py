#!/usr/bin/env python3
# coding: utf-8

import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import os
import math
from tqdm import tqdm
import time
import face_alignment
from src.pose_lib.cyclegan_gen.generator import Generator
from .utils.ddfa import ToTensorGjz, NormalizeGjz, str2bool
import scipy.io as sio
from .utils.inference import parse_roi_box_from_landmark, crop_img, predict_68pts, parse_roi_box_from_bbox, get_5lmk_from_68lmk
from .utils.params import param_mean, param_std
from . import data
from .data.data_utils import get_multipose_test_input
from .options.test_options import TestOptions
from .models.test_model import TestModel
from .util import util
import time
from .models.networks.rotate_render import TestRender
import math
import matplotlib.pyplot as plt
from .helper import affine_align, landmark_68_to_5, create_path
import torchvision.models as t_models
import torch.nn as nn


class PoseServer:
    def __init__(self):
        model = t_models.mobilenet_v2(pretrained=False)
        classifier_layers = list(model.classifier[:-1])
        output_layer = nn.Linear(in_features=1280, out_features=62, bias=True)
        classifier_layers.append(output_layer)
        model.classifier = nn.Sequential(*classifier_layers)
        checkpoint_fp = 'src/pose_lib/3d_fitting.pth.tar'
        checkpoint = torch.load(checkpoint_fp, map_location=lambda storage, loc: storage)['state_dict']
        model_dict = model.state_dict()
        for k in checkpoint.keys():
            model_dict[k.replace('module.', '')] = checkpoint[k]
        model.load_state_dict(model_dict)

        model.eval()

        self.alignment_model = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device="cpu")

        self.face_model_generator = model

        self.opt = TestOptions()
        self.opt = self.opt.parse()
        self.opt.no_gaussian_landmark = True
        self.opt.label_mask = True
        self.opt.align = True
        self.opt.batchSize=1
        self.opt.isTrain = False


        self.opt.resnet_n_downsample = 4
        self.opt.resnet_n_blocks = 9
        self.opt.resnet_kernel_size=3
        self.opt.resnet_initial_kernel_size=7

        self.render_layer = TestRender(self.opt)

        self.opt.name = "rs_model"
        self.model = TestModel(self.opt)
        self.model.eval()
        self.model = self.model.cuda()
                        


    def prepare_rotation(self, angle):
        
        self.opt.yaw_poses = [math.radians(angle)]
        self.data_info = data.dataset_info()
        self.datanum = self.data_info.get_dataset(self.opt)[0]
        self.folderlevel = self.data_info.folder_level[self.datanum]

        self.dataloader = data.create_dataloader_test(self.opt)


    def generate_3d_model(self, img):

        landmark_list = []


        transform = transforms.Compose([ToTensorGjz(), NormalizeGjz(mean=127.5, std=128)])


        img_ori = img

        cv2.imwrite("src/pose_lib/face_data/Images/target.jpg", img)

        pts_res = []
        ind = 0

        preds = self.alignment_model.get_landmarks(img_ori[:, :, ::-1])
        pts_2d_68 = preds[0]
        pts_2d_5 = get_5lmk_from_68lmk(pts_2d_68)
        landmark_list.append(pts_2d_5)
        roi_box = parse_roi_box_from_landmark(pts_2d_68.T)

        img = crop_img(img_ori, roi_box)

        img = cv2.resize(img, dsize=(120, 120), interpolation=cv2.INTER_LINEAR)
        input = transform(img).unsqueeze(0)

        with torch.no_grad():
            param = self.face_model_generator(input)
            param = param.squeeze().cpu().numpy().flatten().astype(np.float32)

        pts68 = predict_68pts(param, roi_box)

        roi_box = parse_roi_box_from_landmark(pts68)
        img_step2 = crop_img(img_ori, roi_box)
        img_step2 = cv2.resize(img_step2, dsize=(120, 120), interpolation=cv2.INTER_LINEAR)
        input = transform(img_step2).unsqueeze(0)

        with torch.no_grad():
            param = self.face_model_generator(input)
            param = param.squeeze().cpu().numpy().flatten().astype(np.float32)


        #Export parameters

        save_name = "src/pose_lib/face_data/params/target.txt"
        this_param = param * param_std + param_mean
        this_param = np.concatenate((this_param, roi_box))
        this_param.tofile(save_name, sep=' ')

        # #Export landmarks

        save_path = "src/pose_lib/face_data/realign_lmk"

        with open(save_path, 'w') as f:
            # f.write('{} {} {} {}')
            land = np.array(landmark_list[0])
            land = land.astype(np.int)
            land_str = ' '.join([str(x) for x in land])
            msg = f'target.jpg 0 {land_str}\n'
            f.write(msg)

        print("Created 3d model")


    def rotate_face(self, img, angle, reuse=False):

        if not reuse:
            self.generate_3d_model(img)

        self.prepare_rotation(angle)


        data = [x for x in self.dataloader][0]

        data = get_multipose_test_input(data, self.render_layer, self.opt.yaw_poses, [])

        rotated_landmarks = data['rotated_landmarks'][:, :, :2].cpu().numpy().astype(np.float)

        generate_rotated = self.model.forward(data)

        rotated_keypoints = landmark_68_to_5(rotated_landmarks[0])

        image_numpy = util.tensor2im(generate_rotated[0])

        warped = affine_align(image_numpy, rotated_keypoints.reshape(5, 2))

        print("Completed processing !!")

        return warped

