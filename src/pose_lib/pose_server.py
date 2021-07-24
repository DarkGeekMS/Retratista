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
from .utils.ddfa import ToTensorGjz, NormalizeGjz, str2bool
import scipy.io as sio
from .utils.inference import parse_roi_box_from_landmark, crop_img, predict_68pts, predict_dense, parse_roi_box_from_bbox, get_colors, get_5lmk_from_68lmk
from .utils.estimate_pose import parse_pose
from .utils.params import param_mean, param_std
from .utils.render import get_depths_image, cget_depths_image, crender_colors
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
from . import mobilenet_v1


class PoseServer:
    def __init__(self):
        arch = "mobilenet_1"
        model = getattr(mobilenet_v1, arch)(num_classes=62)  # 62 = 12(pose) + 40(shape) +10(expression)
        checkpoint_fp = 'src/pose_lib/phase1_pdc.pth.tar'
        checkpoint = torch.load(checkpoint_fp, map_location=lambda storage, loc: storage)['state_dict']
        model_dict = model.state_dict()
        # because the model is trained by multiple gpus, prefix module should be removed
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

        self.dataloaders = data.create_dataloader_test(self.opt)


    def generate_3d_model(self, img):

        landmark_list = []


        tri = sio.loadmat('src/pose_lib/visualize/tri.mat')['tri']
        transform = transforms.Compose([ToTensorGjz(), NormalizeGjz(mean=127.5, std=128)])


        img_ori = img

        cv2.imwrite("src/pose_lib/face_data/Images/target.jpg", img)

        pts_res = []
        Ps = []  
        poses = [] 
        vertices_lst = [] 
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

        pts68 = predict_68pts(param, roi_box)

        pts_res.append(pts68)
        P, pose = parse_pose(param)
        Ps.append(P)
        poses.append(pose)

        # dense face 3d vertices
        vertices = predict_dense(param, roi_box)

        wfp_2d_img = "target.png"
        colors = get_colors(img_ori, vertices)
        # h, w, c = 120, 120, 3
        h, w, c = img_ori.shape
        img_2d = crender_colors(vertices.T, (tri - 1).T, colors[:, ::-1], h, w)
        cv2.imwrite(wfp_2d_img, img_2d[:, :, ::-1])

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


        for data in self.dataloaders[0]:
            data = get_multipose_test_input(data, self.render_layer, self.opt.yaw_poses, [])

            img_path = data['path']
            poses = data['pose_list']
            rotated_landmarks = data['rotated_landmarks'][:, :, :2].cpu().numpy().astype(np.float)
            rotated_landmarks_106 = data['rotated_landmarks_106'][:, :, :2].cpu().numpy().astype(np.float)


            generate_rotated = self.model.forward(data, mode='single')

            for b in range(generate_rotated.shape[0]):
                rotated_keypoints = landmark_68_to_5(rotated_landmarks[b])

                image_numpy = util.tensor2im(generate_rotated[b])


                warped = affine_align(image_numpy, rotated_keypoints.reshape(5, 2))

                print("Completed processing !!")

                return warped

