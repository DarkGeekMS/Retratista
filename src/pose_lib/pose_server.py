#!/usr/bin/env python3
# coding: utf-8

__author__ = 'cleardusk'

"""
The pipeline of 3DDFA prediction: given one image, predict the 3d face vertices, 68 landmarks and visualization.

[todo]
1. CPU optimization: https://pmchojnacki.wordpress.com/2018/10/07/slow-pytorch-cpu-performance
"""

import torch
import torchvision.transforms as transforms
from . import mobilenet_v1
import numpy as np
import cv2
import os
import math
from tqdm import tqdm
import time
import face_alignment
from .utils.ddfa import ToTensorGjz, NormalizeGjz, str2bool
import scipy.io as sio
from .utils.inference import get_suffix, parse_roi_box_from_landmark, crop_img, predict_68pts, dump_to_ply, dump_vertex, \
    draw_landmarks, predict_dense, parse_roi_box_from_bbox, get_colors, write_obj_with_colors, get_aligned_param, get_5lmk_from_68lmk
from .utils.cv_plot import plot_pose_box
from .utils.estimate_pose import parse_pose
from .utils.params import param_mean, param_std
from .utils.render import get_depths_image, cget_depths_image, cpncc, crender_colors
from .utils.paf import gen_img_paf
import argparse
import torch.backends.cudnn as cudnn
import torch.multiprocessing as multiprocessing
from .models.networks.sync_batchnorm import DataParallelWithCallback
import sys
from . import data
from .data.data_utils import get_multipose_test_input
from .util.iter_counter import IterationCounter
from .options.test_options import TestOptions
from .models.test_model import TestModel
from .util.visualizer import Visualizer
from .util import html, util
from torch.multiprocessing import Process, Queue, Pool
from .data.data_utils import data_prefetcher
from skimage import transform as trans
import time
from .models.networks.rotate_render import TestRender
import math
import matplotlib.pyplot as plt



STD_SIZE = 120

def create_path(a_path, b_path):
    name_id_path = os.path.join(a_path, b_path)
    if not os.path.exists(name_id_path):
        os.makedirs(name_id_path)
    return name_id_path


def create_paths(save_path, img_path, foldername='orig', folderlevel=2, pose='0'):
    save_rotated_path_name = create_path(save_path, foldername)

    path_split = img_path.split('/')
    rotated_file_savepath = save_rotated_path_name
    for level in range(len(path_split) - folderlevel, len(path_split)):
        file_name = path_split[level]
        if level == len(path_split) - 1:
            file_name = str(pose) + '_' + file_name
        rotated_file_savepath = os.path.join(rotated_file_savepath, file_name)
    return rotated_file_savepath

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


def save_img(img, save_path):
    image_numpy = util.tensor2im(img)
    util.save_image(image_numpy, save_path, create_dir=True)
    return image_numpy


class PoseServer:
    def __init__(self):
        arch = "mobilenet_1"
        model = getattr(mobilenet_v1, arch)(num_classes=62)  # 62 = 12(pose) + 40(shape) +10(expression)
        checkpoint_fp = 'phase1_pdc.pth.tar'
        checkpoint = torch.load(checkpoint_fp, map_location=lambda storage, loc: storage)['state_dict']
        model_dict = model.state_dict()
        # because the model is trained by multiple gpus, prefix module should be removed
        for k in checkpoint.keys():
            model_dict[k.replace('module.', '')] = checkpoint[k]
        model.load_state_dict(model_dict)
        #cudnn.benchmark = True
        model.eval()

        self.alignment_model = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device="cpu")

        self.face_model_generator = model

        self.opt = TestOptions()
        self.opt = self.opt.parse()
        self.opt.names = "rs_model"
        self.opt.dataset = "example" 
        self.opt.list_start = 0
        self.opt.list_end = 0
        self.opt.dataset_mode = "allface" 
        self.opt.gpu_ids = 0
        self.opt.netG = "rotatespade"
        self.opt.norm_G = "spectralsyncbatch"
        self.opt.model = "rotatespade" 
        self.opt.label_nc = 5
        self.opt.nThreads = 3
        self.opt.heatmap_size = 2.5
        self.opt.chunk_size = 1
        self.opt.no_gaussian_landmark = True
        self.opt.device_count = 1
        self.opt.render_thread = 1
        self.opt.label_mask = True
        self.opt.align = True
        self.opt.erode_kernel = 21 
        self.opt.batchSize=1
        self.opt.isTrain = False


        self.opt.resnet_n_downsample = 4
        self.opt.resnet_n_blocks = 9
        self.opt.resnet_kernel_size=3
        self.opt.resnet_initial_kernel_size=7

        ngpus = self.opt.device_count

        self.render_gpu_ids = list(range(ngpus - self.opt.render_thread, ngpus))
        self.render_layer = TestRender(self.opt)
        self.render_layer_list = [self.render_layer]

        self.opt.gpu_ids = [0]
        print('Testing gpu ', self.opt.gpu_ids)
        if self.opt.names is None:
            self.model = TestModel(self.opt)
            self.model.eval()
            self.model = torch.nn.DataParallel(model.cuda(),
                                        device_ids=self.opt.gpu_ids,
                                        output_device=self.opt.gpu_ids[-1],
                                        )
            self.models = [self.model]
            self.names = [self.opt.name]
            self.save_path = create_path(create_path(self.opt.save_path, self.opt.name), self.opt.dataset)
            self.save_paths = [save_path]
            self.f = [open(
                    os.path.join(self.save_path, self.opt.dataset + str(self.opt.list_start) + str(self.opt.list_end) + '_rotate_lmk.txt'), 'w')]
        else:
            self.models = []
            self.names = []
            self.save_paths = []
            self.f = []
            for name in self.opt.names.split(','):
                self.opt.name = name
                self.model = TestModel(self.opt)
                self.model.eval()
                self.model = torch.nn.DataParallel(self.model.cuda(),
                                            device_ids=self.opt.gpu_ids,
                                            output_device=self.opt.gpu_ids[-1],
                                            )
                self.models.append(self.model)
                self.names.append(name)
                self.save_path = create_path(create_path(self.opt.save_path, self.opt.name), self.opt.dataset)
                self.save_paths.append(self.save_path)
                self.f_rotated = open(
                    os.path.join(self.save_path, self.opt.dataset + str(self.opt.list_start) + str(self.opt.list_end) + '_rotate_lmk.txt'), 'w')
                self.f.append(self.f_rotated)



    def prepare_rotation(self, angle):
        
        self.opt.yaw_poses = [math.radians(angle)]
        self.data_info = data.dataset_info()
        self.datanum = self.data_info.get_dataset(self.opt)[0]
        self.folderlevel = self.data_info.folder_level[self.datanum]

        self.dataloaders = data.create_dataloader_test(self.opt)

        self.visualizer = Visualizer(self.opt)
        self.iter_counter = IterationCounter(self.opt, len(self.dataloaders[0]) * self.opt.render_thread)
        #self.test_tasks = init_parallel_jobs(self.testing_queue, self.dataloaders, self.iter_counter, self.opt, self.render_layer_list)


        # create a webpage that summarizes the all results  
    def generate_3d_model(self, img):
        # 1. load pre-tained model

        landmark_list = []


        tri = sio.loadmat('src/pose_lib/visualize/tri.mat')['tri']
        transform = transforms.Compose([ToTensorGjz(), NormalizeGjz(mean=127.5, std=128)])

        # 2. parse images list 

        img_ori = img

        cv2.imwrite("src/pose_lib/face_data/Images/target.jpg", img)

        pts_res = []
        Ps = []  # Camera matrix collection
        poses = []  # pose collection, [todo: validate it]
        vertices_lst = []  # store multiple face vertices
        ind = 0

        # face alignment model use RGB as input, result is a tuple with landmarks and boxes
        preds = self.alignment_model.get_landmarks(img_ori[:, :, ::-1])
        pts_2d_68 = preds[0]
        pts_2d_5 = get_5lmk_from_68lmk(pts_2d_68)
        landmark_list.append(pts_2d_5)
        roi_box = parse_roi_box_from_landmark(pts_2d_68.T)

        img = crop_img(img_ori, roi_box)
        # import pdb; pdb.set_trace()


        # forward: one step
        img = cv2.resize(img, dsize=(STD_SIZE, STD_SIZE), interpolation=cv2.INTER_LINEAR)
        input = transform(img).unsqueeze(0)
        with torch.no_grad():
            param = self.face_model_generator(input)
            param = param.squeeze().cpu().numpy().flatten().astype(np.float32)

        # 68 pts
        pts68 = predict_68pts(param, roi_box)

        roi_box = parse_roi_box_from_landmark(pts68)
        img_step2 = crop_img(img_ori, roi_box)
        img_step2 = cv2.resize(img_step2, dsize=(STD_SIZE, STD_SIZE), interpolation=cv2.INTER_LINEAR)
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
        # aligned_param = get_aligned_param(param)
        # vertices_aligned = predict_dense(aligned_param, roi_box)
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


    def rotate_face(self, img, angle, reuse=False):

        if not reuse:
            self.generate_3d_model(img)

        self.prepare_rotation(angle)

        # test
        landmarks = []

        process_num = self.opt.list_start
        first_time = time.time()
        try:
            for data in self.dataloaders[0]:
                start_time = time.time()
                data = get_multipose_test_input(data, self.render_layer, self.opt.yaw_poses, [])

                current_time = time.time()
                time_per_iter = (current_time - start_time) / self.opt.batchSize
                message = '(************* each image render time: %.3f *****************) ' % (time_per_iter)
                print(message)

                img_path = data['path']
                poses = data['pose_list']
                rotated_landmarks = data['rotated_landmarks'][:, :, :2].cpu().numpy().astype(np.float)
                rotated_landmarks_106 = data['rotated_landmarks_106'][:, :, :2].cpu().numpy().astype(np.float)


                generate_rotateds = []
                for model in self.models:
                    generate_rotated = model.forward(data, mode='single')
                    generate_rotateds.append(generate_rotated)

                for n, name in enumerate(self.names):
                    self.opt.name = name
                    for b in range(generate_rotateds[n].shape[0]):
                        # get 5 key points
                        rotated_keypoints = landmark_68_to_5(rotated_landmarks[b])
                        # get savepaths
                        rotated_file_savepath = create_paths(self.save_paths[n], img_path[b], folderlevel=self.folderlevel, pose=poses[b])

                        image_numpy = save_img(generate_rotateds[n][b], rotated_file_savepath)
                        rotated_keypoints_str = rotated_file_savepath + ' 1 ' + ' '.join([str(int(n)) for n in rotated_keypoints]) + '\n'
                        print('process image...' + rotated_file_savepath)
                        self.f[n].write(rotated_keypoints_str)

                        current_time = time.time()
                        if n == 0:
                            if b <= self.opt.batchSize:
                                process_num += 1
                            print('processed num ' + str(process_num))
                        if self.opt.align:
                            aligned_file_savepath = create_paths(self.save_paths[n], img_path[b], 'aligned', folderlevel=self.folderlevel, pose=poses[b])
                            warped = affine_align(image_numpy, rotated_keypoints.reshape(5, 2))
                            util.save_image(warped, aligned_file_savepath, create_dir=True)

                        # save 106 landmarks
                        rotated_keypoints_106 = rotated_landmarks_106[b] # shape: 106 * 2


                current_time = time.time()
                time_per_iter = (current_time - start_time) / self.opt.batchSize
                message = '(************* each image time total: %.3f *****************) ' % (time_per_iter)
                print(message)

                return warped

        except Exception as e:
            print(e)
            pass

