import torch
from . import networks
from ..util import util
from ..data import curve
import numpy as np
import os
from .rotatespade_model import RotateSPADEModel


class TestModel(RotateSPADEModel):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        networks.modify_commandline_options(parser, is_train)
        return parser

    def __init__(self, opt):
        super(TestModel, self).__init__(opt)


    def forward(self, data):
        real_image = data['image'].cuda()
        rotated_landmarks = data['rotated_landmarks']
        original_angles = data['original_angles']
        self.rotated_seg, rotated_seg_all = \
            self.get_seg_map(rotated_landmarks, self.opt.no_gaussian_landmark, self.opt.crop_size, original_angles)
        rotated_mesh = data['rotated_mesh'].cuda(0)
        if self.opt.label_mask:
            rotated_mesh = (rotated_mesh + rotated_seg_all[:, 4].unsqueeze(1) + rotated_seg_all[:, 0].unsqueeze(1))
            rotated_mesh[rotated_mesh >= 1] = 0
        with torch.no_grad():
            fake_rotate = self.generate_fake(rotated_mesh, real_image, self.rotated_seg)

        return fake_rotate


