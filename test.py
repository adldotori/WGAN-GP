import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import argparse
import numpy as np
from PIL import Image
import imageio

from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from model import *
from loss import *
from dataloader import *

torch.manual_seed(17)

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num-workers', type=int, default = 4)
    parser.add_argument('-e', '--epoch', type=int, default=40)
    parser.add_argument('-b', '--batch-size', type=int, default = 100)
    parser.add_argument('-d', '--display-step', type=int, default = 600)
    opt = parser.parse_args()
    return opt

def imsave(result,path):
    img = result[0] * 255
    img = img.cpu().clamp(0,255)
    img = img.detach().numpy().astype('uint8')
    Image.fromarray(img).save(path)

def test(opt):
    video = []
    for i in range(33):
        generator = Generator().cuda()
        generator.load_state_dict(torch.load('checkpoint_{}.pt'.format((i+1) * 600)))
        generator.train()

        # Test
        z = Variable(torch.randn(100, 100)).cuda()
        sample_images = generator(z)
        grid = make_grid(sample_images, nrow=10, normalize=True)
        imsave(grid, 'result.png')
        video.append(imageio.imread('result.png'))

    imageio.mimsave('result.gif', video, fps=3)

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    opt = get_opt()
    test(opt)