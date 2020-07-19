import os
import torch
import argparse

from options.base_options import BaseOptions


class TrainOptions(BaseOptions):
    """Define options for training and test"""

    def __init__(self):
        super(TrainOptions, self).__init__()

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)

        parser.add_argument('--name', type=str, help='network cfig name')
        parser.add_argument('--mode', type=str, default='train', help='network cfig name')
        parser.add_argument('--train', type=bool, default=True, help='train or eval')

        # Dataset
        parser.add_argument('--dataset', type=str, default='brdf', help='diligent / synthetic / full')
        parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
        parser.add_argument('--batch_size', type=int, default=4, help='input batch size')
        parser.add_argument('--shuffle', type=bool, default=True, help='shuffle the dataset for training')
        parser.add_argument('--image_size', type=int, default=256, help='if load all images together')
        parser.add_argument('--light_c', type=int, default=3, help='if load all images together')
        
        # Model
        parser.add_argument('--cuda', type=bool, default=True, help='use cuda')
        parser.add_argument('--inc', type=int, default=3, help='input channels')
        parser.add_argument('--outc', type=int, default=3, help='output channels')
        parser.add_argument('--ngf',  type=int, default=64, help='ngf')
        parser.add_argument('--nepoch', type=int, default=[14, 10, 9], help='number of total epochs to train')
        parser.add_argument('--reuse', type=bool, default=False, help='reuse model')
        parser.add_argument('--start_epoch', type=int, default=1, help='#epoch to start')
        parser.add_argument('--gpu_id', type=int, default=[0, 1], help='#epoch to start')

        # Visualization and saving
        parser.add_argument('--outf', type=str, default='./samples', help='folder to output temp samples')

        return parser
