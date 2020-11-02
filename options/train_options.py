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
        parser.add_argument('--train', type=bool, default=True, help='train or eval')

        # Datasets
        parser.add_argument('--workers', type=int, default=4, help='number of workers')
        parser.add_argument('--batch_size', type=int, default=4, help='input batch size')
        parser.add_argument('--shuffle', type=bool, default=True, help='if shuffle the dataset')
        parser.add_argument('--image_size', type=int, default=256, help='image size')
        parser.add_argument('--data_root', type=str, default='./data/dataset/Synthetic', help='data root')
        
        # Model
        parser.add_argument('--cuda', type=bool, default=True, help='use cuda')
        parser.add_argument('--nepoch', type=int, default=[14, 10, 9], help='number of total epochs')
        parser.add_argument('--reuse', type=bool, default=False, help='if reuse model')
        parser.add_argument('--gpu_id', type=int, default=[0, 1], help='gpu id for usage')
        parser.add_argument('--start_epoch', type=int, default=1, help='the number of epoch to start')

        # Visualization and saving
        parser.add_argument('--outf', type=str, default='./data', help='folder to output temp samples')

        return parser
