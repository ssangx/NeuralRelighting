import os
import torch
import argparse
from options.base_options import BaseOptions


class TestOptions(BaseOptions):
    """Define options for training and test"""

    def __init__(self):
        super(TestOptions, self).__init__()

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)

        parser.add_argument('--mode', type=str, default='test', help='network cfig name')
        parser.add_argument('--name', type=str, default='test', help='network cfig name')
        parser.add_argument('--train', type=bool, default=False, help='train or eval')
        parser.add_argument('--image_size', type=int, default=256, help='if load all images together')

        # Model
        parser.add_argument('--cuda', type=bool, default=True, help='use cuda')
        parser.add_argument('--inc', type=int, default=3, help='input channels')
        parser.add_argument('--outc', type=int, default=3, help='output channels')
        parser.add_argument('--ngf',  type=int, default=64, help='ngf')
        parser.add_argument('--reuse', type=bool, default=True, help='reuse model')
        parser.add_argument('--shuffle', type=bool, default=False, help='reuse model')
        parser.add_argument('--start_epoch', type=int, default=14, help='#epoch to start')

        # Visualization and saving
        parser.add_argument('--image_dir', type=str, default='./images', help='folder to output temp samples')
        parser.add_argument('--outf', type=str, default='../samples', help='folder to output temp samples')

        return parser
