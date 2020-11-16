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
        parser.add_argument('--train', type=bool, default=False, help='train or eval')

        # Model
        parser.add_argument('--cuda', type=bool, default=True, help='use cuda')
        parser.add_argument('--shuffle', type=bool, default=False, help='reuse model')

        # Visualization and saving
        parser.add_argument('--outf', type=str, default='./data', help='folder to output temp samples')

        return parser
