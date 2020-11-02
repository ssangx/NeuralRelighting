import argparse


class BaseOptions():
    """Define options for training and test"""

    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    def initialize(self, parser):
        # Model
        return parser

    def parse(self):
        self.parser = self.initialize(self.parser)
        self.args = self.parser.parse_args()
        return self.args

    