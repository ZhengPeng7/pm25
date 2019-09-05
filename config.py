import time
from datetime import datetime
import torch


class Config(object):
    def __init__(self, testset_num=1):
        # General
        self.testset_num = testset_num
        self.dir_root = '../datasets/PM2.5'
        self.save_dir = '../weights/PM2.5/weights_{}_testset{}'.format(
            str(datetime.utcnow()).split()[0], self.testset_num
        )
        self.save_dir_test = '../results/PM2.5/results_{}_testset{}'.format(
            str(datetime.utcnow()).split()[0], self.testset_num
        )
        # Training
        self.epochs = 20            # Training epochs
        self.batch_size = 2         # ...
        self.batch_size_test = 1
        self.lr = 1e-4
        self.weight_decay = 0.0
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.losses = []
        self.time_st = time.time()
        self.GPUs = "0,1"
