import time
from datetime import datetime
import torch


class Config(object):
    def __init__(self, testset_num=1):
        # General
        self.testset_num = testset_num
        self.dir_root = '../datasets/PM2.5'
        # self.save_dir = '../weights/PM2.5/weights_{}_testset{}'.format(
        #     str(datetime.utcnow()).split()[0], self.testset_num
        # )
        self.save_dir = '../weights/PM2.5/weights_{}_testset{}'.format('2019-09-11', self.testset_num)
        self.save_dir_test = self.save_dir.replace('weights', 'results')
        self.time_st = time.time()
        # Training
        self.epochs = 20            # Training epochs
        self.batch_size = 8         # ...
        self.batch_size_test = 1
        self.lr = 1e-4
        self.weight_decay = 0.0
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.losses = []
        self.GPUs = "0,1"
        self.num_val_per_epoch = 4
