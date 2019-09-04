import time
from datetime import datetime
import torch


class Config(object):
    def __init__(self):
        # General
        self.dir_root = '../datasets/PM2.5'
        self.save_dir = '../weights/PM2.5/weights_{}'.format(str(datetime.utcnow()).split()[0])
        # Training
        self.epochs = 100            # Training epochs
        self.batch_size = 4         # ...
        self.batch_size_test = 1
        self.lr = 1e-4
        self.weight_decay = 0.0
        self.testset_num = 12
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.losses = []
        self.time_st = time.time()
        self.GPUs = "0,1"
