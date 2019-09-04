import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt

from config import Config
from network import Network


# Configurations

config = Config()
os.environ["CUDA_VISIBLE_DEVICES"] = config.GPUs
criterion = torch.nn.MSELoss(reduction='sum').to(config.device)

model = Network(pretrain=True)
model = nn.DataParallel(model)
model = model.cuda()

optimizer = optim.Adam(model.parameters(), lr=config.lr, betas=(0.9, 0.999), weight_decay=config.weight_decay)

if not os.path.exists(config.save_dir):
    os.makedirs(config.save_dir)
print('save_dir:', config.save_dir)
cudnn.benchmark = True


# Data generator
from data import gen_paths_of_image, DataGen
image_paths = gen_paths_of_image(root_path='../datasets/PM2.5/fog_1508_data')
TBVs = np.loadtxt('data_preparation/TBVs.txt')
entropies = np.loadtxt('data_preparation/entropies.txt')
pm = np.loadtxt('data_preparation/pm25.txt')
config.testset_num = 20
image_paths = image_paths[config.testset_num//2:len(image_paths)-config.testset_num//2]
TBVs = TBVs[config.testset_num//2:len(image_paths)-config.testset_num//2]
entropies = entropies[config.testset_num//2:len(image_paths)-config.testset_num//2]
pm = pm[config.testset_num//2:len(image_paths)-config.testset_num//2]
seed = 7
random.seed(seed); random.shuffle(image_paths)
random.seed(seed); random.shuffle(TBVs)
random.seed(seed); random.shuffle(entropies)
random.seed(seed); random.shuffle(pm)

gen_train = DataGen(image_paths, TBVs, entropies, pm, batch_size=config.batch_size)

print('Train Len', gen_train.data_len)


# Training
model.train()
for epoch in range(config.epochs):
    losses_curr = []
    for idx_load in range(0, gen_train.data_len, config.batch_size):
        batch_image, TBV, entropy, pm = gen_train.gen_batch()
        pm_pred = model(torch.from_numpy(batch_image).float().cuda())

        loss = criterion(pm_pred, pm)
        loss = loss.to(config.device)
        losses_curr.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    loss = np.mean(losses_curr)
    config.losses.append(loss)
    dict_ckpt = {'epoch': epoch + 1, 'state_dict': model.state_dict(), 'loss': loss}
    path_ckpt = os.path.join(config.save_dir, 'RRNet_epoch{}_loss{:.5f}.pth'.format(epoch+1, loss))
    torch.save(dict_ckpt, path_ckpt)
    print('epoch={}, loss={}, time={}m'.format(epoch+1, loss, int((time.time()-config.time_st)/60)))

# Loss plot
plt.plot(config.losses)
plt.legend(['losses'])
plt.title('Final Loss = {:5.f}'.format(config.losses[-1]))
plt.show()
