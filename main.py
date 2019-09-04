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

if not os.path.exists(config.save_dir):
    os.makedirs(config.save_dir)
print('save_dir:', config.save_dir)
cudnn.benchmark = True


# Data generator
from data import gen_paths_of_image, DataGen
image_paths = gen_paths_of_image(root_path='../datasets/PM2.5data/fog_1508_data')
TBVs = np.loadtxt('data_preparation/TBVs.txt').tolist()
entropies = np.loadtxt('data_preparation/entropies.txt').tolist()
pm = np.loadtxt('data_preparation/pm25.txt').tolist()

TBV_min = np.min(TBVs)
TBV_range = np.max(TBVs) - TBV_min
entro_min = np.min(entropies)
entro_range = np.max(entropies) - entro_min

TBVs = (TBVs - TBV_min) / TBV_range
entropies = (entropies - entro_min) / entro_range

config.testset_num = 81
image_paths = image_paths[:-config.testset_num]
TBVs = TBVs[:-config.testset_num]
entropies = entropies[:-config.testset_num]
pm = pm[:-config.testset_num]

seed = 7
random.seed(seed); random.shuffle(image_paths)
random.seed(seed); random.shuffle(TBVs)
random.seed(seed); random.shuffle(entropies)
random.seed(seed); random.shuffle(pm)

gen_train = DataGen(image_paths, TBVs, entropies, pm, batch_size=config.batch_size)

print('Train Len', gen_train.data_len)

# Training
model = Network(pretrain=True)
model = nn.DataParallel(model)
model = model.cuda()
criterion = torch.nn.MSELoss(reduction='sum').to(config.device)
optimizer = optim.Adam(model.parameters(), lr=config.lr, betas=(0.9, 0.999), weight_decay=config.weight_decay)
model.train()
for epoch in range(config.epochs):
    losses_curr = []
    for idx_load in range(0, gen_train.data_len, config.batch_size):
        batch_image, batch_TBV, batch_entropy, batch_pm = gen_train.gen_batch()
        pm_pred = model(
            torch.from_numpy(batch_image).float().cuda(),
            torch.from_numpy(batch_TBV).float().cuda(),
            torch.from_numpy(batch_entropy).float().cuda()
        )

        with open('preds.txt', 'a+') as fout:
            fout.write('{:.3f}, {:.3f} --|-- {:.3f}, {:.3f}\n'.format(pm_pred[0].item(), pm_pred[1].item(), batch_pm[0], batch_pm[1]))
        loss = criterion(pm_pred, torch.tensor(batch_pm).float().cuda().unsqueeze(-1))
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
    print('\nepoch={}, loss={}, time={}m'.format(epoch+1, loss, int((time.time()-config.time_st)/60)))

# Loss plot
plt.plot(config.losses)
plt.legend(['losses'])
plt.title('Final Loss = {:5.f}'.format(config.losses[-1]))
plt.show()
