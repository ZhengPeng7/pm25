import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import pandas as pd
import matplotlib.pyplot as plt

from config import Config
from network import Network


# Configurations

config = Config()
os.environ["CUDA_VISIBLE_DEVICES"] = config.GPUs

if not os.path.exists(config.save_dir_test):
    os.makedirs(config.save_dir_test)
print('save_dir:', config.save_dir_test)
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
image_paths = image_paths[-config.testset_num:]
TBVs = TBVs[-config.testset_num:]
entropies = entropies[-config.testset_num:]
pm = pm[-config.testset_num:]

gen_test = DataGen(image_paths, TBVs, entropies, pm, batch_size=config.batch_size_test, training=False)

# Test
model = Network(pretrain=True)
# model = nn.DataParallel(model)
model = model.cuda()
model.eval()

config.save_dir = config.save_dir.replace('09-05', '09-04')
paths_weights = [
    os.path.join(config.save_dir, p) for p in
    sorted(os.listdir(config.save_dir), key=lambda x: int(x.split('epoch')[1].split('_')[0]))
]
best_preds, best_MAPEs, best_MAPE_mean, best_path_weights = [1e6], [1e6], 1e6, 'zhengpeng'

for path_weights in paths_weights[:2]:
    print('Processing {}...'.format(path_weights.split('/')[-1]))
    st_k = list(torch.load(path_weights)['state_dict'].keys())
    st_v = list(torch.load(path_weights)['state_dict'].values())
    st = {}
    for idx in range(len(st_k)):
        st[st_k[idx].replace('module.', '')] = st_v[idx]
    model.load_state_dict(st)
    pm_preds = []
    MAPEs = []
    for idx_load in range(0, gen_test.data_len, gen_test.batch_size):
        batch_image, batch_TBV, batch_entropy, batch_pm = gen_test.gen_batch()
        with torch.no_grad():
            pm_pred = model(
            torch.from_numpy(batch_image).float().cuda(),
            torch.from_numpy(batch_TBV).float().cuda(),
            torch.from_numpy(batch_entropy).float().cuda()
        ).squeeze().item()
        pm_preds.append(pm_pred)
        batch_pm = np.squeeze(batch_pm)
        MAPE = np.abs(pm_pred - batch_pm) / batch_pm * 100
        MAPEs.append(MAPE)
    print('\tMAPE_mean = {:.3f}, best_MAPE_mean = {:.3f}'.format(np.mean(MAPEs), best_MAPE_mean))
    if np.mean(MAPEs) < best_MAPE_mean:
        best_preds = pm_preds
        best_MAPEs = MAPEs
        best_path_weights = path_weights
        best_MAPE_mean = np.mean(MAPEs)
results = np.hstack([
    np.array(pm).reshape(-1, 1),
    np.array(best_preds).reshape(-1, 1),
    np.array(best_MAPEs).reshape(-1, 1)
])

# Save raw images and prediction results.
# for image_path in image_paths:
#     cv2.imwrite(os.path.join(config.save_dir_test, image_path.split('/')[-1]), cv2.imread(image_path))

path_results = os.path.join(
    config.save_dir_test,
    best_path_weights.split('/')[-1].replace('RRNet', 'results_of_weights_').replace('.pth', '_meanMAPE{:.3f}.csv'.format(best_MAPE_mean))
)
pd.DataFrame(results).to_csv(path_results, index=False, header=['Label', 'Predictions', 'MAPE(%)'])

plt.plot(pm, 'r')
plt.plot(best_preds, 'g')
plt.plot(best_MAPEs, 'b')
plt.legend(['Label', 'Prediction', 'MAPE'])
plt.title('RM2.5')
plt.savefig(os.path.join(config.save_dir_test, 'results_plot.png'))