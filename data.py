import os
import cv2
import numpy as np
from PIL import Image, ImageFile
import torch


ImageFile.LOAD_TRUNCATED_IMAGES = True


def gen_paths_of_image(root_path='../datasets/PM2.5data/fog_1508_data'):
    paths = []
    scene_dirs = sorted(
        [os.path.join(root_path, p) for p in os.listdir(root_path)],
        key=lambda x: int(x.split('_')[-1])
    )
    for scene_dir in scene_dirs:
        for image_path in [os.path.join(scene_dir, p) for p in sorted(
                os.listdir(scene_dir), key=lambda x: int(x.split('.')[-2].split('_')[-1]))]:
            paths.append(image_path)
    return paths


def image_preprocessing(unit, preproc=False):
    unit = (unit.astype(np.float32) / 255) - 0.5
    if preproc:
        pass
    unit = np.transpose(unit, (2, 0, 1))
    return unit


def load_image(pth):
    file_suffix = pth.split('.')[-1].lower()
    if file_suffix in ['jpg', 'png']:
        try:
            unit = cv2.cvtColor(cv2.imread(pth), cv2.COLOR_BGR2RGB)
        except:
            unit = np.array(Image.open(pth).convert('RGB'))
    elif file_suffix in ['npy']:
        unit = np.load(pth)
    elif file_suffix in ['.h5']:
        print('Under dev now.')
        return None
    else:
        print('Unsupported file type.')
        return None
    return unit


class DataGen():
    def __init__(self, paths, TBVs, entropies, pm, batch_size=4):
        self.anchor = 0
        self.paths = np.asarray(paths)
        self.TBVs = TBVs
        self.entropies = entropies
        self.pm = pm
        self.batch_size = batch_size
        self.data_len = len(paths) if isinstance(paths, list) else np.squeeze(self.paths).shape[0]
        self.images = []
        for path in paths:
            self.images.append(load_image(path)[124:856, ...])

    def gen_batch(self):
        batch_image, batch_TBV, batch_entropy, batch_pm = [], [], [], []
        for _ in range(self.batch_size):
            image = image_preprocessing(self.images[self.anchor])
            batch_image.append(image if np.random.random() < 0.5 else image[:, ::-1, :].copy())

            batch_TBV.append(np.expand_dims(np.zeros(image.shape[1:])+self.TBVs[self.anchor], 0))
            batch_entropy.append(np.expand_dims(np.zeros(image.shape[1:])+self.entropies[self.anchor], 0))
            batch_pm.append(self.pm[self.anchor])
            self.anchor = (self.anchor + 1) % self.data_len
        batch_image, batch_TBV = np.asarray(batch_image), np.asarray(batch_TBV)
        batch_entropy, batch_pm = np.asarray(batch_entropy), np.asarray(batch_pm)
        return batch_image, batch_TBV, batch_entropy, batch_pm


def denorm_on_imgprec(tensor):
    return ((tensor + 0.5) * 255).astype(np.uint8)
