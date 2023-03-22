import random
import glob
import re

from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm

import tensorslow as ts
from tensorslow.linalg import Tensor
from tensorslow.datasets import Dataset


class MNIST(Dataset):
    train_path = Path(f'{ts.__path__[0]}/datasets/mnist_data/train/**/*.png')
    test_path = Path(f'{ts.__path__[0]}/datasets/mnist_data/valid/**/*.png')

    def __init__(self, batch_size=32, load_train=True, load_test=False, shuffle=True, normalize=True, seed=520):
        random.seed(seed)        
        self.batch_size = batch_size
        self.load_train = load_train
        self.load_test = load_test
        self.shuffle = shuffle
        self.normalize = normalize
        data = self.download()
        self._x_train = data['x_train']
        self._y_train = data['y_train']
        self._x_test = data['x_test']
        self._y_test = data['y_test']

    def download(self):
        data = dict(x_train=None, y_train=None, x_test=None, y_test=None)
        if self.load_train:
            train_load = 'Loading MNIST Training Data'
            x, y = self._download_dataset(self.train_path, desc=train_load)
            data['x_train'] = x
            data['y_train'] = y

        if self.load_test:
            test_load = 'Loading MNIST Test Data'
            x, y = self._download_dataset(self.test_path, desc=test_load)
            data['x_test'] = x
            data['y_test'] = y

        return data

    def _download_dataset(self, path, desc=''):
        tensors = []
        x_data, y_data = [], []
        paths = list(glob.glob(str(path), recursive=True))
        if self.shuffle:
            random.shuffle(paths)

        for i, img_path in tqdm(enumerate(paths), total=len(paths), desc=desc):
            label = int(re.split(r"\\|/", img_path)[-2])
            if i % self.batch_size == 0 and x_data:
                np_data = np.vstack(x_data) 
                x = Tensor(np_data.tolist(), np_data.shape, data_tensor=True)
                y = Tensor(y_data, (len(y_data),))
                tensors.append((x, y))
                x_data, y_data = [], []
            img = Image.open(img_path)
            x_np = np.array(img, dtype=float).reshape(-1)
            if self.normalize:
                x_np /= 255.0
            x_data.append(x_np)
            y_data.append(label)
        if self.shuffle:
            random.shuffle(tensors)

        x_data, y_data = [], []
        for x, y in tensors:
            x_data.append(x)
            y_data.append(y)
        return x_data, y_data


