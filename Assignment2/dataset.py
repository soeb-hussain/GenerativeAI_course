import pickle
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class CIFAR10(Dataset):

    def __init__(self, root, train = True, transforms = None):
        self.root = root
        self.transforms = transforms
        self.split = train

        self.data = []
        self.targets = []
        self.train_data = [file for file in os.listdir(root) if "data_batch" in file]
        self.test_data = [file for file in os.listdir(root) if "test_batch" in file]

        data_split = self.train_data if self.split else self.test_data

        for files in data_split:
            entry = self.extract(os.path.join(root,files))
            self.data.append(entry["data"])
            self.targets.extend(entry["labels"]) 

        self.data = np.vstack(self.data)