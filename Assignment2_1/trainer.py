import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader


class TrainingConfig:
    
    lr=3e-4
    betas=(0.9,0.995)
    weight_decay=5e-4
    num_workers=0
    max_epochs=10
    batch_size=64
    ckpt_path=None #Specify a model path here. Ex: "./Model.pt"
    shuffle=True
    pin_memory=True
    verbose=True
    
    def __init__(self,**kwargs):
        for key,value in kwargs.items():
            setattr(self,key,value)


class Trainer:
    def __init__(self, model, train_dataset, test_dataset, config):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config

        self.train_losses = []
        self.train_accuracies = []
        self.test_losses = []
        self.test_accuracies = [] 

        self.device = 'cpu'

        if torch.cuda.is_available():
            raw_model = self.model.module if hasattr(self.model,"mo")