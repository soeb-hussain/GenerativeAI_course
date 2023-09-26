import torch
import numpy as np

from trainer import TrainingConfig, Trainer
from dataset import CIFAR10
from torchvision.transforms import Compose,ToTensor,RandomHorizontalFlip,RandomRotation,ColorJitter,Normalize

train_set = CIFAR10(root="./cifar-10-batches-py"
                    , train = True
                    , transforms = Compose([
                        ToTensor()
                        , RandomHorizontalFlip()
                        , RandomRotation(degrees=10)
                        , ColorJitter(brightness=0.5)
                        , Normalize( mean= (0.49, 0.48, 0.44),
                                    std = (0.24, 0.24, 0.26))
                    ]))


test_set = CIFAR10(root="./cifar-10-batches-py",train=False,
                   transforms=Compose([
                        ToTensor(),
                        Normalize(mean=(0.4913997551666284, 0.48215855929893703, 0.4465309133731618),
                                  std=(0.24703225141799082, 0.24348516474564, 0.26158783926049628))
                    ]))

train_config = TrainingConfig(max_epochs=100,
                              lr=0.00023570926966106847,
                              weight_decay=0.00021257445443209662,
                              ckpt_path="./models/CIFAR10.pt",
                              batch_size=64,
                              num_workers=4)


print('ending')