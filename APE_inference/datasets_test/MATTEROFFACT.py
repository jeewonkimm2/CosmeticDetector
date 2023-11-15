
import os
import math
import random
from collections import defaultdict

import torch
import torchvision
import torchvision.transforms as transforms

from torchvision.datasets import ImageFolder
from torchvision import datasets

class_name = 'MATTEROFFACT'

classes = 'MATTEROFFACT'

templates = [
    "a photo of {}.",
    "a MATTEROFFACT photo of the {}.",
    "a photo of the {}.",
]

class MATTEROFFACT:
    
    dataset_dir = class_name

    def __init__(self, root, num_shots, preprocess):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, 'images')

        train_preprocess = transforms.Compose([
            transforms.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))])
        test_preprocess = preprocess

        # self.train = ImageFolder(f'./data/cosmetic/images/train/MATTEROFFACT', transform=train_preprocess)
        # self.val = ImageFolder(f'./data/cosmetic/images/val/MATTEROFFACT', transform=test_preprocess)
        self.test = ImageFolder(f'./data/cosmetic/images/test/MATTEROFFACT', transform=test_preprocess)
