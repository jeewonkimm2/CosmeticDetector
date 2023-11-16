import os
import math
import random
from collections import defaultdict
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision import datasets

class DatasetsGenerator:
    def __init__(self,class_name=None):
        self.class_name = class_name

    def generate_module(self):
        module_content = f'''
import os
import math
import random
from collections import defaultdict

import torch
import torchvision
import torchvision.transforms as transforms

from torchvision.datasets import ImageFolder
from torchvision import datasets

class_name = '{self.class_name}'

classes = '{self.class_name}'

templates = [
    "a photo of {{}}.",
    "a {self.class_name} photo of the {{}}.",
    "a photo of the {{}}.",
]

class {self.class_name}:
    
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

        self.train = ImageFolder(f'./data/cosmetic/images/train/{self.class_name}', transform=train_preprocess)
        self.val = ImageFolder(f'./data/cosmetic/images/val/{self.class_name}', transform=test_preprocess)
        # self.test = ImageFolder(f'./data/cosmetic/images/test/{self.class_name}', transform=test_preprocess)
        
        self.template = templates
        self.classnames = class_name
        self.cupl_path = f'./gpt3_prompts/CuPL_prompts_{self.class_name}.json'

        split_by_label_dict = defaultdict(list)
        for i in range(len(self.train.imgs)):
            split_by_label_dict[self.train.targets[i]].append(self.train.imgs[i])
        imgs = []
        targets = []

        for label, items in split_by_label_dict.items():
            imgs = imgs + random.sample(items, num_shots)
            targets = targets + [label for i in range(num_shots)]
        self.train.imgs = imgs
        self.train.targets = targets
        self.train.samples = imgs
        
'''

        # Construct the module file path based on the class name
        module_file_path = os.path.join('./datasets', f"{self.class_name}.py")

        # Save the module content to the module file
        with open(module_file_path, "w") as module_file:
            module_file.write(module_content)

        print(f"Python module file saved to {module_file_path}")

# # inference
# if __name__ == "__main__":
    
#     folder_path = './data/cosmetic/images/train'

#     # 해당 폴더 내의 모든 항목 가져오기
#     entries = os.listdir(folder_path)

#     # 폴더만 필터링하여 리스트에 저장
#     folder_names = [entry for entry in entries if os.path.isdir(os.path.join(folder_path, entry))]
    
    
#     # class_name = "Medical"  # Change the class name as needed
#     for cls in folder_names:
#         generator = DatasetsGenerator(class_name = cls)
#         generator.generate_module()
