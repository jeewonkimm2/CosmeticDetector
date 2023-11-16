
    
    
import os
import random
import argparse
import yaml
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms

import clip
from utils import *

class GenerateFeatures:
    def __init__(self, clip_model, preprocess, cls_name=None):
        # self.clip_model, self.preprocess = clip.load('ViT-B/32')
        self.clip_model = clip_model
        self.preprocess = preprocess
        self.cls_name = cls_name
        
    # 훈련 데이터셋 이미지 특성 추출 : CLIP모델을 사용해서 이미지 특성 추출
    def extract_few_shot_feature(self, cfg, clip_model, train_loader_cache):
        cache_keys = []
        cache_values = []
        with torch.no_grad():
            # Data augmentation for the cache model
            for augment_idx in range(cfg['augment_epoch']):
                train_features = []
                print('Augment Epoch: {:} / {:}'.format(augment_idx, cfg['augment_epoch']))
                for i, (images, target) in enumerate(tqdm(train_loader_cache)):
                    images = images.cuda()
                    image_features = clip_model.encode_image(images)
                    train_features.append(image_features)
                    if augment_idx == 0:
                        target = target.cuda()
                        cache_values.append(target)
                cache_keys.append(torch.cat(train_features, dim=0).unsqueeze(0))
            
        cache_keys = torch.cat(cache_keys, dim=0).mean(dim=0)
        cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
        cache_keys = cache_keys.permute(1, 0)
        cache_values = F.one_hot(torch.cat(cache_values, dim=0)).half()
        torch.save(cache_keys, cfg['cache_dir'] + '/keys_' + str(cfg['shots']) + "shots.pt")
        # torch.save(cache_keys, cfg['cache_dir'] + '/keys_' + shots + "shots.pt")
        torch.save(cache_values, cfg['cache_dir'] + '/values_' + str(cfg['shots']) + "shots.pt")
        # torch.save(cache_values, cfg['cache_dir'] + '/values_' + shots + "shots.pt")

        return
    
    # Val/test 데이터셋 이미지 특성 추출 : CLIP모델을 사용해서 이미지 특성 추출
    def extract_val_test_feature(self, cfg, split, clip_model, loader):
        features, labels = [], []
        with torch.no_grad():
            for i, (images, target) in enumerate(tqdm(loader)):
                images, target = images.cuda(), target.cuda()
                image_features = clip_model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                features.append(image_features)
                labels.append(target)
        features, labels = torch.cat(features), torch.cat(labels)
        torch.save(features, cfg['cache_dir'] + "/" + split + "_f.pt")
        torch.save(labels, cfg['cache_dir'] + "/" + split + "_l.pt")
        return
    
    # 텍스트 특성 추출 : CLIP모델을 사용해서 텍스트 특성 추출(주어진 클래스와 템플릿에 따라 텍스트 구성)
    # def extract_text_feature(self, cfg, classnames, prompt_path, clip_model, template):
    def extract_text_feature(self, cfg, prompt_path, clip_model, template):
    
        f = open(prompt_path)
        prompts = json.load(f)
        with torch.no_grad():
            clip_weights = []
            # for classname in classnames:
            #     # Tokenize the prompts
            #     classname = classname.replace('_', ' ')
                
            #     template_texts = [t.format(classname) for t in template]
            #     cupl_texts = prompts[classname]
            #     texts = template_texts + cupl_texts
            #     texts_token = clip.tokenize(texts, truncate=True).cuda()
            #     # prompt ensemble for ImageNet
            #     class_embeddings = clip_model.encode_text(texts_token)
            #     class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            #     # 평균을 취함으로써 여러 문장의 특징을 하나로 요약하고, 클래스에 대한 종합적인 특성을 나타내는 벡터를 얻을 수 있음
            #     class_embedding = class_embeddings.mean(dim=0)
            #     class_embedding /= class_embedding.norm()
            #     clip_weights.append(class_embedding)
                    # Tokenize the prompts
            # classname = self.cls_name
            # classname = classname.replace('_', ' ')
            
            image_folder = f"./data/cosmetic/images/train/{self.cls_name}"
            class_folders = []

            for folder_name in os.listdir(image_folder):
                folder_path = os.path.join(image_folder, folder_name)
                if os.path.isdir(folder_path):
                    class_folders.append(folder_name)
            for cls in class_folders:
                template_texts = [t.format(cls) for t in template]
                cupl_texts = prompts[cls]
                texts = template_texts + cupl_texts
                texts_token = clip.tokenize(texts, truncate=True).cuda()
                # prompt ensemble for ImageNet
                class_embeddings = clip_model.encode_text(texts_token)
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                # 평균을 취함으로써 여러 문장의 특징을 하나로 요약하고, 클래스에 대한 종합적인 특성을 나타내는 벡터를 얻을 수 있음
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                clip_weights.append(class_embedding)

            clip_weights = torch.stack(clip_weights, dim=1).cuda()
        torch.save(clip_weights, cfg['cache_dir'] + "/text_weights_cupl_t.pt")
        return
    
    def generate_features(self):
        
        train_path = f'./data/cosmetic/images/train/{self.cls_name}'
        
        file_list = os.listdir(train_path)
        image_count = len([filename for filename in file_list if filename.endswith('.jpg')])
        print(f'폴더 내의 이미지 파일 개수: {image_count}')
    
        # k_shot = image_count
        k_shot = 1
        set = self.cls_name
        
        cfg = yaml.load(open('./configs/{}.yaml'.format(set), 'r'), Loader=yaml.Loader)

        cache_dir = os.path.join('./caches', cfg['dataset'])
        os.makedirs(cache_dir, exist_ok=True)
        cfg['cache_dir'] = cache_dir
        
        random.seed(1)
        torch.manual_seed(1)
        
        cfg['shots'] = k_shot
        
        # 변경된 "shots" 값을 저장합니다.
        with open('./configs/{}.yaml'.format(set), 'w') as yaml_file:
            yaml.dump(cfg, yaml_file, default_flow_style=False)
        
        dynamic_class = globals()[self.cls_name]

        dataset = dynamic_class(cfg['root_path'], cfg['shots'], self.preprocess)
        # dataset = example(cfg['root_path'], cfg['shots'], self.preprocess)
         
        val_loader = torch.utils.data.DataLoader(dataset.test, batch_size=64, num_workers=8, shuffle=False)
        
        # # Construct the cache model by few-shot training set
        # print("\nConstructing cache model by few-shot visual features and labels.")
        # self.extract_few_shot_feature(cfg, self.clip_model, train_loader_cache)
        
        # # val 이미지 특징 추출 (Extract val/test features)
        # print("\nLoading visual features and labels from val and test set.")
        # self.extract_val_test_feature(cfg, "val", self.clip_model, val_loader)
        
        # test 이미지 특징 추출 (Extract val/test features)
        self.extract_val_test_feature(cfg, 'test', self.clip_model, val_loader)
        
        # # self.extract_text_feature(cfg, dataset.classnames, dataset.cupl_path, self.clip_model, dataset.template)
        # self.extract_text_feature(cfg, dataset.cupl_path, self.clip_model, dataset.template)

        
        
# # inference
# if __name__ == "__main__":
#     class_name = "example"  # Change the class name as needed
#     clip_model, preprocess = clip.load('ViT-B/32')
#     GenerateFeatures = GenerateFeatures(clip_model, preprocess, cls_name = class_name)
#     GenerateFeatures.generate_features()