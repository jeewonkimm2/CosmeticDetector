
from datasets.ABBOTT import ABBOTT

from GenerateFeatures import GenerateFeatures
import os
import random
import argparse
import yaml
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms

# from datasets import build_dataset
# from datasets.utils import build_data_loader
import clip
from utils import *

import matplotlib.pyplot as plt
from PIL import Image
import csv
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True



def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--class_name', type=str, help='Name of new class name')
    # parser.add_argument('--inference_type', type=str, help='Type of inference')
    
    args = parser.parse_args()
    return args

# APE-T 모델의 훈련 및 평가
# cache_key : 입력 특성과 관련된 임베딩 벡터 저장
# cache_values : 카테고리에 대한 임베딩 벡터의 집합 저장
def APE_T(cfg, cache_keys, cache_values, val_features, val_labels, test_features, test_labels, clip_weights, clip_model, train_loader_F):
    args = get_arguments()
    feat_dim, cate_num = clip_weights.shape
    # 카테고리 개수, -1, 카테고리 개수로 형태를 바꿈
    cache_values = cache_values.reshape(cate_num, -1, cate_num)
    # 전치(transpose) 이후, cfg['shots'], feature 차원의 형태로 reshape
    cache_keys = cache_keys.t().reshape(cate_num, cfg['shots'], feat_dim).reshape(cate_num, -1, feat_dim)
    
    cfg['w'] = cfg['w_training']
    # -1, feature 차원의 형태로 reshape
    cache_keys, cache_values = cache_keys.reshape(-1, feat_dim), cache_values.reshape(-1, cate_num)
    adapter = APE_Training(cfg, clip_weights, clip_model, cache_keys).cuda()
    
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=cfg['lr'], eps=cfg['eps'], weight_decay=1e-1)  # 
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg['train_epoch'] * len(train_loader_F))
    Loss = SmoothCrossEntropy()
    
    # 초기 beta, alpha 생성
    beta, alpha = cfg['init_beta'], cfg['init_alpha']
    # 최고 정확도, epoch 저장 위해 변수 초기화
    best_acc, best_epoch = 0.0, 0
    feat_num = cfg['feat_num']
    
    # Train 시작!
    # config의 train_epoch만큼 반복
    for train_idx in range(cfg['train_epoch']):
        # Train
        adapter.train()
        correct_samples, all_samples = 0, 0
        loss_list = []
        print('Train Epoch: {:} / {:}'.format(train_idx, cfg['train_epoch']))

        # 훈련 데이터 : 이미지와 타겟 가져오기
        for i, (images, target) in enumerate(tqdm(train_loader_F)):
            images, target = images.cuda(), target.cuda()
            with torch.no_grad():
                image_features = clip_model.encode_image(images)
                # 정규화
                image_features /= image_features.norm(dim=-1, keepdim=True)

            # adapter 모델을 사용해서 새로운 캐시 키(new_cache_keys), 새로운 CLIP 가중치(new_clip_weights), R_FW 값 얻기
            # R_FW : 이미지 특성과 캐시 키(입력 특성과 관련된 임베딩 벡터) 간의 유사도
            new_cache_keys, new_clip_weights, R_FW = adapter(cache_keys, clip_weights, cache_values)
            R_fF = image_features @ new_cache_keys.half().t()

            # 캐시 key와 value 사이의 관계 -> APE-T 모델이 캐시를 활용하여 예측하는데 사용. R_fW와 cache_logits을 조합하여 최종 예측 결과 생성
            cache_logits = ((-1) * (beta - beta * R_fF)).exp() @ R_FW
            R_fW = 100. * image_features @ new_clip_weights
            ape_logits = R_fW + cache_logits * alpha

            loss = Loss(ape_logits, target)

            acc = cls_acc(ape_logits, target)
            # 맞은 샘플 수
            correct_samples += acc / 100 * len(ape_logits)
            # 전체 샘플 수
            all_samples += len(ape_logits)
            loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        # 훈련 데이터의 정확도, 맞은 샘플 수, 전체 샘플 수, 평균 손실
        current_lr = scheduler.get_last_lr()[0]
        print("Training and testing on training dataset is ongoing...")
        print('LR: {:.6f}, Acc: {:.4f} ({:}/{:}), Loss: {:.4f}'.format(current_lr, correct_samples / all_samples, correct_samples, all_samples, sum(loss_list)/len(loss_list)))

        # Eval : test on validation set
        adapter.eval()
        with torch.no_grad():
            new_cache_keys, new_clip_weights, R_FW = adapter(cache_keys, clip_weights, cache_values)

            R_fF = test_features @ new_cache_keys.half().t()
            cache_logits = ((-1) * (beta - beta * R_fF)).exp() @ R_FW
            R_fW = 100. * test_features @ new_clip_weights
            ape_logits = R_fW + cache_logits * alpha
        acc = cls_acc(ape_logits, test_labels)

        print("**** APE-T's test accuracy: {:.2f}. ****\n".format(acc))
        if acc > best_acc:
            best_acc = acc
            best_epoch = train_idx
            print("Saving model ...")
            torch.save(adapter, cfg['cache_dir'] + "/APE-T_" + str(cfg['shots']) + "shots.pt")
        else:
            print("Saving model ...")
            torch.save(adapter, cfg['cache_dir'] + "/APE-T_" + str(cfg['shots']) + "shots.pt")

    
    # 훈련이 끝나고 최고 정확도를 갖는 모델 로드
    adapter = torch.load(cfg['cache_dir'] + "/APE-T_" + str(cfg['shots']) + "shots.pt")
    print(f"**** After fine-tuning, APE-T's best test accuracy: {best_acc:.2f}, at epoch: {best_epoch}. ****\n")
    
    
    
    # validation set을 통해 hyperparameter를 탐색하는 단계 진입
    print("\n-------- Searching hyperparameters on the val set. --------")
    # Search Hyperparameters
    # 변수 초기화
    best_search_acc = 0
    best_beta, best_alpha = 0, 0
    # beta와 alpha를 찾을 범위 설정
    beta_list = [i * (cfg['search_scale'][0] - 0.1) / cfg['search_step'][0] + 0.1 for i in range(cfg['search_step'][0])]
    alpha_list = [i * (cfg['search_scale'][1] - 0.1) / cfg['search_step'][1] + 0.1 for i in range(cfg['search_step'][1])]
    
    
    for beta in beta_list:
        for alpha in alpha_list:
            with torch.no_grad():
                new_cache_keys, new_clip_weights, R_FW = adapter(cache_keys, clip_weights, cache_values)
                
                R_fF = val_features @ new_cache_keys.half().t()
                cache_logits = ((-1) * (beta - beta * R_fF)).exp() @ R_FW
                R_fW = 100. * val_features @ new_clip_weights
                ape_logits = R_fW + cache_logits * alpha
            acc = cls_acc(ape_logits, val_labels)
            if acc > best_search_acc:
                print("New best setting, alpha: {:.2f}, beta: {:.2f}; accuracy: {:.2f}".format(alpha, beta, acc))
                best_search_acc = acc
                best_alpha, best_beta = alpha, beta
                
    # YAML 파일 경로        
    file_path = "params.yaml"

    # 기존 YAML 파일 읽기
    with open(file_path, "r") as file:
        data = yaml.safe_load(file)

    data = {
        cfg['dataset']:{
            "alpha":best_alpha,
            "beta": best_beta
        }
    }
    
        # 수정된 데이터 YAML 파일에 쓰기
    with open(file_path, "w") as file:
        yaml.safe_dump(data, file)
    
    
    
    
    contents = yaml.load(open(f'./configs/{args.class_name}.yaml', 'r'), Loader=yaml.Loader)
    contents['best_alpha'] = best_alpha
    contents['best_beta'] = best_beta
    with open(f'./configs/{args.class_name}.yaml', 'w') as yaml_file:
        yaml.dump(contents, yaml_file, default_flow_style=False)


    
    print(f"best_beta : {best_beta}")
    print(f"best_alpha : {best_alpha}")

    
    
    
    

        
        
        
        
        
        
        
        
        
        





    print("\nAfter searching, the best val accuarcy: {:.2f}.\n".format(best_search_acc))


    # print("\n-------- Evaluating on the test set. --------")
    # adapter = torch.load(cfg['cache_dir'] + "/APE-T_" + str(cfg['shots']) + "shots.pt")
    # with torch.no_grad():
    #     new_cache_keys, new_clip_weights, R_FW = adapter(cache_keys, clip_weights, cache_values)
        
    #     R_fF = test_features @ new_cache_keys.half().t()
    #     cache_logits = ((-1) * (best_beta - best_beta * R_fF)).exp() @ R_FW

    #     R_fW = 100. * test_features @ new_clip_weights
    #     ape_logits = R_fW + cache_logits * cfg['best_alpha']


    #     # 레이블 추론
    #     predicted_labels = torch.argmax(ape_logits, dim=1)

    # acc = cls_acc(ape_logits, test_labels)

    # print("predicted label : ", predicted_labels)
    # print("**** APE-T's test accuracy: {:.2f}. ****\n".format(acc))
    
    # args = get_arguments()

    # # Retrieve the original and refined indices and features after forward pass
    # original_indices = adapter.original_indices.cpu().numpy()
    # original_features = adapter.original_features.cpu().numpy()


    # print("images의 크기:", len(images))
    # print("original_indices의 크기:", len(original_indices))
    # print("original_features의 크기:", original_features.shape)
    



    
    

def main():
    
    # Load config file
    args = get_arguments()
    # assert (os.path.exists(args.config))
    



    print(f"Current clss is {args.class_name}")
    assert (os.path.exists(f'configs/{args.class_name}.yaml'))

    # cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    cfg = yaml.load(open(f'configs/{args.class_name}.yaml', 'r'), Loader=yaml.Loader)
    
    
    shots = cfg['shots']
    print(shots)
    
    
    # cfg['shots'] = args.shot
    cfg['shots'] = shots

    cache_dir = os.path.join('./caches', cfg['dataset'])
    os.makedirs(cache_dir, exist_ok=True)
    cfg['cache_dir'] = cache_dir

    # CLIP
    clip_model, preprocess = clip.load(cfg['backbone'])
    clip_model.eval()

    # Prepare dataset
    random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])

    # Textual features
    print("\nGetting textual features as CLIP's classifier.")
    clip_weights = load_text_feature(cfg)

    # Construct the cache model by few-shot training set
    print("\nConstructing cache model by few-shot visual features and labels.")
    cache_keys, cache_values = load_few_shot_feature(cfg)

    # Pre-load val features
    print("\nLoading visual features and labels from val set.")
    val_features, val_labels = loda_val_test_feature(cfg, "val")
    
    # Pre-load test features
    print("\nLoading visual features and labels from test set.")
    test_features, test_labels = loda_val_test_feature(cfg, "val")


    dynamic_class = globals()[args.class_name]
    brand = dynamic_class(cfg['root_path'], cfg['shots'], preprocess)
    train_loader_F = torch.utils.data.DataLoader(brand.train, batch_size=256, num_workers=8, shuffle=True)



    APE_T(cfg, cache_keys, cache_values, val_features, val_labels, test_features, test_labels, clip_weights, clip_model, train_loader_F)

    

if __name__ == '__main__':
    main()

