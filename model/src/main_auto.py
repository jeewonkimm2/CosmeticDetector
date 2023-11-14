"""
새로운 클래스가 추가되면

0. val 데이터 증강
1. 이미지 어텐션 기반 크롭, prompt 생성
2. configs 생성
3. datasets폴더에 py폴더 생성

4. caches 생성 (extract_features.py 수정) - train, val
5. val 활용하여 best parameter 찾기
6. inference
"""

import argparse
import clip
import os

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from ImageAugmentor import ImageAugmentor
from ImageProcessor import ImageProcessor
from PromptGeneration import ImageCaptioning
from ConfigGenerator import ConfigGenerator
from DatasetsGenerator import DatasetsGenerator
from GenerateFeatures import GenerateFeatures

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--class_name', type=str, help='Name of new class name')
    args = parser.parse_args()
    return args


def main():

    args = get_arguments()
    

        
    print(f'Current Class : {args.class_name}')


    # 0단계
    # val 이미지 증강 클래스 초기화
    aug_input_dir = f'./data/cosmetic/images/train/{args.class_name}'
    aug_output_dir = f'./data/cosmetic/images/val/{args.class_name}'
    augmentor = ImageAugmentor(aug_input_dir, aug_output_dir)
    # val 이미지 증강 및 저장 수행
    augmentor.augment_and_save_images()
    
    # 1단계 : Prompt
    ImageProcessor_ = ImageProcessor(cls_name = args.class_name)
    ImageProcessor_.generate_image()
    ImageProcessor_.generate_prompt()

    
    # 2단계 : configs
    config_generator = ConfigGenerator(args.class_name)
    # Add sections and keys with comments
    config_generator.add_key("root_path", "data")
    config_generator.add_key("seed", 3407, "The seed 3407 comes from the paper:\n''Torch.manual_seed(3407) is all you need:\nOn the influence of random seeds in deep learning architectures for computer vision''")
    config_generator.add_key("search_hp", True)
    config_generator.add_key("shots", 0)
    config_generator.add_key("search_scale", [7, 7, 0.5])
    config_generator.add_key("search_step", [200, 20, 5])
    config_generator.add_key("init_beta", 1)
    config_generator.add_key("init_alpha", 2)
    config_generator.add_key("init_gamma", 0.1)
    config_generator.add_key("best_beta", 0)
    config_generator.add_key("best_alpha", 0)
    config_generator.add_key("best_gamma", 0)
    config_generator.add_key("eps", 0.001)
    config_generator.add_key("feat_num", 500)
    config_generator.add_key("w_training_free", [0.7, 0.3])
    config_generator.add_key("w_training", [0.2, 0.8])
    config_generator.add_key("dataset", args.class_name, comment_before="# ------ Basic Config ------")
    config_generator.add_key("backbone", "ViT-B/32")
    config_generator.add_key("lr", 0.0001)
    config_generator.add_key("augment_epoch", 10)
    config_generator.add_key("train_epoch", 20)
    output_file_path = f"./configs/{args.class_name}.yaml"
    config_generator.generate_config(output_file_path)
    
    
    # 3단계 : dataset.py
    generator = DatasetsGenerator(class_name = args.class_name)
    generator.generate_module()
    
    # import라인 추가
    new_code = f"""
from datasets.{args.class_name} import {args.class_name}
    """

    file_path1 = "./GenerateFeatures.py"
    
    
    with open(file_path1, "r") as file1:
        existing_code = file1.read()

    modified_code1 = new_code + existing_code
    
    with open(file_path1, "w") as file:
        file.write(modified_code1)
        
        
    file_path2 = "./main_train.py"

    with open(file_path2, "r") as file2:
        existing_code = file2.read()

    modified_code2 = new_code + existing_code

    with open(file_path2, "w") as file:
        file.write(modified_code2)

    # file_path3 = "./inference_existing.py"

    # with open(file_path3, "r") as file3:
    #     existing_code = file3.read()

    # modified_code3 = new_code + existing_code

    # with open(file_path3, "w") as file:
    #     file.write(modified_code3)       

    
    

if __name__ == '__main__':
    main()

