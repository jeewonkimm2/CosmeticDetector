import albumentations as A
import matplotlib.pyplot as plt
import os
import numpy as np

class ImageAugmentor:
    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.transform = A.Compose([
            A.HorizontalFlip(p=0.3),  # 30% 확률로 좌우 반전
            A.RandomRotate90(p=0.3),  # 30% 확률로 90도 회전
            A.RandomBrightnessContrast(p=0.5),  # 50% 확률로 밝기와 대조 조절
            A.Sharpen(alpha=0.5, lightness=0.2, p=0.5),
            A.Blur(blur_limit=7, p=0.2),
            A.Affine(p=0.5, rotate=(-50, 50)),
        ])
    
    def augment_and_save_images(self):
        # 입력 디렉토리 내의 모든 클래스 폴더 순회
        for class_dir in os.listdir(self.input_dir):
            class_dir_path = os.path.join(self.input_dir, class_dir)
            
            if os.path.isdir(class_dir_path):
                # 클래스 폴더 내의 이미지 파일 순회
                for filename in os.listdir(class_dir_path):
                    if filename.endswith('.jpg'):  # 이미지 파일인 경우에만 처리
                        # 이미지 파일 경로
                        image_path = os.path.join(class_dir_path, filename)

                        # 이미지 불러오기
                        image = plt.imread(image_path)
                        
                        



                        # 각 이미지당 3번의 증강을 적용하고 저장
                        for i in range(3):
                            # 증강된 이미지 생성
                            # num_channels = image.shape[2]
                            # if num_channels == 3:
                            #     transformed = self.transform(image=image)
                            # else:
                            #     transformed = self.transform(image=image)
                            transformed = self.transform(image=image)
                            transformed_image = transformed['image']
                            
                            transformed_image = transformed_image.astype(np.float32) / 255.0

                            # 클래스 이름을 포함한 폴더 경로 생성
                            class_output_dir = os.path.join(self.output_dir, class_dir)

                            # 폴더가 없을 경우 생성
                            if not os.path.exists(class_output_dir):
                                os.makedirs(class_output_dir)

                            # 저장할 이미지 파일 경로 설정 (원본 파일명에 _transformed를 추가하여 저장)
                            output_image_path = os.path.join(class_output_dir, f"{filename.replace('.jpg', '')}_transformed_{i+1}.jpg")
                            
                            transformed_image = (transformed_image * 255).astype(np.uint8)
                            # 이미지 저장
                            plt.imsave(output_image_path, transformed_image)

        print("모든 이미지 증강 및 저장 완료.")
        
