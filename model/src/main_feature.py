from GenerateFeatures import GenerateFeatures
import clip
import argparse
import random
from ultralytics import YOLO
import os



def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--class_name', type=str, help='Name of new class name')
    args = parser.parse_args()
    return args

# 화장품 유무에 따라 삭제
def object_detect():
    
    count_detected = 0  # 검출된 개체 수
    count_undetected = 0  # 데이터셋 총 개수
    
    model_path = './checkpoints/best.pt'
    model = YOLO(model_path)
    
    dataset_root = './data/cosmetic/images/train'

    # 데이터셋 폴더 내 이미지 목록 가져오기
    for category in os.listdir(dataset_root):
        category_path = os.path.join(dataset_root, category)
        if not os.path.isdir(category_path):
            continue  # 폴더가 아닌 파일은 무시
        for class_folder in os.listdir(category_path):
            class_folder_path = os.path.join(category_path, class_folder)
            if not os.path.isdir(class_folder_path):
                continue  # 폴더가 아닌 파일은 무시
            image_list = os.listdir(class_folder_path)
            for image_name in image_list:
                # 이미지 파일 경로
                image_path = os.path.join(class_folder_path, image_name)
                
                if image_name != '0.jpg':
                    try:
                        # 입력 데이터에 대한 예측값 계산
                        results = model.predict(source=image_path, save=False)
                        for result in results:
                            exist = result.boxes.xyxy.tolist()
                            if len(exist) == 0:
                                # 검출되지 않았으면 이미지 파일 삭제
                                os.remove(image_path)
                                count_undetected += 1
                            else:
                                count_detected += 1
                    except FileNotFoundError:
                        # 이미지 파일이 없는 경우도 삭제 처리
                        os.remove(image_path)
                        count_undetected += 1

    # # 결과 출력
    # print("검출된 개체 수:", count_detected)
    # print("no detect 개수:", count_undetected)
    
    
    
def find_shots():


    # 디렉토리 경로 설정
    base_dir = './data/cosmetic/images/train'

    # 카테고리별 클래스별 이미지 개수를 저장할 딕셔너리 초기화
    category_class_image_counts = {}

    # 기본 디렉토리 탐색
    for category_folder in os.listdir(base_dir):
        category_path = os.path.join(base_dir, category_folder)
        
        # 각 카테고리 폴더 아래의 클래스 폴더 탐색
        for class_folder in os.listdir(category_path):
            class_path = os.path.join(category_path, class_folder)
            
            # 클래스 폴더 아래의 이미지 개수 세기
            if os.path.isdir(class_path):
                num_images = len(os.listdir(class_path))
                
                # 카테고리별 클래스별 이미지 개수를 딕셔너리에 저장
                if category_folder in category_class_image_counts:
                    if class_folder in category_class_image_counts[category_folder]:
                        category_class_image_counts[category_folder][class_folder].append(num_images)
                    else:
                        category_class_image_counts[category_folder][class_folder] = [num_images]
                else:
                    category_class_image_counts[category_folder] = {class_folder: [num_images]}

    # 각 클래스에서 이미지 수를 최대 이미지 개수로 맞춥니다.
    for category, class_counts in category_class_image_counts.items():
        common_max_count = max(1, min(min(counts) for counts in class_counts.values()))  # Ensure common_max_count is at least 1
        
        for class_folder, counts in class_counts.items():
            class_path = os.path.join(base_dir, category, class_folder)
            images = os.listdir(class_path)
            
            # 이미지 수가 common_max_count보다 많으면 랜덤으로 선택하여 삭제
            if len(images) > common_max_count:
                # if common_max_count == 1:
                images.remove('0.jpg')  # Retain 0.jpg if common_max_count is 1
                images_to_remove = random.sample(images, len(images) - common_max_count+1)
                for image_to_remove in images_to_remove:
                    image_to_remove_path = os.path.join(class_path, image_to_remove)
                    os.remove(image_to_remove_path)  # 이미지 파일 삭제

    

def main():
    
    object_detect()
    find_shots()
    
    args = get_arguments()
    clip_model, preprocess = clip.load('ViT-B/32')
    GenerateFeatures_ = GenerateFeatures(clip_model, preprocess, cls_name = args.class_name)
    GenerateFeatures_.generate_features()
    
if __name__ == '__main__':
    main()