import os
import cv2
from ultralytics import YOLO


class CosmeticDetector:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(CosmeticDetector, cls).__new__(cls)
        return cls._instance

    def __init__(self, model_path):
        # 모델 초기화가 한 번만 되도록 조건 추가
        if not hasattr(self, 'model'):
            self.model = YOLO(model_path)

    def detect_objects(self, image_path):
        results = self.model.predict(source=image_path, save=False)
        boxes = results[0].boxes
        return boxes

    def crop_and_save_images(self, image_path, save_dir):
        image = cv2.imread(image_path)
        boxes = self.detect_objects(image_path)

        image_name_without_ext = os.path.splitext(
            os.path.basename(image_path))[0]

        for i, box in enumerate(boxes):
            box_coordinates = box.xyxy
            x1, y1, x2, y2 = map(int, box_coordinates[0].tolist())
            cropped_image = image[y1:y2, x1:x2]

            # 이미지 이름을 기반으로 새로운 파일 이름 생성
            save_path = os.path.join(
                save_dir, f"{image_name_without_ext}-{i+1}.jpg")
            cv2.imwrite(save_path, cropped_image)

    def process_folder(self, folder_path, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for entry in os.scandir(folder_path):
            if entry.is_file() and os.path.splitext(entry.name)[1].lower() == '.jpg':
                self.crop_and_save_images(entry.path, save_dir)
