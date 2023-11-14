from ultralytics import YOLO
import cv2

class CosmeticDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
    
    def detect_objects(self, image_path):
        results = self.model.predict(source=image_path, save=False)
        boxes = results[0].boxes
        return boxes
    
    def crop_and_save_images(self, image_path, save_dir):
        image = cv2.imread(image_path)
        boxes = self.detect_objects(image_path)
        
        for i, box in enumerate(boxes):
            box_coordinates = box.xyxy
            x1, y1, x2, y2 = map(int, box_coordinates[0].tolist())
            cropped_image = image[y1:y2, x1:x2]
            save_path = f"{save_dir}/cropped_image_{i}.png"
            cv2.imwrite(save_path, cropped_image)
            print(f"Cropped image {i+1} saved at: {save_path}")

# 예시
model_path = '/home/iai/Desktop/Jeewon/Capstone/checkpoints/best.pt'
image_path = '/home/iai/Desktop/Jeewon/Git/CosmeticDetector/object_detection/cosmeticpic.jpg'
save_dir = '/home/iai/Desktop/Jeewon/Git/CosmeticDetector/object_detection/'

# 클래스 인스턴스 생성
detector = CosmeticDetector(model_path)

# 객체 탐지 및 이미지 크롭하여 저장
detector.crop_and_save_images(image_path, save_dir)
