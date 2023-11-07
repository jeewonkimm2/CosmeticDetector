import os
from google.cloud import vision
import Levenshtein


class ImageBrandExtractor:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ImageBrandExtractor, cls).__new__(cls)
        return cls._instance

    def __init__(self, model_path=None):
        # 'client' 속성이 없을 경우에만 초기화
        if not hasattr(self, 'client'):
            self.client = vision.ImageAnnotatorClient()

    def extract_text(self, image_path):
        with open(image_path, 'rb') as image_file:
            content = image_file.read()
        image = vision.Image(content=content)
        response = self.client.text_detection(image=image)
        texts = response.text_annotations
        if texts:
            return texts[0].description
        else:
            return ""

    def extract_texts_from_folder(self, folder_path, allowed_extensions=['.jpg']):
        results = []
        for filename in os.listdir(folder_path):
            if os.path.splitext(filename)[1].lower() in allowed_extensions:
                filepath = os.path.join(folder_path, filename)
                extracted_text = self.extract_text(filepath)
                if extracted_text.strip():
                    result_entry = {
                        "filename": filename,
                        "text": extracted_text
                    }
                    results.append(result_entry)
        return results

    @staticmethod
    def find_most_similar_brand(ocr_text, brand_list):
        most_similar_brand = None
        min_distance = float('inf')

        # 1차 검사
        ocr_text = ocr_text.replace(" ", "").replace(
            "-", "").replace("&", "").replace(".", "").replace("+", "").lower()

        for brand in brand_list:
            # 브랜드가 ocr_text에 정확히 포함(1차)
            if brand in ocr_text:
                return brand

            # 거리 찾기
            distances = [Levenshtein.distance(
                brand_word, ocr_word) for brand_word in brand.split() for ocr_word in ocr_text.split()]
            distance = min(distances)

            if distance < min_distance:
                min_distance = distance
                most_similar_brand = brand

        return most_similar_brand

    def extract_and_find_brand(self, folder_path, brand_list, allowed_extensions=['.jpg']):
        extracted_texts = self.extract_texts_from_folder(folder_path)

        results = []
        for entry in extracted_texts:
            filename = entry["filename"]
            ocr_text = entry["text"]
            brand_names = self.find_most_similar_brand(ocr_text, brand_list)

            if brand_names:
                result_entry = {
                    "filename": filename,
                    "brand_name": brand_names
                }
                results.append(result_entry)
        return results
