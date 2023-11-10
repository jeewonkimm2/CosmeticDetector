from cosmetic_finder import CosmeticFinder
from transcript_brand_extractor import TranscriptBrandExtractor
from image_brand_extractor import ImageBrandExtractor
from s3_utils import S3Utils
from video_manager import VideoManager
from image_processor import ImageProcessor
from API_communicator import APICommunicator
from data_handler import DataHandler

import os
import shutil
from flask import Flask, request, jsonify
from flask_cors import CORS


def prepare_environment():
    # 환경 설정 및 brand 리스트 준비
    image_processor = ImageProcessor()
    csv_path = 'products_list.csv'
    brand_list = image_processor.extract_unique_brands(csv_path)
    return brand_list, csv_path


def process_video(youtube_link):
    # 비디오 처리
    video_manager = VideoManager()
    save_path = video_manager.unique_folder()
    video_path = video_manager.download_video(youtube_link, save_path)
    threshold = 6.75
    transition_times = video_manager.calculate_transitions(
        video_path, threshold)
    return save_path, video_path, transition_times


def extract_frames(video_path, save_path, transition_times):
    # 프레임 추출
    video_manager = VideoManager()
    save_dir = os.path.join(save_path, 'frames')
    video_manager.save_frames_from_times(
        video_path, transition_times, save_dir)
    return save_dir


def detect_brands_in_frames(save_path):
    # 프레임에서 브랜드 검출
    save_dir = os.path.join(save_path, 'cropped_image')
    frames_folder_path = os.path.join(save_path, 'frames')
    yolo_model_path = 'YOLO.pt'
    detector = CosmeticFinder(yolo_model_path)
    detector.process_folder(frames_folder_path, save_dir)
    return save_dir


def extract_and_compare_brands(youtube_link, brand_list, cropped_images_folder_path, transition_times):
    # 브랜드 추출 및 비교
    imageExtractor = ImageBrandExtractor()
    brand_in_image = imageExtractor.extract_and_find_brand(
        cropped_images_folder_path, brand_list)

    transcriptExtractor = TranscriptBrandExtractor()
    brand_in_transcript = transcriptExtractor.extract_brands_from_youtube_url(
        youtube_link, brand_list)

    image_processor = ImageProcessor()
    brand_with_image = image_processor.update_brand_in_image(
        brand_in_image, brand_in_transcript, transition_times)
    return brand_with_image


def cleanup_and_upload(save_path, brand_with_image, cropped_images_folder_path):
    # 정리 및 업로드
    image_processor = ImageProcessor()
    class_names = list({item['brand_name'] for item in brand_with_image})
    image_processor.copy_images_to_new_folder(
        brand_with_image, cropped_images_folder_path, save_path)
    image_processor.delete_except_APE_folder(save_path)

    s3_utils = S3Utils()
    bucket_name = 'cosmetic-detector-bucket'
    s3_directory = save_path
    s3_utils.upload_directory_to_s3(save_path, bucket_name, s3_directory)
    return class_names


def perform_api_inference(save_path, class_names, csv_path):
    # API를 사용하여 추론 수행
    API_communicator = APICommunicator()
    api_url = "http://10.100.20.170:8000/find_cosmetic"
    response_data = API_communicator.send_post_request_to_api(
        api_url, save_path, class_names)

    print(response_data)

    data_handler = DataHandler()
    product_info_list = data_handler.read_csv_to_list(csv_path)
    result = data_handler.get_matching_products(
        response_data, product_info_list)
    return result


def analyze_video(youtube_link):
    # 비디오 분석 주 함수
    brand_list, csv_path = prepare_environment()
    save_path, video_path, transition_times = process_video(
        youtube_link)
    extract_frames(
        video_path, save_path, transition_times)
    cropped_images_folder_path = detect_brands_in_frames(save_path)
    brand_with_image = extract_and_compare_brands(
        youtube_link, brand_list, cropped_images_folder_path, transition_times)
    class_names = cleanup_and_upload(
        save_path, brand_with_image, cropped_images_folder_path)
    result = perform_api_inference(save_path, class_names, csv_path)

    # 후처리: 로컬 및 S3 정리
    shutil.rmtree(save_path)
    s3_utils = S3Utils()
    bucket_name = 'cosmetic-detector-bucket'
    s3_utils.delete_files_in_s3_directory(bucket_name, save_path)

    return result


# Flask app 설정
app = Flask(__name__)
CORS(app)


@app.route('/analyze', methods=['POST'])
def analyze_youtube_video():
    # 비디오 분석 Flask 라우트
    data = request.get_json()
    youtube_link = data.get('youtube_link')
    if not youtube_link:
        return jsonify({"error": "YouTube link is required!"}), 400
    result = analyze_video(youtube_link)
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8000)
