from scene_change import VideoSceneTransitions
import cv2
import os
from pytube import YouTube

# 조절 가능
threshold = 40

def download_video_from_youtube(youtube_link, save_path):
    yt = YouTube(youtube_link)
    stream = yt.streams.filter(progressive=True, file_extension='mp4').first()
    video_title = stream.title

    # 파일명 생성
    filename = video_title + ".mp4"
    filepath = os.path.join(save_path, filename)

    # 유튜브 동영상 다운로드
    stream.download(output_path=save_path, filename=filename)

    return filepath


# 유튜브 링크 (예시)
youtube_link = "https://www.youtube.com/watch?v=2-8Jjrc0Hp4"
# 동영상 저장 경로
save_path = "./"


# 유튜브 동영상 다운로드 및 파일 경로 얻기
video_filepath = download_video_from_youtube(youtube_link, save_path)
# print("다운로드한 동영상 파일 경로: ", video_filepath)

transitions_calculator = VideoSceneTransitions(threshold)
transition_times = transitions_calculator.calculate_transitions(video_filepath)

# 화면 전환 시간 계산
print("화면 전환 시간: ", transition_times)

# 시간 추출 후 동영상 삭제
os.remove(video_filepath)
