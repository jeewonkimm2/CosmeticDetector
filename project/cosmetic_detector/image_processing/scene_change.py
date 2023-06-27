import cv2
import os
from pytube import YouTube
import yt_dlp as youtube_dl


def divide_video(youtube_link):

    # 동영상 저장 경로
    save_path = "./"

    download_video(youtube_link, save_path)

    file_path = "C:\\Users\\LBC\\Desktop\\clip_cosmetic_classification\\project\\cosmetic_detector\\video.webm"


    # 조절 가능
    threshold = 40

    transitions_calculator = VideoSceneTransitions(threshold)
    transition_times = transitions_calculator.calculate_transitions(file_path)


    # 시간 추출 후 동영상 삭제
    os.remove(file_path)

    # 처리 결과 반환
    return transition_times

def download_video(url, save_path):
    # 파일명 생성
    filename = "video"
    filepath = os.path.join(save_path, filename)

    ydl_opts = {
        'format': 'bestvideo+bestaudio/best',
        'outtmpl': filepath,  # 다운로드 경로와 파일명 지정
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

class VideoSceneTransitions:
    def __init__(self, threshold=40):
        self.threshold = threshold

    def calculate_transitions(self, video_path):

        cap = cv2.VideoCapture(video_path)
        prev_frame = None
        transitions = []
        fps = cap.get(cv2.CAP_PROP_FPS)

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            if prev_frame is not None:
                diff = cv2.absdiff(frame, prev_frame)
                diff_mean = diff.mean()

                if diff_mean > self.threshold:
                    transitions.append(cap.get(cv2.CAP_PROP_POS_FRAMES))

            prev_frame = frame

        cap.release()

        transition_times = [(frame / fps) for frame in transitions]

        return transition_times