import cv2
import os
from pytube import YouTube
import yt_dlp as youtube_dl


def divide_video(youtube_link):

    # 동영상 저장 경로
    save_path = "./"

    print(1)
    # video_filepath = download_video_from_youtube(youtube_link, save_path)
    video_filepath = download_video(youtube_link)
    print(2)

    # 조절 가능
    threshold = 40

    transitions_calculator = VideoSceneTransitions(threshold)
    transition_times = transitions_calculator.calculate_transitions(video_filepath)

    # 처리 결과 반환
    return transition_times

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

def download_video(url):
    ydl_opts = {'format': 'bestvideo+bestaudio/best'}
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