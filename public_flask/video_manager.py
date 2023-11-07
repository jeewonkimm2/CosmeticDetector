import os
import youtube_dl
import cv2
import uuid


class VideoManager:
    def __init__(self):
        pass

    def download_video(self, url, save_path):
        # 파일명 생성
        video_name = "video"
        video_path_without_ext = os.path.join(save_path, video_name)

        ydl_opts = {
            'format': 'bestvideo+bestaudio/best',
            'outtmpl': video_path_without_ext + '.%(ext)s'  # 확장자를 자동으로 결정하도록 함
        }

        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
            info = ydl.extract_info(url, download=False)
            # 만약 ext 정보가 없다면 mp4를 기본값으로 사용
            file_extension = info.get('ext', 'mp4')

        return video_path_without_ext + '.' + file_extension

    def calculate_transitions(self, video_path, threshold):
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

                if diff_mean > threshold:
                    transitions.append(cap.get(cv2.CAP_PROP_POS_FRAMES))

            prev_frame = frame

        cap.release()

        transition_times = [(frame / fps) for frame in transitions]

        return transition_times

    def save_frames_from_times(self, video_path, times, save_dir):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for idx, t in enumerate(times):
            frame_pos = int(t * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)

            ret, frame = cap.read()
            if ret:
                # 저장될 이미지의 이름
                frame_filename = os.path.join(save_dir, f"frame_{idx}.jpg")
                cv2.imwrite(frame_filename, frame)

        cap.release()

    def unique_folder(self, base_path="."):
        folder_name = str(uuid.uuid4())
        full_path = os.path.join(base_path, folder_name)
        os.makedirs(full_path, exist_ok=True)
        return full_path
