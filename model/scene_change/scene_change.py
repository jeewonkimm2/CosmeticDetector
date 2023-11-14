import cv2

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
