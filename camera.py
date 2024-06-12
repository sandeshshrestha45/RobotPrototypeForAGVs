import cv2

class Camera:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)  # 0 for the first camera, 1 for the second, and so on
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 416)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 416)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        print(f"FPS set to: {self.fps}")

    def read_frame(self):
        ret, frame = self.cap.read()
        return ret, frame

    def release(self):
        self.cap.release()

    def close_windows(self):
        cv2.destroyAllWindows()