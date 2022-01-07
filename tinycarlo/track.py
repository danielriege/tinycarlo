import cv2

class Track():
    def __init__(self):
        self.track = cv2.imread('tinycarlo/track.png')

    def get_track(self):
        return self.track.copy()