import math
import numpy as np
import cv2

class Camera():
    def __init__(self, track, car, resolution):
        self.track = track
        self.car = car
        self.resolution = resolution
    
    def capture_frame(self):
        track_image = self.track.get_track()
        alpha = math.degrees(self.car.rotation)+90
        rows, cols, _ = track_image.shape
        T_M = np.float32([ [1,0,cols//2-self.car.position[0]], [0,1,rows//2-self.car.position[1]] ])
        R_M = cv2.getRotationMatrix2D((cols//2, rows//2),alpha,1)
        translated = cv2.warpAffine(track_image,T_M, (cols, rows))
        rotated = cv2.warpAffine(translated,R_M, (cols, rows))
        
        x1 = cols//2 - self.resolution[0]//2
        x2 = x1 + self.resolution[0]
        y1 = rows//2 - self.car.wheelbase
        y2 = y1 - self.resolution[1]
        croped = rotated[y2:y1,x1:x2,:]
        return croped
