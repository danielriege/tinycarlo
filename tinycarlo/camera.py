import math
import numpy as np
import cv2

class Camera():
    def __init__(self, track, car, resolution):
        self.track_image = track.get_track()
        self.car = car
        self.resolution = resolution
    
    def capture_frame(self):
        alpha = math.degrees(self.car.rotation)+90
        rows, cols, _ = self.track_image.shape
        R_M = np.concatenate((cv2.getRotationMatrix2D((cols//2, rows//2),alpha,1), np.array([[0,0,1]])))
        T_M = np.float32([ [1,0,cols//2-self.car.position[0]], [0,1,rows//2-self.car.position[1]] , [0,0,1]])
        
        M = R_M @ T_M
        transformed = cv2.warpAffine(self.track_image,M[:-1,:], (cols, rows))
        
        x1 = cols//2 - self.resolution[0]//2
        x2 = x1 + self.resolution[0]
        y1 = rows//2 - self.car.wheelbase
        y2 = y1 - self.resolution[1]
        croped = transformed[y2:y1,x1:x2,:]
        return croped
