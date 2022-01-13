import math
import numpy as np
import cv2

class Camera():
    def __init__(self, track, car, resolution):
        self.track = track
        self.track_image = track.get_track()
        self.car = car
        self.resolution = resolution

        self.rows, self.cols, _ = self.track_image.shape
        self.x1 = self.cols//2 - self.resolution[1]//2
        self.x2 = self.x1 + self.resolution[1]
        self.y1 = self.rows//2 - self.car.wheelbase
        self.y2 = self.y1 - self.resolution[0]
    
    def capture_frame(self):
        transformed = self.track.get_transformed()
        croped = transformed[self.y2:self.y1,self.x1:self.x2,:]
        return croped
    
    ######## 
    # For Visualisation
    
    def get_frame_points(self):
        # points are relative from middle of rear axcle. List of vectors
        pts = [
            [self.car.wheelbase+10, -self.resolution[0]//2,1],
            [self.car.wheelbase+self.resolution[1], -self.resolution[0]//2,1],
            [self.car.wheelbase+self.resolution[1], +self.resolution[0]//2,1],
            [self.car.wheelbase+10, +self.resolution[0]//2,1]
        ]
        T_M = self.car.get_transformation_matrix()
        transformed = [T_M.dot(pt) for pt in pts]
        return np.array(transformed)[:,:-1]
