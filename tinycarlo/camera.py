import math
import numpy as np
import cv2

class Camera():
    def __init__(self, track, car, resolution):
        self.track_image = track.get_track()
        self.car = car
        self.resolution = resolution

        self.rows, self.cols, _ = self.track_image.shape
        self.x1 = self.cols//2 - self.resolution[0]//2
        self.x2 = self.x1 + self.resolution[0]
        self.y1 = self.rows//2 - self.car.wheelbase
        self.y2 = self.y1 - self.resolution[1]
    
    def capture_frame(self):
        alpha = math.degrees(self.car.rotation)+90
        R_M = np.concatenate((cv2.getRotationMatrix2D((self.cols//2, self.rows//2),alpha,1), np.array([[0,0,1]])))
        T_M = np.float32([ [1,0,self.cols//2-self.car.position[0]], [0,1,self.rows//2-self.car.position[1]] , [0,0,1]])
        
        M = R_M @ T_M
        transformed = cv2.warpAffine(self.track_image,M[:-1,:], (self.cols, self.rows))
    
        croped = transformed[self.y2:self.y1,self.x1:self.x2,:]
        return croped
    
    def get_frame_points(self):
        pts = [
            [self.car.wheelbase, -self.resolution[0]//2,1],
            [self.car.wheelbase+self.resolution[1], -self.resolution[0]//2,1],
            [self.car.wheelbase+self.resolution[1], +self.resolution[0]//2,1],
            [self.car.wheelbase, +self.resolution[0]//2,1]
        ]
        T_M = self.car.get_transformation_matrix()
        transformed = [T_M.dot(pt) for pt in pts]
        return np.array(transformed)[:,:-1]
