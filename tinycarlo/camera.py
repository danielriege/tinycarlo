import math
import numpy as np
import cv2

class Camera():
    def __init__(self, track, car, resolution):
        self.track = track
        self.track_image = track.get_track()
        self.car = car
        self.resolution = resolution

        # decouple crop window size from output resolution
        self.crop_size = (480, 640)

        self.rows, self.cols, _ = self.track_image.shape
        self.x1 = self.cols//2 - self.crop_size[1]//2
        self.x2 = self.x1 + self.crop_size[1]
        self.y1 = self.rows//2 - self.car.wheelbase
        self.y2 = self.y1 - self.crop_size[0]
    
    def capture_frame(self):
        transformed = self.track.get_transformed()
        croped = transformed[self.y2:self.y1,self.x1:self.x2,:]
        resized = cv2.resize(croped, list(reversed(self.resolution))) # reversed because cv2 is weird and uses (width, height)
        return resized
    
    ######## 
    # For Visualisation
    
    def get_frame_points(self):
        # points are relative from middle of rear axcle. List of vectors
        pts = [
            [self.car.wheelbase+10, -self.crop_size[1]//2,1],
            [self.car.wheelbase+self.crop_size[0], -self.crop_size[1]//2,1],
            [self.car.wheelbase+self.crop_size[0], +self.crop_size[1]//2,1],
            [self.car.wheelbase+10, +self.crop_size[1]//2,1]
        ]
        T_M = self.car.get_transformation_matrix()
        transformed = [T_M.dot(pt) for pt in pts]
        return np.array(transformed)[:,:-1]
