import math
import numpy as np
import cv2

class Camera():
    def __init__(self, track, car, resolution, roi, position, id_):
        self.track = track
        self.car = car
        self.resolution = resolution
        self.position = position
        self.id = id_

        # decouple crop window size from output resolution
        self.crop_size = roi
    
    def __set_new_frame_points(self):
        self.track_image = self.track.get_track()
        self.rows, self.cols, _ = self.track_image.shape

        self.x1 = self.cols//2 - self.crop_size[1]//2 + self.position[1]
        self.x2 = self.cols//2 + self.crop_size[1]//2 + self.position[1]
        self.y1 = self.rows//2 - self.car.wheelbase - self.position[0]
        self.y2 = self.rows//2 - self.car.wheelbase - self.crop_size[0] - self.position[0]
    
    def capture_frame(self):
        self.__set_new_frame_points()
        transformed = self.track.get_transformed()
        croped = transformed[self.y2:self.y1,self.x1:self.x2,:]
        resized = cv2.resize(croped, list(reversed(self.resolution))) # reversed because cv2 is weird and uses (width, height)
        return resized
    
    ######## 
    # For Visualisation
    
    def get_frame_points(self):
        # points are relative from middle of rear axcle. List of vectors
        base_y = self.car.wheelbase + self.position[0]
        pts = [
            [base_y, -self.crop_size[1]//2 + self.position[1],1],
            [base_y + self.crop_size[0], -self.crop_size[1]//2 + self.position[1],1],
            [base_y +self.crop_size[0], +self.crop_size[1]//2 + self.position[1],1],
            [base_y, +self.crop_size[1]//2 + self.position[1],1]
        ]
        T_M = self.car.get_transformation_matrix()
        transformed = [T_M.dot(pt) for pt in pts]
        return np.array(transformed)[:,:-1]
