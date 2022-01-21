import cv2
import math
import numpy as np
from os import path

class Track():
    def __init__(self):
        file_path = path.join(path.dirname(__file__), 'track.png')
        self.track = cv2.imread(file_path)
        self.small_overview = cv2.resize(self.track, (self.track.shape[1]//2, self.track.shape[0]//2))
        self.rows, self.cols, _ = self.track.shape

    def get_track(self):
        return self.track.copy()

    def get_small_track(self):
        return self.small_overview.copy()

    def transform(self, position, rotation):
        '''
        Transforms the track plane into wanted perspective.
        position is a 2D vector and rotation is in rad.
        '''
        alpha = math.degrees(rotation)+90
        R_M = np.concatenate((cv2.getRotationMatrix2D((self.cols//2, self.rows//2),alpha,1), np.array([[0,0,1]])))
        T_M = np.float32([ [1,0,self.cols//2-position[0]], [0,1,self.rows//2-position[1]] , [0,0,1]])
        
        M = R_M @ T_M
        self.transformed = cv2.warpAffine(self.track,M[:-1,:], (self.cols, self.rows))
    
    def get_transformed(self):
        return self.transformed
