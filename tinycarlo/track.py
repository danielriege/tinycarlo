import cv2
import math
import json
import numpy as np
from os import path

class Track():
    def __init__(self):
        file_path = path.join(path.dirname(__file__), 'track.png')
        self.track = cv2.imread(file_path)
        gt_path = path.join(path.dirname(__file__), 'track.png.json')
        self.gt_clockwise, self.gt_counterclockwise = self.__set_groundtruth(gt_path)
        self.small_overview = cv2.resize(self.track, (self.track.shape[1]//2, self.track.shape[0]//2))
        self.rows, self.cols, _ = self.track.shape

    def __create_cone_track(self, cone_path):
        with open(cone_path, 'r') as f:
            data = json.load(f)
        cone_objects = data['objects']

        return_cones = {}
        track_image = np.zeros((data['size']['height'], data['size']['width'], 3), dtype=np.uint8)
        for cone_object in cone_objects:
            points = cone_object['points']['exterior']
            color = (0,0,0)
            if cone_object['classTitle'] == 'left':
                return_cones['left'] = points
                color = (255,0,0)
            elif cone_object['classTitle'] == 'right':
                return_cones['right'] = points
                color = (0,255,255)
            for point in points:
                track_image = cv2.circle(track_image, point, radius=6, color=color, thickness=-1)
        return return_cones, track_image

    def __set_groundtruth(self, gt_path):
        with open(gt_path, 'r') as f:
            data = json.load(f)
        objects = data['objects']
        clockwise, counterclockwise = [],[]
        for object in objects:
            if object['classTitle'] == 'gt_counterclockwise':
                counterclockwise = object['points']['exterior']
            elif object['classTitle'] == 'gt_clockwise':
                clockwise = object['points']['exterior']
        return clockwise, counterclockwise

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
