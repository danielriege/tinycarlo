import cv2
import math
import numpy as np
import random
from os import path

class Track():
    def __init__(self, track_config, base_path, overview_downscale, trajectory_color):
        self.current_track = 0
        self.tracks = []
        self.overviews = []
        self.dimensions = []
        self.track_spawns = []
        self.trajectories = []
        if base_path == None or len(track_config) == 0:
            self.__create_empty_plane()
            print('No track provided. Creating empty plane.')
        else:
            for track_ in track_config:
                track_plane = track_.get('image', None)
                if track_plane != None:
                    track_path = path.join(base_path, track_plane)
                    track_image = cv2.imread(track_path)
                    # binarize plane
                    track_image = self.__binarize_track_image(track_image)
                    # extract trajectory points
                    track_image, trajectory_points = self.__extract_trajectory(track_image, trajectory_color)

                    self.tracks.append(track_image)
                    self.trajectories.append(trajectory_points)

                    self.overviews.append(cv2.resize(track_image, (int(track_image.shape[1]/overview_downscale), int(track_image.shape[0]/overview_downscale))))
                    rows, cols, _ = track_image.shape
                    self.dimensions.append((rows, cols))
                else:
                    self.__create_empty_plane()
                spawns = track_.get('spawns', [500.0,500.0,0.0])
                self.track_spawns.append(spawns)
    
    def __create_empty_plane(self):
        track_image = np.zeros((1000,1000,3), dtype=int)
        self.tracks.append(track_image)
        self.overviews.append(track_image)
        self.dimensions.append((1000,1000))
        self.trajectories.append(np.array([0,0]))

    def get_track(self):
        return self.tracks[self.current_track].copy()

    def get_small_track(self):
        return self.overviews[self.current_track].copy()

    def get_spawns(self):
        return self.track_spawns[self.current_track]

    def set_next_track(self):
        self.current_track = random.randint(0,len(self.tracks)-1)
    
    def get_cte(self, position):
        trajectory_points = self.trajectories[self.current_track]
        cte = self.__clostest_points(position, trajectory_points)
        return cte
    
    def __clostest_points(self, point, point_list):
        dist_2 = np.sum((point - point_list)**2, axis=-1)
        closest_index = np.argmin(dist_2)
        return np.sqrt(dist_2[closest_index])


    def __binarize_track_image(self, image):
        _, im_th = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY)
        return im_th

    def __extract_trajectory(self, image, trajectory_color):
        points = np.where(np.all(image == trajectory_color, axis=-1))
        transposed_points = np.flip(np.transpose(points),-1)
        track_image = image
        track_image[points] = 0
        return track_image, transposed_points

    def transform(self, position, rotation):
        '''
        Transforms the track plane into wanted perspective.
        position is a 2D vector and rotation is in rad.
        '''
        rows, cols = self.dimensions[self.current_track]
        alpha = math.degrees(rotation)+90
        R_M = np.concatenate((cv2.getRotationMatrix2D((cols//2, rows//2),alpha,1), np.array([[0,0,1]])))
        T_M = np.float32([ [1,0,cols//2-position[0]], [0,1,rows//2-position[1]] , [0,0,1]])
        
        M = R_M @ T_M
        self.transformed = cv2.warpAffine(self.tracks[self.current_track],M[:-1,:], (cols, rows))
    
    def get_transformed(self):
        return self.transformed
