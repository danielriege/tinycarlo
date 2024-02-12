import math
import numpy as np
import cv2
import random

import time

class Car():
    def __init__(self, T, map, car_config):
        self.map = map
        self.track_width = car_config.get('track_width', 0.03)
        self.wheelbase = car_config.get('wheelbase', 0.08)
        self.max_steering_change = car_config.get('max_steering_change', None)
        self.T = T
        self.reset()

        self.wheel_offset = self.track_width/5 #from chassis
        self.wheel_length = self.wheelbase/3 
        self.wheel_width = self.wheel_length/6 

    def reset(self):
        self.position, self.rotation = self.map.sample_spawn()
        self.steering_angle = 0.0
        self.steering_input = 0.0
        self.radius = 0.0

    def step(self, fwd_vel, steering_angle):
        dt = self.T

        fwd_vel *= 1 # max is 1 m/s
        self.steering_input = steering_angle
        new_steering_angle = steering_angle * 33 # max steering is 33 degree
        if self.max_steering_change is None:
            self.steering_angle = new_steering_angle
        else:
            # apply smoothness
            max_steering_in_T = self.max_steering_change*self.T
            steering_change = np.clip(new_steering_angle - self.steering_angle, -max_steering_in_T, max_steering_in_T)
            self.steering_angle = self.steering_angle + steering_change

        vxn = math.cos(self.rotation)
        vyn = math.sin(self.rotation)

        if self.steering_angle == 0:
            self.radius = 0

            self.position[0] = self.position[0] + fwd_vel * vxn * dt * 1000
            self.position[1] = self.position[1] + fwd_vel * vyn * dt * 1000
        else:
            self.radius = self.wheelbase/1000 / (math.tan(math.radians(self.steering_angle)))
            ang_vel = fwd_vel / self.radius
            dyaw = ang_vel * dt

            nx = vyn # normalvector
            ny = -vxn

            tx = nx * self.radius * 1000
            ty = ny * self.radius * 1000

            R_M = np.array([[math.cos(dyaw), -math.sin(dyaw)],[math.sin(dyaw), math.cos(dyaw)]])

            rotated_vec = R_M.dot([tx, ty])
        
            self.position[0] = self.position[0] - tx + rotated_vec[0]
            self.position[1] = self.position[1] - ty + rotated_vec[1]
        
            self.rotation += dyaw

    def get_transformation_matrix(self):
        ''' 
        Returns a Transformation matrix which points to middle of rear axcle in world
        '''
        R_M = np.array([[math.cos(self.rotation), -math.sin(self.rotation),0],[math.sin(self.rotation), math.cos(self.rotation),0], [0,0,1,]])
        T_M = np.array([[1,0,self.position[0]], [0,1,self.position[1]], [0,0,1]])
        return T_M @ R_M
    
    def get_3d_transformation_matrix(self):
        ''' 
        Returns a Transformation matrix which points to middle of rear axcle in world
        '''
        R_M = np.array([[math.cos(-self.rotation), -math.sin(-self.rotation),0, 0],[math.sin(-self.rotation), math.cos(-self.rotation),0, 0], [0,0,1,0], [0,0,0,1]])
        T_M = np.array([[1,0,0,-self.position[0]], [0,1,0,-self.position[1]], [0,0,1,0], [0,0,0,1]])
        return R_M @ T_M
    
    def check_colission(self, obstacles):
        '''
        Checks for colissions with road markings. None if no colission.
        '''
        start = time.time()
        transformed = self.track.get_transformed()
        rows, cols, _ = transformed.shape
        x1 = cols//2-self.track_width//2
        x2 = cols//2+self.track_width//2
        y1 = rows//2-self.wheelbase
        y2 = rows//2
        croped = transformed[y1:y2,x1:x2,:]
        colored_pixels = np.where(croped[:,:,:] > 50)
        for obstacle in obstacles:
            for y,x in zip(colored_pixels[0], colored_pixels[1]):
                if (croped[y,x] == obstacle['color']).all():
                    return obstacle['color']
        return None

    # For Visualisation

    def get_chassis_points(self):
        T_M = self.get_transformation_matrix()
        # points are relative from middle of rear axcle. List of vectors
        pts = [[0, -self.track_width/2,1], 
        [0, self.track_width/2,1], 
        [self.wheelbase, self.track_width/2,1], 
        [self.wheelbase, -self.track_width/2,1]]

        transformed = [T_M.dot(pt) for pt in pts]
        return np.array(transformed)[:,:-1]

    def get_wheel_points(self):
        T_M = self.get_transformation_matrix()
        fl_angle, fr_angle = self.__ackermann_steering()
        fl_R_M = np.concatenate((cv2.getRotationMatrix2D((self.wheelbase-self.wheel_length/2, -self.track_width/2),math.degrees(fl_angle),1), np.array([[0,0,1]])))
        fr_R_M = np.concatenate((cv2.getRotationMatrix2D((self.wheelbase-self.wheel_length/2, self.track_width/2),math.degrees(fr_angle),1), np.array([[0,0,1]])))
        
        fl = [[self.wheelbase-self.wheel_length, -self.track_width/2,1], [self.wheelbase, -self.track_width/2, 1]]
        fr = [[self.wheelbase-self.wheel_length, self.track_width/2,1], [self.wheelbase, self.track_width/2,1]]
        # rotate front wheels by steering angle
        fl = [(T_M @ fl_R_M).dot(pt) for pt in fl]
        fl = np.array(fl)[:,:-1]
        fr = [(T_M @ fr_R_M).dot(pt) for pt in fr]
        fr = np.array(fr)[:,:-1]

        rl = [[0, -self.track_width/2,1], [self.wheel_length, -self.track_width/2,1]]
        rr = [[0, self.track_width/2,1], [self.wheel_length, self.track_width/2,1]]

        rl = [T_M.dot(pt) for pt in rl]
        rl = np.array(rl)[:,:-1]
        rr = [T_M.dot(pt) for pt in rr]
        rr = np.array(rr)[:,:-1]
        return [fl, fr, rl, rr]
    
    def __ackermann_steering(self):
        """
        Calculates the steering angle for each wheel (Ackermann steering geometry). Just for visuals
        """
        if self.radius == 0:
            return (0,0)
        else:
            wb = self.wheelbase/1000
            tw = (self.track_width/1000)
            inner = math.atan(wb/(self.radius-(tw/2+0.000001))) * -1
            outer = math.atan(wb/(self.radius+(tw/2+0.000001))) * -1
            if self.radius > 0:
                return (outer, inner) # left, right
            else:
                return (inner, outer) # left, right 

