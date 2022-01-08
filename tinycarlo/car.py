import math
import numpy as np
import cv2

class Car():
    def __init__(self, track, track_width, wheelbase, T):
        self.track = track
        self.track_width = track_width
        self.wheelbase = wheelbase
        self.T = T
        self.reset()

        self.wheel_offset = 5 #from chassis
        self.wheel_length = 40 # in mm
        self.wheel_width = 6 # in mm

    def reset(self):
        self.position = np.array([700,1460])
        self.rotation = 0.48
        self.steering_angle = 0.0
        self.radius = 0.0

    def step(self, fwd_vel, steering_angle):
        dt = self.T

        fwd_vel *= 1 # max is 1 m/s
        self.steering_angle = steering_angle * 33 # max steering is 33 degree

        self.radius = self.wheelbase/1000 / (math.tan(math.radians(self.steering_angle)+0.000001))
        ang_vel = fwd_vel / self.radius
        dyaw = ang_vel * dt

        vx = fwd_vel * math.cos(self.rotation)
        vy = fwd_vel * math.sin(self.rotation)

        dx = (vx * math.cos(dyaw) - vy * math.sin(dyaw)) * dt
        dy = (vx * math.sin(dyaw) + vy * math.cos(dyaw)) * dt

        self.position[0] += dx*1000
        self.position[1] += dy*1000
        self.rotation += dyaw

    def calculate_steering_front(self):
        wb = self.wheelbase/1000
        tw = (self.track_width/1000)
        inner = math.atan(wb/(self.radius-(tw//2+0.000001))) * -1
        outer = math.atan(wb/(self.radius+(tw//2+0.000001))) * -1
        if self.radius > 0:
            return (outer, inner) # left, right
        else:
            return (inner, outer)

    def get_transformation_matrix(self):
        ''' 
        Returns a Transformation matrix which points to middle of rear axcle in world
        '''
        R_M = np.array([[math.cos(self.rotation), -math.sin(self.rotation),0],[math.sin(self.rotation), math.cos(self.rotation),0], [0,0,1]])
        T_M = np.array([[1,0,self.position[0]], [0,1,self.position[1]], [0,0,1]])
        return T_M @ R_M
    
    def check_colission(self):
        '''
        Checks for colissions with road markings. returns 'g' for green line and 'r' for red line. None if no colission. 
        if colission is with both, 'r' is returned
        '''
        transformed = self.track.get_transformed()
        rows, cols, _ = transformed.shape
        x1 = cols//2-self.track_width//2
        x2 = cols//2+self.track_width//2
        y1 = rows//2-self.wheelbase
        y2 = rows//2
        croped = transformed[y1:y2,x1:x2,:]
        colission_pixels_red = np.where(croped[:,:,2] > 100)
        colission_pixels_green = np.where(croped[:,:,1] > 100)
        if colission_pixels_red[0].shape[0] > 0:
            return 'r'
        elif colission_pixels_green[0].shape[0] > 0:
            return 'g'
        else:
            return None

    ######## 
    # For Visualisation

    def get_chassis_points(self):
        T_M = self.get_transformation_matrix()
        # points are relative from middle of rear axcle. List of vectors
        pts = [[0, -self.track_width//2,1], [0, self.track_width//2,1], [self.wheelbase, self.track_width//2,1], [self.wheelbase, -self.track_width//2,1]]

        transformed = [T_M.dot(pt) for pt in pts]
        return np.array(transformed)[:,:-1]

    def get_wheel_points(self):
        T_M = self.get_transformation_matrix()
        fl_angle, fr_angle = self.calculate_steering_front()
        fl_R_M = np.concatenate((cv2.getRotationMatrix2D((self.wheelbase-self.wheel_length//2, -self.track_width//2-self.wheel_offset),math.degrees(fl_angle),1), np.array([[0,0,1]])))
        fr_R_M = np.concatenate((cv2.getRotationMatrix2D((self.wheelbase-self.wheel_length//2, self.track_width//2+self.wheel_offset),math.degrees(fr_angle),1), np.array([[0,0,1]])))
        
        fl = [[self.wheelbase-self.wheel_length, -self.track_width//2-self.wheel_offset,1], [self.wheelbase, -self.track_width//2-self.wheel_offset, 1]]
        fr = [[self.wheelbase-self.wheel_length, self.track_width//2+self.wheel_offset,1], [self.wheelbase, self.track_width//2+self.wheel_offset,1]]
        # rotate front wheels by steering angle
        fl = [(T_M @ fl_R_M).dot(pt) for pt in fl]
        fl = np.array(fl)[:,:-1]
        fr = [(T_M @ fr_R_M).dot(pt) for pt in fr]
        fr = np.array(fr)[:,:-1]

        rl = [[0, -self.track_width//2-self.wheel_offset,1], [self.wheel_length, -self.track_width//2-self.wheel_offset,1]]
        rr = [[0, self.track_width//2+self.wheel_offset,1], [self.wheel_length, self.track_width//2+self.wheel_offset,1]]

        rl = [T_M.dot(pt) for pt in rl]
        rl = np.array(rl)[:,:-1]
        rr = [T_M.dot(pt) for pt in rr]
        rr = np.array(rr)[:,:-1]
        return [fl, fr, rl, rr]

