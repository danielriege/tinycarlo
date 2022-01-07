import math
import numpy as np

class Car():
    def __init__(self, track_width, wheelbase, T):
        self.track_width = track_width
        self.wheelbase = wheelbase
        self.T = T
        self.reset()

    def reset(self):
        self.position = np.array([700,1460])
        self.rotation = 0.48

    def step(self, fwd_vel, steering_angle):
        fwd_vel *= 1 # max is 1 m/s
        steering_angle *= 33 # max steering is 33 degree
        dt = self.T
        ang_vel = fwd_vel / (self.wheelbase/1000 / (math.tan(math.radians(steering_angle)+0.000001)))
        dyaw = ang_vel * dt

        vx = fwd_vel * math.cos(self.rotation)
        vy = fwd_vel * math.sin(self.rotation)

        dx = (vx * math.cos(dyaw) - vy * math.sin(dyaw)) * dt
        dy = (vx * math.sin(dyaw) + vy * math.cos(dyaw)) * dt

        self.position[0] += dx*1000
        self.position[1] += dy*1000
        self.rotation += dyaw

    def get_polyline_points(self):
        R_M = np.array([[math.cos(self.rotation), -math.sin(self.rotation),0],[math.sin(self.rotation), math.cos(self.rotation),0], [0,0,1]])
        tx = self.position[0]
        ty = self.position[1]
        T_M = np.array([[1,0,tx], [0,1,ty], [0,0,1]])
        pts = [[0, -self.track_width//2,1], [0, self.track_width//2,1], [self.wheelbase, self.track_width//2,1], [self.wheelbase, -self.track_width//2,1]]

        rotated = [R_M.dot(pt) for pt in pts]
        translated = [T_M.dot(pt) for pt in rotated]
        return np.array(translated)[:,:-1]

