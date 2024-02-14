import math
import numpy as np
import cv2
from typing import Any, List, Tuple, Dict, Optional
from tinycarlo.map import Map, Node

class Car():
    def __init__(self, T: float, map: Map, car_config: Dict[str, Any]):
        self.map: Map = map
        self.track_width: float = car_config.get('track_width', 0.03)
        self.wheelbase: float = car_config.get('wheelbase', 0.08)
        self.max_steering_change: Optional[float] = car_config.get('max_steering_change', None)
        self.T: float = T

        self.wheel_offset: float = self.track_width/5 #from chassis
        self.wheel_length: float = self.wheelbase/3 
        self.wheel_width: float = self.wheel_length/6 

        self.position: Node
        self.rotation: float
        self.steering_angle: float
        self.steering_input: float
        self.radius: float

    def reset(self, np_random: Any) -> None:
        """
        Resets the position to a random spawn point and sets the steering angle to 0
        """
        self.position, self.rotation = self.map.sample_spawn(np_random)
        self.steering_angle = 0.0
        self.steering_input = 0.0
        self.radius = 0.0

    def step(self, fwd_vel: float, steering_angle: float) -> None:
        dt: float = self.T

        fwd_vel *= 1 # max is 1 m/s
        self.steering_input = steering_angle
        new_steering_angle: float = steering_angle * 33 # max steering is 33 degree
        if self.max_steering_change is None:
            self.steering_angle = new_steering_angle
        else:
            # apply smoothness
            max_steering_in_T: float = self.max_steering_change*self.T
            steering_change: float = np.clip(new_steering_angle - self.steering_angle, -max_steering_in_T, max_steering_in_T)
            self.steering_angle = self.steering_angle + steering_change

        vxn: float = math.cos(self.rotation)
        vyn: float = math.sin(self.rotation)

        if self.steering_angle == 0:
            self.radius = 0

            self.position[0] = self.position[0] + fwd_vel * vxn * dt * 1000
            self.position[1] = self.position[1] + fwd_vel * vyn * dt * 1000
        else:
            self.radius = self.wheelbase/1000 / (math.tan(math.radians(self.steering_angle)))
            ang_vel: float = fwd_vel / self.radius
            dyaw: float = ang_vel * dt

            nx: float = vyn # normalvector
            ny: float = -vxn

            tx: float = nx * self.radius * 1000
            ty: float = ny * self.radius * 1000

            R_M: np.ndarray = np.array([[math.cos(dyaw), -math.sin(dyaw)],[math.sin(dyaw), math.cos(dyaw)]])

            rotated_vec: np.ndarray = R_M.dot([tx, ty])
        
            self.position[0] = self.position[0] - tx + rotated_vec[0]
            self.position[1] = self.position[1] - ty + rotated_vec[1]
        
            self.rotation += dyaw

    def get_transformation_matrix(self) -> np.ndarray:
        ''' 
        Returns a Transformation matrix which points to middle of rear axcle in world
        '''
        R_M: np.ndarray = np.array([[math.cos(self.rotation), -math.sin(self.rotation),0],[math.sin(self.rotation), math.cos(self.rotation),0], [0,0,1,]])
        T_M: np.ndarray = np.array([[1,0,self.position[0]], [0,1,self.position[1]], [0,0,1]])
        return T_M @ R_M
    
    def get_3d_transformation_matrix(self) -> np.ndarray:
        ''' 
        Returns a Transformation matrix which points to middle of rear axcle in world
        '''
        R_M: np.ndarray = np.array([[math.cos(-self.rotation), -math.sin(-self.rotation),0, 0],[math.sin(-self.rotation), math.cos(-self.rotation),0, 0], [0,0,1,0], [0,0,0,1]])
        T_M: np.ndarray = np.array([[1,0,0,-self.position[0]], [0,1,0,-self.position[1]], [0,0,1,0], [0,0,0,1]])
        return R_M @ T_M

    # For Visualisation

    def get_chassis_points(self) -> np.ndarray:
        T_M = self.get_transformation_matrix()
        # points are relative from middle of rear axcle. List of vectors
        pts: List[Tuple[float, float, float]] = [[0, -self.track_width/2,1], 
        [0, self.track_width/2,1], 
        [self.wheelbase, self.track_width/2,1], 
        [self.wheelbase, -self.track_width/2,1]]

        transformed: np.ndarray = [T_M.dot(pt) for pt in pts]
        return np.array(transformed)[:,:-1]

    def get_wheel_points(self) -> List[np.ndarray]:
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
    
    def __ackermann_steering(self) -> Tuple[float, float]:
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

