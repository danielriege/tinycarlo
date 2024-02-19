import math
import numpy as np
import cv2
from typing import Any, List, Tuple, Dict, Optional
from tinycarlo.map import Map, Edge, Node
from tinycarlo.helper import clip_angle

class Car():
    def __init__(self, T: float, map: Map, car_config: Dict[str, Any]):
        self.map: Map = map
        self.track_width: float = car_config.get('track_width', 0.03)
        self.wheelbase: float = car_config.get('wheelbase', 0.08)
        self.max_velocity: float = car_config.get('max_velocity', 1)
        self.max_steering_angle: float = car_config.get('max_steering_angle', 35)
        self.steering_speed: Optional[float] = car_config.get('steering_speed', None)
        self.max_acceleration: float = car_config.get('max_acceleration', None)
        self.max_deceleration: float = car_config.get('max_deceleration', None)
        self.T: float = T

        self.wheel_offset: float = self.track_width/5 #from chassis
        self.wheel_length: float = self.wheelbase/3 
        self.wheel_width: float = self.wheel_length/6 

        self.position: Tuple[float, float] # position of middle of rear axcle
        self.position_front: Tuple[float, float] # position of middle of front axcle
        self.rotation: float
        self.steering_angle: float
        self.radius: float
        self.velocity: float
        self.local_path: List[Edge]
        self.last_maneuver: int

    def reset(self, np_random: Any) -> None:
        """
        Resets the position to a random spawn point and sets the steering angle to 0
        """
        self.position, self.rotation, nearest_edge = self.map.sample_spawn(np_random)
        self.local_path = [nearest_edge]
        self.__update_position_front()
        self.steering_angle = 0.0
        self.radius = 0.0
        self.velocity = 0.0
        self.last_maneuver = 0

    def get_info(self) -> Tuple[float, float, Dict[str, float], List[Node]]:
        empty_info = 0, 0, {layer_name: 0 for layer_name in self.map.get_laneline_names()}, []

        # calculate heading and cross track error by first updating nearest edge and next edge
        if self.local_path is None or len(self.local_path) < 2:
            return empty_info
        cte: float = self.map.lanepath.distance_to_edge(self.position_front, self.local_path[1])
        heading_error: float = clip_angle(self.map.lanepath.orientation_of_edge(self.local_path[1]) - self.rotation)

        # calculate distances to nearest edge and add to touching, if car is too close
        distances: Dict[str, float] = {}
        for i,layer_name in enumerate(self.map.get_laneline_names()):
            distance_to_nearest_front = abs(self.map.lanelines[i].distance_to_node(self.position_front, self.map.lanelines[i].get_nearest_node(self.position_front)))
            distance_to_nearest_rear = abs(self.map.lanelines[i].distance_to_node(self.position, self.map.lanelines[i].get_nearest_node(self.position)))
            distances[layer_name] = max(distance_to_nearest_front, distance_to_nearest_rear)
        # set local path for reference tracking. Instead of having a list of edges we want to have a list of coordinates
        local_path_coordinates = [self.map.lanepath.nodes[edge[1]] for edge in self.local_path]

        return cte, heading_error, distances, local_path_coordinates

    def step(self, velocity: float, steering_angle: float, maneuver: int) -> None:
        """
        Simulate one time step of the car given a velocity and a steering angle in [-1,1] range.
        Actual value depends on configured max_velocity and max_steering_angle.
        """
        dt: float = self.T

        # set velocity
        new_velocity = velocity * self.max_velocity
        if self.max_acceleration is not None:
            new_velocity = np.clip(new_velocity, self.velocity - self.max_deceleration * dt, self.velocity + self.max_acceleration * dt)
        self.velocity = new_velocity

        # Set steering angle
        new_steering_angle: float = steering_angle * self.max_steering_angle
        if self.steering_speed is not None:
            new_steering_angle = np.clip(new_steering_angle, self.steering_angle - self.steering_speed * dt, self.steering_angle + self.steering_speed * dt)
        self.steering_angle = new_steering_angle


        vxn: float = math.cos(self.rotation)
        vyn: float = math.sin(self.rotation)

        if self.steering_angle == 0:
            self.radius = 0

            self.position[0] = self.position[0] + self.velocity * vxn * dt 
            self.position[1] = self.position[1] + self.velocity * vyn * dt
        else:
            self.radius = self.wheelbase / (math.tan(math.radians(self.steering_angle)))
            ang_vel: float = self.velocity / self.radius
            dyaw: float = ang_vel * dt

            nx: float = vyn # normalvector
            ny: float = -vxn

            tx: float = nx * self.radius
            ty: float = ny * self.radius

            R_M: np.ndarray = np.array([[math.cos(dyaw), -math.sin(dyaw)],[math.sin(dyaw), math.cos(dyaw)]])

            rotated_vec: np.ndarray = R_M.dot([tx, ty])
        
            self.position[0] = self.position[0] - tx + rotated_vec[0]
            self.position[1] = self.position[1] - ty + rotated_vec[1]
        
            self.rotation += dyaw
            if self.rotation > math.pi:
                self.rotation -= 2 * math.pi
            elif self.rotation < -math.pi:
                self.rotation += 2 * math.pi
        self.__update_position_front()

        # calculate local path for reference tracking
        maneuver_dir_world_frame = clip_angle((self.map.lanepath.orientation_of_edge(self.local_path[0]) + maneuver * math.pi/2))
        if maneuver == 2 and self.last_maneuver != 2:
            # u turn maneuver
            nearest_edge = self.map.lanepath.get_nearest_edge_with_orientation(self.position_front, maneuver_dir_world_frame)
            maneuver_dir_world_frame = clip_angle(maneuver_dir_world_frame + math.pi)
        else:
            nearest_edge = self.map.lanepath.get_nearest_connected_edge(self.position_front, self.local_path[0], maneuver_dir_world_frame)
        self.last_maneuver = maneuver

        looking_ahead = 3 # nodes to look ahead for local path
        self.local_path = [nearest_edge]
        for _ in range(looking_ahead):
            last_edge = self.local_path[-1]
            if self.velocity > 0:
                next_edge = last_edge[1], self.map.lanepath.pick_node_given_orientation(last_edge[1], maneuver_dir_world_frame, self.map.lanepath.get_next_nodes(last_edge[1]))
            else:
                next_edge = last_edge[0], self.map.lanepath.pick_node_given_orientation(last_edge[0], maneuver_dir_world_frame, self.map.lanepath.get_prev_nodes(last_edge[0]))
            self.local_path.append(next_edge)

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
    
    def __update_position_front(self) -> None:
        self.position_front = (self.position[0] + self.wheelbase * math.cos(self.rotation), self.position[1] + self.wheelbase * math.sin(self.rotation))

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

