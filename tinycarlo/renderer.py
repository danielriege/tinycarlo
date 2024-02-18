import numpy as np
import cv2
from typing import List, Tuple

from tinycarlo.map import Map, Node, LayerColor
from tinycarlo.car import Car
from tinycarlo.helper import getenv

class Renderer():
    def __init__(self, map: Map, car: Car, overview_pixel_per_meter: int = 1000):
        self.map: Map = map
        self.car: Car = car
        self.overview_pixel_per_meter: int =  overview_pixel_per_meter
        self.static_overview: np.ndarray = self.__render_static_overview()

    def render_overview(self) -> np.ndarray:
        # Map render
        image = self.static_overview.copy()

        #car render
        car_pts = self.car.get_chassis_points()
        image = cv2.polylines(image, self.__scale_points(car_pts), True, (255,0,0), 3)
        for wheel in self.car.get_wheel_points():
            image = cv2.polylines(image, self.__scale_points(wheel), False, (255,0,255), np.int32(self.car.wheel_width * self.overview_pixel_per_meter))

        # car local path render
        for edge in self.car.local_path:
            image = cv2.polylines(image, self.__scale_points([self.map.lanepath.nodes[edge[0]], self.map.lanepath.nodes[edge[1]]]), False, (255,0,0), 3)

        return image
    
    def render_camera_frame_rgb(self, points: List[List[Tuple[Node, Node]]], colors: List[LayerColor], resolution: Tuple[int, int], line_thickness: int) -> np.ndarray:
        """
        Renders RGB camera frame given np.array of points and colors.
        """
        frame = np.zeros(resolution + [3], dtype=np.uint8)
        for point, color in zip(points, colors):
            for line in point:
                frame = cv2.polylines(frame, np.int32([line]), False, color, line_thickness)
        return frame

    def __render_static_overview(self) -> np.ndarray:
        """
        Renders the static parts of the overview, which will only be rendered once.
        """
        height, width = self.map.dimension
        overview = np.zeros((int(height*self.overview_pixel_per_meter), int(width*self.overview_pixel_per_meter), 3), dtype=np.uint8)
        # Map render
        for polyline, color in zip(self.map.get_lanelines(), self.map.get_laneline_colors()):
            for line in polyline:
                overview = cv2.polylines(overview, self.__scale_points(line), False, color, 3)
        
         # render car paths
        path = self.map.get_lanepath()
        for line in path:
            overview = cv2.polylines(overview, self.__scale_points(line), False, (50,50,50), 3)
        return overview
    
    def __scale_points(self, points: np.ndarray) -> np.ndarray:
        return np.int32(np.array([points]) * self.overview_pixel_per_meter)
