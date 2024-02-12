import numpy as np
import cv2

class Renderer():
    def __init__(self, map, car, cameras, overview_pixel_per_meter=1000):
        self.map = map
        self.car = car
        self.cameras = cameras
        self.overview_pixel_per_meter =  overview_pixel_per_meter
        self.static_overview = self.__render_static_overview()

    def render_overview(self):
        # Map render
        image = self.static_overview.copy()

        #car render
        car_pts = self.car.get_chassis_points()
        image = cv2.polylines(image, self.__scale_points(car_pts), True, (255,0,0), 3)
        for wheel in self.car.get_wheel_points():
            image = cv2.polylines(image, self.__scale_points(wheel), False, (255,0,255), np.int32(self.car.wheel_width * self.overview_pixel_per_meter))

        # camera render
        if self.cameras:
            for camera in self.cameras:
                camera_pts = camera.get_frame_points()
                image = cv2.polylines(image, np.int32([camera_pts]), True, (255,255,255), 3)

        return image

    def __render_static_overview(self):
        """
        Renders the static parts of the overview, which will only be rendered once.
        """
        height, width = self.map.get_map_size()
        overview = np.zeros((int(height*self.overview_pixel_per_meter), int(width*self.overview_pixel_per_meter), 3), dtype=np.uint8)
        # Map render
        for polyline, color in zip(*self.map.get_polylines()):
            for line in polyline:
                overview = cv2.polylines(overview, self.__scale_points(line), False, color, 3)
        return overview
    
    def __scale_points(self, points):
        return np.int32(np.array([points]) * self.overview_pixel_per_meter)
