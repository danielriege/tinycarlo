import numpy as np
import cv2

class Renderer():
    def __init__(self, track, car):
        self.track = track
        self.car = car

    def render_overview(self):
        image = self.track.get_track()

        car_pts = self.car.get_polyline_points()

        image = cv2.polylines(image, np.int32([car_pts]), True, (255,0,255), 5)
        return image

    def render_car_view(self):
        pass
    