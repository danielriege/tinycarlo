import numpy as np
import cv2

class Renderer():
    def __init__(self, track, car):
        self.track = track
        self.car = car

    def render_overview(self, loop_time):
        image = self.track.get_track()

        car_pts = self.car.get_polyline_points()

        image = cv2.polylines(image, np.int32([car_pts]), True, (255,0,255), 5)

        image = cv2.putText(image, f'possible: {int(1/loop_time)} FPS', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
        return image
    