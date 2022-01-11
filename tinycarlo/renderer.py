import numpy as np
import cv2

class Renderer():
    def __init__(self, track, car, cameras):
        self.track = track
        self.car = car
        self.cameras = cameras

    def render_overview(self, loop_time, reward_sum, step_cnt):
        # Track render
        image = self.track.get_small_track()

        # car render
        car_pts = self.car.get_chassis_points()//2
        image = cv2.polylines(image, np.int32([car_pts]), True, (255,0,0), 3)
        for wheel in self.car.get_wheel_points():
            image = cv2.polylines(image, np.int32([wheel//2]), False, (255,0,255), self.car.wheel_width)

        # camera render
        for camera in self.cameras:
            camera_pts = camera.get_frame_points()//2
            image = cv2.polylines(image, np.int32([camera_pts]), True, (255,255,255), 3)

        # info render
        image = cv2.putText(image, f'possible: {int(1/loop_time)} FPS', (30,image.shape[0]-150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
        image = cv2.putText(image, f'steps: {step_cnt}', (30,image.shape[0]-100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
        image = cv2.putText(image, f'reward sum: {reward_sum}', (30,image.shape[0]-50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
        return image
    