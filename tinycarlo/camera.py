import math
import numpy as np
import cv2

class Camera():
    def __init__(self, map, car, camera_config):
        self.map = map
        self.car = car
        self.resolution = camera_config.get('resolution', [128, 128])
        self.position = camera_config.get('position', [0, 0, 0])
        self.id = camera_config.get('id', 'unkown')
        self.orientation = camera_config.get('orientation', [0,0,0])
        self.fov = camera_config.get('fov', 90)

        self.E = self.__get_extrinsic_matrix()
        self.K = self.__get_intrinsic_matrix()
    
    ######## 
    # For Visualisation
    
    def get_frame_points(self):
        # points are relative from middle of rear axcle. List of vectors
        base_y = self.car.wheelbase + self.position[0]
        pts = [
            [base_y, -self.crop_size[1]//2 + self.position[1],1],
            [base_y + self.crop_size[0], -self.crop_size[1]//2 + self.position[1],1],
            [base_y +self.crop_size[0], +self.crop_size[1]//2 + self.position[1],1],
            [base_y, +self.crop_size[1]//2 + self.position[1],1]
        ]
        T_M = self.car.get_transformation_matrix()
        transformed = [T_M.dot(pt) for pt in pts]
        return np.array(transformed)[:,:-1]
    
    def capture_frame(self):
        """
        Captures a frame from the camera.
        """
        polylines = []
        for nodes, edges in zip(self.map.get_nodes(), self.map.get_edges()):
            points = np.array(nodes)
            # expand the points to 3D
            points = np.column_stack((points, np.zeros((len(points), 1))))
            
            car_pose = self.car.get_3d_transformation_matrix()
            camera_pose = self.E @ car_pose
            pp, points_camera_frame = self.__project_points(points, self.E, self.K)
            # No we have to handle points that are behind the camera
            depths = points_camera_frame[:,2]
            idx_in_front = np.where(depths < 0)[0] # indices of points that are in front of camera (positive z behind cam)
            idx_in_frame = np.where((pp[:,0] > 0) & (pp[:,0] < self.resolution[1]) & (pp[:,1] > 0) & (pp[:,1] < self.resolution[0]))[0] # indices of points in frame
            # combine 
            idx = np.intersect1d(idx_in_frame, idx_in_front)
            # Now, there is still possibility that e0 is in frame and in front of camera, but e1 is behind camera => glitch
            # for that, we change x and y value of the projected points which are behind the camera
            cx = self.resolution[0] / 2
            cy = self.resolution[1] / 2
            idx_behind = np.where(depths > 0)[0]
            #pp[idx_behind,0] = pp[idx_behind,0] + 2*-cx
            #pp[idx_behind,1] = pp[idx_behind,1] + 2*cx

            # now, out of the projected points we create a list of point pairs, which we can use to draw the lines
            list_of_pairs_for_layer = [(pp[e[0],:], pp[e[1],:]) for e in edges if e[0] in idx or e[1] in idx]
            
            polylines.append(list_of_pairs_for_layer)
        colors = self.map.get_colors()
        frame = self.__render_frame(polylines, colors)
        return frame

        
    def __render_frame(self, points, colors):
        """
        Renders the frame from the camera.
        """
        frame = np.zeros(self.resolution + [3], dtype=np.uint8)
        for point, color in zip(points, colors):
            for line in point:
                frame = cv2.polylines(frame, np.int32([line]), False, color, 1)
        return frame

    def __project_points(self, points, extrinsic_matrix, intrinsic_matrix):
        """
        Projects the given points to the camera plane.
        """
        points_3d_homogeneous = np.column_stack((points, np.ones((len(points), 1))))  # Convert to homogeneous coordinates
        # transform points into camera coordinate system
        points_camera = extrinsic_matrix @ points_3d_homogeneous.T

        # transform into image plane
        points_2d_homogeneous = intrinsic_matrix @ points_camera

        points_2d_normalized = points_2d_homogeneous / points_2d_homogeneous[2]

        return points_2d_normalized[:2].T, points_camera.T

    
    def __get_extrinsic_matrix(self):
        """
        Returns the extrinsic matrix of the camera.
        """
        # Convert Euler angles from degrees to radians
        angles_rad = np.radians(self.orientation + np.array([-90, 0, 90])) # these values are to align camera position relative to car looking forward
        # Convert Euler angles to rotation matrix using Rodrigues formula
        rotation_matrix_pr, _ = cv2.Rodrigues(np.array([1,1,0]) * angles_rad)
        rotation_matrix_y, _ = cv2.Rodrigues(np.array([0,0,1]) * angles_rad)
        # Construct the extrinsic matrix
        translation_matrix = np.column_stack((np.eye(3), -np.array(self.position)))
        return rotation_matrix_pr @ rotation_matrix_y @ translation_matrix
    
    def __get_intrinsic_matrix(self):
        """
        Returns the intrinsic matrix of the camera.
        """
        fx, fy = self.__focal_lengths_from_fov(self.fov, self.resolution)
        cx = self.resolution[0] / 2
        cy = self.resolution[1] / 2
        return np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])

    def __focal_lengths_from_fov(self, fov_deg, resolution):
        """
        Returns the focal lengths from the field of view and resolution.
        """
        fov_radians = np.radians(fov_deg)
        fx = resolution[0] / (2 * np.tan(fov_radians / 2))
        fy = resolution[1] / (2 * np.tan(fov_radians / 2))
        return fx, fy