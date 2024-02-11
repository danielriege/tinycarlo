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

            
            camera_pose = self.E 
            pp = self.__project_points(points, camera_pose, self.K)
            # filter out points that are not in the frame
            # we need the index of the points that are in the frame
            indices = np.where((pp[:,0] > 0) & (pp[:,0] < self.resolution[0]) & (pp[:,1] > 0) & (pp[:,1] < self.resolution[1]))[0]
            # in edges, keep only the tuples that contain one of the indices
            edges = [e for e in edges if e[0] in indices or e[1] in indices]
            # now, out of the projected points we create a list of point pairs, which we can use to draw the lines
            list_of_pairs_for_layer = [(pp[e[0],:], pp[e[1],:]) for e in edges]
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
        intrinic = np.eye(4)
        intrinic[:3, :3] = intrinsic_matrix
        camera_matrix = intrinic @ extrinsic_matrix
        points_3d_homogeneous = np.column_stack((points, np.ones((len(points), 1))))  # Convert to homogeneous coordinates

        points_2d_homogeneous = camera_matrix @ points_3d_homogeneous.T

        points_2d_normalized = points_2d_homogeneous / points_2d_homogeneous[2]

        return points_2d_normalized[:2].T

    
    def __get_extrinsic_matrix(self):
        """
        Returns the extrinsic matrix of the camera.
        """
        # Convert Euler angles from degrees to radians
        angles_rad = np.radians(self.orientation)
        # Convert Euler angles to rotation matrix using Rodrigues formula
        rotation_matrix, _ = cv2.Rodrigues(angles_rad)
        # Construct the extrinsic matrix
        E = np.eye(4)
        E[:3, :3] = rotation_matrix
        E[:3, 3] = self.position
        return E
    
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