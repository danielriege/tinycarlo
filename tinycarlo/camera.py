import time
import numpy as np
import cv2
from typing import Any, List, Tuple, Dict, Optional

from tinycarlo.map import Map, Node, LayerColor
from tinycarlo.car import Car
from tinycarlo.renderer import Renderer
from tinycarlo.helper import getenv

class Camera():
    def __init__(self, map: Map, car: Car, renderer: Renderer, camera_config: Dict[str, Any]):
        self.map: Map = map
        self.car: Car = car
        self.renderer: Renderer = renderer
        self.resolution: Tuple[int, int] = camera_config.get('resolution', [128, 128])
        self.position: Tuple[float, float, float] = camera_config.get('position', [0, 0, 0])
        self.orientation: Tuple[int, int, int] = camera_config.get('orientation', [0,0,0])
        self.fov: int = camera_config.get('fov', 90)
        self.max_range: float = camera_config.get('max_range', None)
        self.line_thickness: int = camera_config.get('line_thickness', 1)

        self.E: np.ndarray = self.__get_extrinsic_matrix()
        self.K: np.ndarray = self.__get_intrinsic_matrix()

        self.last_frame_rgb: Optional[np.ndarray] = None
        self.last_frame_classes: Optional[np.ndarray] = None
    
    ######## 
    # For Visualisation
    
    def get_frame_points(self) -> np.ndarray:
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
    
    def get_last_frame_rgb(self) -> Optional[np.ndarray]: return self.last_frame_rgb
    
    def get_last_frame_classes(self) -> Optional[np.ndarray]: return self.last_frame_classes
    
    def capture_frame(self, format: str) -> np.ndarray:
        """
        Captures a frame from the camera.
        """
        polylines: List[List[Tuple[Node, Node]]] = []
        for nodes, edges in zip(self.map.get_laneline_nodes(), self.map.get_laneline_edges()):
            # expand the points to 3D by adding z = 0
            points: np.ndarray = np.column_stack((np.array(nodes), np.zeros((len(nodes), 1))))
            
            camera_pose = self.E @ self.car.get_3d_transformation_matrix()

            st_transform = time.perf_counter()
            points_camera_frame = self.__transform_into_camera_frame(points, camera_pose)
            td_transform = time.perf_counter() - st_transform

            depths = points_camera_frame[:,2]
            # Now, there is still possibility that e0 is in frame and in front of camera, but e1 is behind camera => glitch when projecting
            st_correct = time.perf_counter()
            idx_front = np.where(depths < 0)[0]
            for edge in [e for e in edges if e[0] not in idx_front and e[1] in idx_front]:
                points_camera_frame[edge[0],:] = self.__point_on_line_at_z(points_camera_frame[edge[1],:], points_camera_frame[edge[0],:])
                # needed otherwise this edge would not be considered if the other point is out of range or behind camera
                idx_front = np.append(idx_front, edge[0])
            for edge in [e for e in edges if e[0] in idx_front and e[1] not in idx_front]:
                points_camera_frame[edge[1],:] = self.__point_on_line_at_z(points_camera_frame[edge[0],:], points_camera_frame[edge[1],:])
                idx_front = np.append(idx_front, edge[1])
            td_correct = time.perf_counter() - st_correct
            # same for points out of range
            idx_in_range = np.where(depths > -self.max_range if self.max_range else True)[0]
            for edge in [e for e in edges if e[0] not in idx_in_range and e[1] in idx_in_range]:
                points_camera_frame[edge[0],:] = self.__point_on_line_at_z(points_camera_frame[edge[1],:], points_camera_frame[edge[0],:], -self.max_range)
                idx_in_range = np.append(idx_in_range, edge[0])
            for edge in [e for e in edges if e[0] in idx_in_range and e[1] not in idx_in_range]:
                points_camera_frame[edge[1],:] = self.__point_on_line_at_z(points_camera_frame[edge[0],:], points_camera_frame[edge[1],:], -self.max_range)
                idx_in_range = np.append(idx_in_range, edge[1])
            td_correct_range = time.perf_counter() - st_correct - td_correct

            pp = self.__project_to_image_plane(points_camera_frame, self.K)
            idx_in_frame = np.where((pp[:,0] > 0) & (pp[:,0] < self.resolution[1]) & (pp[:,1] > 0) & (pp[:,1] < self.resolution[0]))[0] # indices of points in frame
            # combine 
            idx = np.intersect1d(idx_in_frame, idx_front)
            idx = np.intersect1d(idx, idx_in_range)
            # now, out of the projected points we create a list of point pairs, which we can use to draw the lines
            list_of_pairs_for_layer: List[Tuple[Node, Node]] = [(pp[e[0],:], pp[e[1],:]) for e in edges if e[0] in idx or e[1] in idx]
            
            polylines.append(list_of_pairs_for_layer)

            if getenv("DEBUG"):
                print(f"obs node sz {len(nodes)} transform: {td_transform*1000:.4f} ms | cor: {td_correct*1000:.4f} ms | corr rng: {td_correct_range*1000:.4f} ms")
        colors: List[LayerColor] = self.map.get_laneline_colors()

        st_render = time.perf_counter()
        self.last_frame_rgb = self.renderer.render_camera_frame_rgb(polylines, colors, self.resolution, self.line_thickness)
        if format == "classes":
            self.last_frame_classes = self.renderer.render_camera_frame_classes(polylines, self.resolution, self.line_thickness)
        
        if getenv("DEBUG"):
            print(f"obs render: {(time.perf_counter()-st_render)*1000:.4f} ms")
        return self.last_frame_rgb if format == "rgb" else self.last_frame_classes

    def __point_on_line_at_z(self, p0: np.ndarray, p1: np.ndarray, target_z: float = -0.00001) -> Optional[np.ndarray]:
        """
        In 3D camera frame, an edge can go through the camera. This function returns a point on the line at z = 0, 
        so the edge can correctly be projected to the image plane.
        """
        direction = p0 - p1
        if direction[2] == 0:
            return None # line is parallel to z-axis
        t = (target_z - p1[2]) / direction[2]
        pz = p1 + t * direction
        return pz

    def __transform_into_camera_frame(self, points: np.ndarray, extrinsic_matrix: np.ndarray) -> np.ndarray:
        """
        Transforms the points from world coordinate system into camera coordinate system.
        z-axis is pointing behind the camera!
        """
        points_3d_homogeneous = np.column_stack((points, np.ones((len(points), 1))))  # Convert to homogeneous coordinates
        # transform points into camera coordinate system
        return (extrinsic_matrix @ points_3d_homogeneous.T).T
    
    def __project_to_image_plane(self, points_camera_frame: np.ndarray, intrinsic_matrix: np.ndarray) -> np.ndarray:
        """
        Projects the points onto the image plane.
        The points are expected to be in camera frame and in homogeneous coordinates.
        """
        points_2d_homogeneous = intrinsic_matrix @ points_camera_frame.T

        points_2d_normalized = points_2d_homogeneous / points_2d_homogeneous[2]

        return points_2d_normalized[:2].T

    
    def __get_extrinsic_matrix(self) -> np.ndarray:
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
    
    def __get_intrinsic_matrix(self) -> np.ndarray:
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

    def __focal_lengths_from_fov(self, fov_deg: int, resolution: Tuple[int, int]) -> Tuple[float, float]:
        """
        Returns the focal lengths from the field of view and resolution.
        """
        fov_radians = np.radians(fov_deg)
        fx = resolution[0] / (2 * np.tan(fov_radians / 2))
        fy = resolution[1] / (2 * np.tan(fov_radians / 2))
        return fx, fy