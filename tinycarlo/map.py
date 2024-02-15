import math
import json
import os
from typing import Dict, Any, Optional, List, Union, Tuple

# Types as for a map json file
Node = Tuple[float,float] # node is a tuple of x and y coordinates relative to the map frame
Edge = Tuple[int,int] # edge is a tuple of two node indices
LayerColor = Tuple[int,int,int]
Layer = Dict[str, Union[List[Node], List[Edge], LayerColor]] # key is either "nodes", "edges" or "layer_color"
Polylines = Dict[str, Layer] # key is the layer name
Trajectory = Layer # a trajectory has the same type as layer, only meaning is different
Trajectories = Dict[str, Trajectory] # key is the lane_path/layer name
MapData = Dict[str, Union[float, Polylines, Trajectories]] # key is either "height", "width", "polylines" or "lane_paths" (for trajectories)

class Map():
    def __init__(self, map_config: Dict[str, Any], base_path: Optional[str] = None):
        self.current_track: int = 0
        self.maps: List[Polylines] = [] # list of maps: each map is only a dict of nodes and edges per layer
        self.dimensions: List[Tuple[float,float]] = [] # dimension for each map (height, width)
        self.trajectories: List[Trajectory] = [] # trajectory graph for each map

        if len(map_config) > 0:
            for i_map_config in map_config:
                if base_path is None:
                    base_path = "./"
                else:
                    base_path = os.path.dirname(base_path)
                map_path: Any = os.path.join(base_path, i_map_config["json_path"])
                pixel_per_meter = i_map_config["pixel_per_meter"]
                with open(map_path) as f:
                    map_data: MapData = json.load(f)
                    self.__change_scale(pixel_per_meter, map_data)
                    self.maps.append(map_data["polylines"])
                    self.dimensions.append((map_data["height"], map_data["width"]))

                    # map could contain multiple lane_paths. We only use the first one
                    # TODO: support multiple lane_paths by flattening dicts
                    lane_paths: Trajectories = map_data["lane_paths"]
                    self.trajectories.append(lane_paths[list(lane_paths.keys())[0]])
        else:
            print("Error: No maps provided")
            exit()

    def __change_scale(self, pixel_per_meter: int, map_data: MapData) -> None:
        map_data["height"] = map_data["height"] / pixel_per_meter
        map_data["width"] = map_data["width"] / pixel_per_meter
        for _, poly_layer in map_data["polylines"].items():
            for nodes in poly_layer["nodes"]:
                nodes[0] = nodes[0] / pixel_per_meter
                nodes[1] = nodes[1] / pixel_per_meter
        for _,graph in map_data["lane_paths"].items():
            for nodes in graph["nodes"]:
                nodes[0] = nodes[0] / pixel_per_meter
                nodes[1] = nodes[1] / pixel_per_meter

    
    def get_polylines(self) -> Tuple[List[List[Tuple[Node, Node]]], List[LayerColor]]:
        """
        Returns the polylines needed to render the map.

        Return value is a Tuple of two lists which can be zipped.
        First list contains the polylines as list of tuples of nodes (node pairs).
        Second list contains the colors for each polyline.
        """
        map_data = self.maps[self.current_track]
        polylines = [[(p["nodes"][e[0]], p["nodes"][e[1]]) for e in p["edges"]] for p in map_data.values()]
        colors = [polyline_dict["layer_color"] for polyline_dict in map_data.values()]
        return polylines, colors
    
    def get_nodes(self) -> List[List[Node]]:
        """
        Returns the nodes of the current map for polyline
        """
        return[p["nodes"] for p in self.maps[self.current_track].values()]
    
    def get_trajectory_nodes(self) -> List[Node]:
        """
        Returns the nodes of the current map for trajectory
        """
        return self.trajectories[self.current_track]["nodes"]
    
    def get_edges(self) -> List[List[Edge]]:
        """
        Returns the edges of the current map for polyline 
        """
        return[p["edges"] for p in self.maps[self.current_track].values()]
    
    def get_colors(self) -> List[LayerColor]:
        """
        Returns the colors of the current map for polyline 
        """
        return [polyline_dict["layer_color"] for polyline_dict in self.maps[self.current_track].values()]
    
    def get_map_size(self) -> Tuple[float,float]:
        """
        Returns the size of the current map as a tuple of (height, width)
        """
        return self.dimensions[self.current_track]
    
    def get_nearest_edge(self, position: Tuple[float, float], current_edge: int, orientation: Optional[float] = None) -> Tuple[int, int]:
        """
        Given a current edge, returns the nearest edge to the position. So its either the current (if the closest) or one of the connected edges.
        Optional: Edges can have multiple connections, so the orientation can be used to determine the next edge.
        """
        if orientation is not None:
            next_node = self.get_next_node(current_edge[1], orientation)
            prev_node = self.get_prev_node(current_edge[0], orientation)
        else:
            next_node = self.__get_next_nodes(current_edge[1])[0]
            prev_node = self.__get_prev_nodes(current_edge[0])[0]
        d0 = self.__distance(position, self.trajectories[self.current_track]["nodes"][current_edge[0]])
        d1 = self.__distance(position, self.trajectories[self.current_track]["nodes"][current_edge[1]])
        dn = self.__distance(position, self.trajectories[self.current_track]["nodes"][next_node])
        dp = self.__distance(position, self.trajectories[self.current_track]["nodes"][prev_node])
        if dn < dp:
            return (current_edge[1], next_node)
        elif dp < d0:
            return (prev_node, current_edge[0])
        else:
            return current_edge
    
    def get_cte_and_headingerror(self, position: Tuple[float, float], rotation: float, n1_idx: int, n2_idx: int) -> Tuple[float, float]:
        """
        Given a position and rotation, returns the cross track error and the heading error to the line between n1 and n2.
        rotation is expected in radians.
        """
        n1, n2 = self.trajectories[self.current_track]["nodes"][n1_idx], self.trajectories[self.current_track]["nodes"][n2_idx]
        # calculate the angle between the two nodes
        angle = math.atan2(n2[1]-n1[1], n2[0]-n1[0])
        # calculate the angle between the car and the line between the two nodes
        angle_diff = angle - rotation
        if angle_diff < -math.pi:
            angle_diff += 2*math.pi
        if angle_diff > math.pi:
            angle_diff -= 2*math.pi
        # calculate the distance between the car and the line between the two nodes
        line_vector = (n2[0]-n1[0], n2[1]-n1[1])
        position_vector = (position[0]-n1[0], position[1]-n1[1])
        # Calculate the perpendicular distance from point P to the line
        cte = position_vector[0]*line_vector[1] - position_vector[1]*line_vector[0] / math.sqrt(line_vector[0]**2 + line_vector[1]**2)
        return cte, angle_diff
        
    def sample_spawn(self, np_random: Any) -> Tuple[Node, float, int]:
        """
        Returns a random spawn point for the current track, as position, rotation
        Sampled from the nodes of trajectory graph
        """
        random_node_idx = np_random.integers(0, len(self.trajectories[self.current_track]["nodes"])-1, size=1, dtype=int)[0]
        next_node = self.__get_next_nodes(random_node_idx)
        if len(next_node) == 0:
            return self.sample_spawn()
        # calculate the angle between the two nodes
        position = self.trajectories[self.current_track]["nodes"][random_node_idx].copy()
        next_position = self.trajectories[self.current_track]["nodes"][next_node[0]]
        rotation = math.atan2(next_position[1]-position[1], next_position[0]-position[0])
        return position, rotation, (random_node_idx, next_node[0])
    
    def __distance(self, a: Node, b: Node) -> float:
        """
        Returns the distance between two nodes
        """
        return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)
    
    def get_next_node(self, node_idx: int, orientation: float) -> int:
        """
        Returns the next node idx in the graph given a node index and an orientation.
        orientation is expected to be in radians (relative to world frame).
        """
        n = self.trajectories[self.current_track]["nodes"][node_idx]
        next_nodes = self.__get_next_nodes(node_idx)
        if len(next_nodes) == 0:
            return next_nodes[0]       
        orientations_per_edge = [math.atan2(next[1]-n[1], next[0]-n[0]) for next in [self.trajectories[self.current_track]["nodes"][nn] for nn in next_nodes]]
        # get idx of the edge with the closest orientation to the car
        idx = min(range(len(orientations_per_edge)), key=lambda i: abs(orientations_per_edge[i]-orientation))
        return next_nodes[idx]
    
    def get_prev_node(self, node_idx: int, orientation: float) -> int:
        """
        Similar as get_next_node, but returns the previous node.
        Note: Orientation will be reversed automatically.
        """
        orientation = orientation + math.pi
        n = self.trajectories[self.current_track]["nodes"][node_idx]
        prev_nodes = self.__get_prev_nodes(node_idx)
        if len(prev_nodes) == 0:
            return prev_nodes[0]        
        orientations_per_edge = [math.atan2(prev[1]-n[1], prev[0]-n[0]) for prev in [self.trajectories[self.current_track]["nodes"][pn] for pn in prev_nodes]]
        # get idx of the edge with the closest orientation to the car
        idx = min(range(len(orientations_per_edge)), key=lambda i: abs(orientations_per_edge[i]-orientation))
        return prev_nodes[idx]

    def __get_next_nodes(self, node_idx: int) -> List[int]:
        """
        Returns the next node idxs in the graph given a node index
        """
        return [e[1] for e in self.trajectories[self.current_track]["edges"] if e[0] == node_idx]
    
    def __get_prev_nodes(self, node_idx: int) -> List[int]:
        """
        Returns the previous node idxs in the graph given a node index
        """
        return [e[0] for e in self.trajectories[self.current_track]["edges"] if e[1] == node_idx]
