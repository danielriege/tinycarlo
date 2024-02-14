import math
import json
import os
from typing import Dict, Any, Optional, List, Union, Tuple

# Types as for a map json file
Node = Tuple[int,int] # node is a tuple of x and y coordinates relative to the map frame
Edge = Tuple[int,int] # edge is a tuple of two node indices
LayerColor = Tuple[int,int,int]
Layer = Dict[str, Union[List[Node], List[Edge], LayerColor]] # key is either "nodes", "edges" or "layer_color"
Polylines = Dict[str, Layer] # key is the layer name
Trajectory = Layer # a trajectory has the same type as layer, only meaning is different
Trajectories = Dict[str, Trajectory] # key is the lane_path/layer name
MapData = Dict[str, Union[int, Polylines, Trajectories]] # key is either "height", "width", "polylines" or "lane_paths" (for trajectories)

class Map():
    def __init__(self, map_config: Dict[str, Any], base_path: Optional[str] = None):
        self.current_track: int = 0
        self.maps: List[Polylines] = [] # list of maps: each map is only a dict of nodes and edges per layer
        self.dimensions: List[Tuple[int,int]] = [] # dimension for each map (height, width)
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
    
    def get_map_size(self) -> Tuple[int,int]:
        """
        Returns the size of the current map as a tuple of (height, width)
        """
        return self.dimensions[self.current_track]


    def sample_spawn(self, np_random: Any) -> Tuple[Node, float]:
        """
        Returns a random spawn point for the current track, as position, rotation
        Sampled from the nodes of trajectory graph
        """
        random_node_idx = np_random.integers(0, len(self.trajectories[self.current_track]["nodes"])-1, size=1, dtype=int)[0]
        next_node = self.__get_next_nodes(random_node_idx)
        if len(next_node) == 0:
            return self.sample_spawn()
        # calculate the angle between the two nodes
        position = self.trajectories[self.current_track]["nodes"][random_node_idx]
        next_position = self.trajectories[self.current_track]["nodes"][next_node[0]]
        rotation = math.atan2(next_position[1]-position[1], next_position[0]-position[0])
        return position, rotation

    def __get_next_nodes(self, node_idx: int) -> List[int]:
        """
        Returns the next node idxs in the graph given a node index
        """
        return [e[1] for e in self.trajectories[self.current_track]["edges"] if e[0] == node_idx]
