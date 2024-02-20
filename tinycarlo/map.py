import math
import json
import os
from typing import Dict, Any, Optional, List, Union, Tuple

from tinycarlo.layer import Layer, Node, Edge, LayerColor, NodeIdx

class Map():
    def __init__(self, map_config: Dict[str, Any], base_path: Optional[str] = None):
        self.lanelines: List[Layer] = [] # list of Layers for the lanelines
        self.lanepath: Optional[Layer] = None # Layer for the lanepath
        self.dimension: Tuple[float,float] = (0,0) # dimension of the map (height, width) in meters

        base_path = "./" if base_path is None else os.path.dirname(base_path)
        map_path: Any = os.path.join(base_path, map_config["json_path"])
        pixel_per_meter: int = map_config["pixel_per_meter"]
        with open(map_path) as f:
            map_data: Dict[str, Any] = json.load(f)
            # map data in json is not scaled to meters, so we need to do it here
            self.__change_scale(pixel_per_meter, map_data)
            
            self.lanelines = [Layer(name, layer["layer_color"], layer["nodes"], layer["edges"]) for name, layer in map_data["lanelines"].items()]
            self.lanepath = Layer("lanepath", map_data["lanepath"]["layer_color"], map_data["lanepath"]["nodes"], map_data["lanepath"]["edges"])
            self.dimension = (map_data["height"], map_data["width"])

    def __change_scale(self, pixel_per_meter: int, map_data: Dict[str, Any]) -> None:
        map_data["height"] = map_data["height"] / pixel_per_meter
        map_data["width"] = map_data["width"] / pixel_per_meter
        for _, poly_layer in map_data["lanelines"].items():
            for nodes in poly_layer["nodes"]:
                nodes[0] = nodes[0] / pixel_per_meter
                nodes[1] = nodes[1] / pixel_per_meter
        for nodes in map_data["lanepath"]["nodes"]:
            nodes[0] = nodes[0] / pixel_per_meter
            nodes[1] = nodes[1] / pixel_per_meter

    def get_laneline_names(self) -> List[str]: return [layer.name for layer in self.lanelines]
    
    def get_lanelines(self) -> List[List[Tuple[Node, Node]]]: return [layer.get_edge_coordinates_list() for layer in self.lanelines]

    def get_laneline_nodes(self) -> List[List[Node]]: return [layer.nodes for layer in self.lanelines]

    def get_laneline_edges(self) -> List[List[Edge]]: return [layer.edges for layer in self.lanelines]

    def get_lanepath(self) -> List[Tuple[Node, Node]]: return self.lanepath.get_edge_coordinates_list()
    
    def get_laneline_colors(self) -> List[LayerColor]: return [layer.color for layer in self.lanelines]
        
    def sample_spawn(self, np_random: Any) -> Tuple[Node, float, Edge]:
        """
        Randomly samples a spawn position and rotation on the lane path.

        Args:
            np_random (Any): The random number generator.

        Returns:
            Tuple[Node, float, Edge]: A tuple containing the spawn position, rotation, and edge information.
        """
        random_node_idx: NodeIdx = np_random.integers(0, len(self.lanepath.nodes)-1, size=1, dtype=int)[0]
        next_node = self.lanepath.get_next_nodes(random_node_idx)
        if len(next_node) == 0:
            return self.sample_spawn()
        # calculate the angle between the two nodes
        position = self.lanepath.nodes[random_node_idx].copy()
        next_position = self.lanepath.nodes[next_node[0]]
        rotation = math.atan2(next_position[1]-position[1], next_position[0]-position[0])
        return position, rotation, (random_node_idx, next_node[0])
