import math
import json
from os import path

class Map():
    def __init__(self, map_config, base_path):
        self.current_track = 0
        self.maps = [] # list of maps: each map is only a dict of nodes and edges per layer
        self.dimensions = [] # dimension for each map (height, width)
        self.trajectories = [] # trajectory graph for each map

        if len(map_config) > 0:
            for i_map_config in map_config:
                if base_path is None:
                    base_path = "./"
                else:
                    base_path = path.dirname(base_path)
                map_path = path.join(base_path, i_map_config["json_path"])
                pixel_per_meter = i_map_config["pixel_per_meter"]
                with open(map_path) as f:
                    map_data = json.load(f)
                    self.__change_scale(pixel_per_meter, map_data)
                    self.maps.append(map_data["polylines"])
                    self.dimensions.append((map_data["height"], map_data["width"]))

                    # map could contain multiple lane_paths. We only use the first one
                    # TODO: support multiple lane_paths by flattening dicts
                    lane_paths = map_data["lane_paths"]
                    self.trajectories.append(lane_paths[list(lane_paths.keys())[0]])
        else:
            print("Error: No maps provided")
            exit()

    def __change_scale(self, pixel_per_meter, map_data):
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

    
    def get_polylines(self):
        """
        Returns the polylines needed to render the map.
        Return value is a Tuple of (list[polylines], list[color]). Polylines is a list of tuples of points, each being a tuple of x,y coordinates.
        """
        map_data = self.maps[self.current_track]
        polylines = [[(p["nodes"][e[0]], p["nodes"][e[1]]) for e in p["edges"]] for p in map_data.values()]
        colors = [polyline_dict["layer_color"] for polyline_dict in map_data.values()]
        return polylines, colors
    
    def get_nodes(self):
        """
        Returns the nodes of the current map as for polyline 
        """
        return[p["nodes"] for p in self.maps[self.current_track].values()]
    
    def get_edges(self):
        """
        Returns the edges of the current map as for polyline 
        """
        return[p["edges"] for p in self.maps[self.current_track].values()]
    
    def get_colors(self):
        """
        Returns the colors of the current map as for polyline 
        """
        return [polyline_dict["layer_color"] for polyline_dict in self.maps[self.current_track].values()]
    
    def get_map_size(self):
        """
        Returns the size of the current map as a tuple of (height, width)
        """
        return self.dimensions[self.current_track]


    def sample_spawn(self, np_random):
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

    def set_next_track(self):
        self.current_track = random.randint(0,len(self.maps)-1)

    def __get_next_nodes(self, node):
        """
        Returns the next nodes in the graph given a node
        """
        return [e[1] for e in self.trajectories[self.current_track]["edges"] if e[0] == node]
