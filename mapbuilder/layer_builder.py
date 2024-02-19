import numpy as np
import cv2

from typing import List, Tuple, Dict, Any, Optional

Node = Tuple[int, int]
Edge = Tuple[int, int]

class LayerBuilder:
    """
    Builds up a layer on the map. A layer consists of a graph which has nodes and edges. 
    Not all nodes have to be connected. We dont care about that, so it is possible to represent several closed graphs in one layer.
    image is the background image on which the layer is drawn.
    """
    
    def __init__(self, layer_name: str, layer_color: Tuple[int, int, int], is_lane_path: bool, image: np.ndarray):
        self.nodes: List[Node] = []
        self.edges: List[Edge] = []
        self.layer_name: str = layer_name
        self.layer_color: Tuple[int, int, int] = layer_color
        self.image: np.ndarray = image
        self.is_lane_path: bool = is_lane_path  

        self.select_node_threshold: int = 10
        self.selected_node_idx: Optional[int] = None

    def render_current_layer(self, final: bool=False) -> np.ndarray:
        """
        Renders the current layer to the background image.
        This is needed to visualize dynamic changes to the layer like selected node etc.
        This does all the drawing.
        Should therefore be called in a loop.
        Returns the rendered image, which also contains the background image.
        """
        render_image = self.image.copy() # Copy the image to not change the original

        for e in self.edges:
            n1 = self.nodes[e[0]]
            n2 = self.nodes[e[1]]
            if self.is_lane_path:
                cv2.arrowedLine(render_image, (n1[0], n1[1]), (n2[0], n2[1]), (0,0,0), 4)
            else:
                cv2.line(render_image, (n1[0], n1[1]), (n2[0], n2[1]), self.layer_color, 2)
        
        if not final:
            for i,n in enumerate(self.nodes):
                if i != self.selected_node_idx:
                    cv2.drawMarker(render_image, (n[0], n[1]), (0,0,255), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)
                else:
                    cv2.drawMarker(render_image, (n[0], n[1]), (255,0,0), markerType=cv2.MARKER_TILTED_CROSS, markerSize=10, thickness=2)

        return render_image
    
    def render_final(self) -> np.ndarray:
        self.selected_node_idx = None
        return self.render_current_layer(final=True)
    
    def build_layer_dict(self) -> Tuple[str, Dict[str, Any]]:
        """
        This should be called after bulding the layer is finished.
        It returns a dictionary which contains the graph information.
        """
        layer_dict = {
            "layer_color": self.layer_color,
            "nodes": self.nodes,
            "edges": self.edges
        }
        return self.layer_name, layer_dict

    def select_node(self, x: int, y: int) -> None:
        """
        Selects a node on the layer. This is used to select a node to add an edge to.
        Given the coordinates the closest node is selected, if the distance is less than a threshold.
        """
        nearest_node_idx = self.find_nearest_node(x, y)
        if nearest_node_idx is not None:
            self.selected_node_idx = nearest_node_idx

    def add_new_node(self, x: int, y: int) -> None:
        """
        Adds a new node to the layer. If a node is already selected, an edge is added between the selected node and the new node.
        The new node is then selected.
        """
        self.nodes.append((x, y))
        if self.selected_node_idx is not None:
            self.edges.append((self.selected_node_idx, len(self.nodes) - 1))

        # Select the new node
        self.selected_node_idx = len(self.nodes) - 1

    def loop_closure(self, nearest_node_idx: Optional[int]) -> None:
        if nearest_node_idx is not None and self.selected_node_idx is not None:
            self.edges.append((self.selected_node_idx, nearest_node_idx))

    def undo(self) -> None:
        """
        Undoes the last action.
        """
        self.nodes.pop()
        self.edges.pop()
        self.selected_node_idx = len(self.nodes) - 1

    def reset(self) -> None:
        """
        Resets the layer to the initial state.
        """
        self.nodes = []
        self.edges = []
        self.selected_node_idx = None

    def move_selected_node(self, x: int, y: int) -> None:
        """
        Moves the selected node to the given coordinates.
        """
        if self.selected_node_idx is not None:
            self.nodes[self.selected_node_idx] = (x, y)

    def find_nearest_node(self, x: int,y: int) -> Optional[int]:
        """
        Finds the nearest node to the given coordinates.
        If the distance is less than a threshold, the node index is returned.
        """
        node_idx = None
        for i,n in enumerate(self.nodes):
            if np.linalg.norm(np.array(n) - np.array([x, y])) < self.select_node_threshold:
                node_idx = i
                break
        return node_idx