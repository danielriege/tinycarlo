from tinycarlo.helper import clip_angle
from typing import List, Tuple, Optional
import math

# Types as for a map json file
Node = Tuple[float,float] # node is a tuple of x and y coordinates relative to the map frame in meters
NodeIdx = int # node index
Edge = Tuple[NodeIdx,NodeIdx] # edge is a tuple of two node indices
LayerColor = Tuple[int,int,int] # RGB color tuple in range 0-255

class Layer():
    """
    Represents a layer in the map. A layer is a graph of nodes and edges and has a color and name.
    """
    def __init__(self, name: str, color: LayerColor, nodes: List[Node], edges: List[Edge]) -> None:
        self.name = name
        self.color = color
        self.nodes = nodes
        self.edges = edges

    def get_edge_coordinates_list(self) -> List[Tuple[Node, Node]]:
        """
        Returns the edges as a list of tuples of node coordinates.
        """
        return [(self.nodes[e[0]], self.nodes[e[1]]) for e in self.edges]
    
    def get_edge_coordinates(self, edge: Edge) -> Tuple[Node, Node]:
        """
        Returns the coordinates of the given edge.
        """
        return self.nodes[edge[0]], self.nodes[edge[1]]
    
    def get_nearest_edge(self, position: Tuple[float, float]) -> Edge:
        """
        Returns the nearest edge to the given position.

        Args:
            position (Tuple[float, float]): The position to find the nearest edge from in meters relative to map frame.

        Returns:
            Edge: The nearest edge to the given position.
        """
        d = [abs(self.distance(position, self.nodes[e[0]]) + self.distance(position, self.nodes[e[1]])) for e in self.edges]
        return self.edges[d.index(min(d))]
    
    def get_nearest_node(self, position: Tuple[float, float]) -> NodeIdx:
        """
        Returns the nearest node to the given position.

        Args:
            position (Tuple[float, float]): The position to find the nearest node from in meters relative to map frame.

        Returns:
            NodeIdx: The index of the nearest node to the given position.
        """
        d = [self.distance(position, n) for n in self.nodes]
        return d.index(min(d))
    
    def get_nearest_edge_with_orientation(self, position: Tuple[float, float], orientation: float) -> Optional[Edge]:
        """
        Returns the nearest edge to the given position with the specified orientation (+/- 30 deg).

        Args:
            position (Tuple[float, float]): The position to find the nearest edge from in meters relative to map frame.
            orientation (float): The orientation of the edge in radians relative to map frame.

        Returns:
            Edge: The nearest edge to the given position with the specified orientation (+/- 30 deg).
        """
        edges_within_orientation_range = [e for e in self.edges if abs(clip_angle(self.orientation_of_edge(e)-orientation)) <= math.radians(30)]
        if len(edges_within_orientation_range) == 0:
            return None
        d = [abs(self.distance(position, self.nodes[e[0]]) + self.distance(position, self.nodes[e[1]])) for e in edges_within_orientation_range]
        return edges_within_orientation_range[d.index(min(d))]

    
    def get_nearest_connected_edge(self, position: Tuple[float, float], edge: Edge, orientation: Optional[float] = None) -> Edge:
        """
        Returns the nearest connected edge to the given position.

        Args:
            position (Tuple[float, float]): The position to find the nearest connected edge from in meters relative to map frame.
            edge (Edge): The current edge.
            orientation (Optional[float], optional): The orientation of the edge in radians relative to map frame. Defaults to None.

        Returns:
            Edge: The nearest connected edge.

        """
        next_nodes, prev_nodes = self.get_next_nodes(edge[1]), self.get_prev_nodes(edge[0])
        next_node = self.pick_node_given_orientation(edge[1], orientation, next_nodes) if orientation is not None else next_nodes[0]
        prev_node = self.pick_node_given_orientation(edge[0], orientation, prev_nodes) if orientation is not None else prev_nodes[0]

        d0,d1 = self.distance(position, self.nodes[edge[0]]), self.distance(position, self.nodes[edge[1]])
        dn, dp = self.distance(position, self.nodes[next_node]), self.distance(position, self.nodes[prev_node])
        if dn < d0 and dn < d1:
            return edge[1], next_node
        elif dp < d0 and dp < d1:
            return prev_node, edge[0]
        else:
            return edge
    
    def pick_node_given_orientation(self, node_idx: NodeIdx, orientation: float, connected_nodes: List[NodeIdx]) -> Optional[NodeIdx]:
        """
        Picks a node from the given list of connected nodes based on the closest orientation to the specified orientation.

        Args:
            node_idx (NodeIdx): The index of the current node.
            orientation (float): The target orientation in radians relative to map frame.
            connected_nodes (List[NodeIdx]): The list of connected node indices.

        Returns:
            Optional[NodeIdx]: The index of the selected node, or None if the list is empty.
        """
        if len(connected_nodes) == 0:
            return None
        if len(connected_nodes) <= 1:
            return connected_nodes[0]
        n = self.nodes[node_idx]
        orientations_per_edge = [math.atan2(self.nodes[nn][1]-n[1], self.nodes[nn][0]-n[0]) for nn in connected_nodes]
        idx = min(range(len(orientations_per_edge)), key=lambda i: abs(clip_angle(orientations_per_edge[i]-orientation)))
        return connected_nodes[idx]
    
    def distance_to_edge(self, position: Tuple[float, float], edge: Edge) -> float:
        """
        Calculates the perpendicular distance from a given position to an edge.

        Args:
            position (Tuple[float, float]): The position coordinates (x, y) of the point in meters relative to map frame.
            edge (Edge): The edge to calculate the perpendicular distance to.

        Returns:
            float: The perpendicular distance from the position to the edge.
        """
        n1, n2 = self.nodes[edge[0]], self.nodes[edge[1]]
        line_vector = (n2[0]-n1[0], n2[1]-n1[1])
        position_vector = (position[0]-n1[0], position[1]-n1[1])
        if line_vector[0] == 0:
            if line_vector[1] > 0:
                return position[0] - n1[0]
            else:
                return n1[0] - position[0]
        # Calculate the perpendicular distance from point P to the line
        return (position_vector[0]*line_vector[1] - position_vector[1]*line_vector[0]) / math.sqrt(line_vector[0]**2 + line_vector[1]**2)
    
    def distance_to_node(self, position: Tuple[float, float], node_idx: NodeIdx) -> float:
        """
        Returns the distance to the given node.

        Args:
            position (Tuple[float, float]): The position to calculate the distance from in meters relative to map frame.
            node_idx (NodeIdx): The index of the node to calculate the distance to.

        Returns:
            float: The distance from the position to the node.
        """
        return self.distance(position, self.nodes[node_idx])
    
    def orientation_of_edge(self, edge: Edge) -> float:
        n1, n2 = self.nodes[edge[0]], self.nodes[edge[1]]
        return math.atan2(n2[1]-n1[1], n2[0]-n1[0])

    def get_next_nodes(self, node_idx: NodeIdx) -> List[NodeIdx]: return [e[1] for e in self.edges if e[0] == node_idx]
    
    def get_prev_nodes(self, node_idx: NodeIdx) -> List[NodeIdx]: return [e[0] for e in self.edges if e[1] == node_idx]
    
    def distance(self, node1: Node, node2: Node) -> float: return math.sqrt((node1[0]-node2[0])**2 + (node1[1]-node2[1])**2)