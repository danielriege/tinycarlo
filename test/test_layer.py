import unittest
from tinycarlo.layer import Layer
import math


def build_test_layer(nodes, edges):
    return Layer("test", (0, 0, 0), nodes, edges)

def length(v):
    return math.sqrt(v[0]**2 + v[1]**2)

def assert_close(a, b, tol=1e-5):
    assert abs(a-b) < tol

class LayerTestCase(unittest.TestCase):

    def test_get_edge_coordinates_list(self):
        n = [(0, 0), (1, 1), (2, 2), (3, 3)]
        e = [(0, 1), (1, 2), (2, 3), (3, 1)]
        layer = build_test_layer(n, e)
        assert layer.get_edge_coordinates_list() == [((0, 0), (1, 1)), ((1, 1), (2, 2)), ((2, 2), (3, 3)), ((3, 3), (1, 1))]

    def test_get_edge_coordinates(self):
        n = [(0, 0), (1, 1), (2, 2), (3, 3)]
        e = [(0, 1), (1, 2), (2, 3), (3, 1)]
        layer = build_test_layer(n, e)
        assert layer.get_edge_coordinates(e[0]) == ((0, 0), (1, 1))
        assert layer.get_edge_coordinates(e[1]) == ((1, 1), (2, 2))
        assert layer.get_edge_coordinates(e[2]) == ((2, 2), (3, 3))
        assert layer.get_edge_coordinates(e[3]) == ((3, 3), (1, 1))

    def test_get_nearest_edge(self):
        n = [(0,0), (4,0), (0,4), (4,4)]
        e = [(0,1), (2,3)]
        layer = build_test_layer(n, e)
        assert layer.get_nearest_edge((0,1)) == e[0]
        assert layer.get_nearest_edge((4,1)) == e[0]
        assert layer.get_nearest_edge((1,0)) == e[0]
        assert layer.get_nearest_edge((1,4)) == e[1]
        assert layer.get_nearest_edge((1,5)) == e[1]
        # test negative positions
        assert layer.get_nearest_edge((0,-1)) == e[0]
        assert layer.get_nearest_edge((-1,0)) == e[0]
        assert layer.get_nearest_edge((-1,-1)) == e[0]
        assert layer.get_nearest_edge((-1,5)) == e[1]
        # test edge cases
        assert layer.get_nearest_edge((0,2.01)) == e[1]
        assert layer.get_nearest_edge((0,1.99)) == e[0]
        assert layer.get_nearest_edge((2,2.01)) == e[1]
        assert layer.get_nearest_edge((2,1.99)) == e[0]
        # at a tie, we expect first in list
        assert layer.get_nearest_edge((2,2)) == e[0]
        # Test with perpendicular edges
        n = [(0,0), (3,0), (3,3)]
        e = [(0,1), (1,2)]
        layer = build_test_layer(n, e)
        assert layer.get_nearest_edge((0,3)) == e[0] # tie
        assert layer.get_nearest_edge((1,1)) == e[0]
        assert layer.get_nearest_edge((1,2)) == e[0]
        assert layer.get_nearest_edge((1,3)) == e[1]
        assert layer.get_nearest_edge((1,4)) == e[1]
        assert layer.get_nearest_edge((2,1)) == e[0] # tie
        assert layer.get_nearest_edge((2,2)) == e[1]
        assert layer.get_nearest_edge((4,0)) == e[1]
        assert layer.get_nearest_edge((3,-1)) == e[0]

    def test_get_nearest_node(self):
        n = [(0,0), (4,0), (0,4), (4,4)]
        layer = build_test_layer(n, [])
        assert layer.get_nearest_node((0,1)) == 0
        assert layer.get_nearest_node((4,1)) == 1
        assert layer.get_nearest_node((1,0)) == 0
        assert layer.get_nearest_node((1,4)) == 2
        assert layer.get_nearest_node((1,5)) == 2
        # test negative positions
        assert layer.get_nearest_node((0,-1)) == 0
        assert layer.get_nearest_node((-1,0)) == 0
        assert layer.get_nearest_node((-1,-1)) == 0
        assert layer.get_nearest_node((-1,5)) == 2
        # test edge cases
        assert layer.get_nearest_node((0,2.01)) == 2
        assert layer.get_nearest_node((0,1.99)) == 0
        assert layer.get_nearest_node((2.1,2.1)) == 3
        assert layer.get_nearest_node((2.1,1.99)) == 1
        assert layer.get_nearest_node((1.99,1.99)) == 0
        assert layer.get_nearest_node((1.99,2.1)) == 2
        # at a tie, we expect first in list
        assert layer.get_nearest_node((2,2)) == 0

    def test_get_nearest_edge_with_orientation(self):
        # with opposite edges
        n = [(0,0), (3,0)]
        e = [(0,1), (1,0)]
        layer = build_test_layer(n, e)
        assert layer.get_nearest_edge_with_orientation((0,0), 0) == e[0]
        assert layer.get_nearest_edge_with_orientation((0,0), math.pi) == e[1]
        assert layer.get_nearest_edge_with_orientation((0,0), -math.pi) == e[1]
        assert layer.get_nearest_edge_with_orientation((0,0), math.radians(29)) == e[0]
        assert layer.get_nearest_edge_with_orientation((0,0), math.radians(-29)) == e[0]
        assert layer.get_nearest_edge_with_orientation((0,0), math.radians(180-29)) == e[1]
        assert layer.get_nearest_edge_with_orientation((0,0), math.radians(-180+29)) == e[1]
        # with perpendicular edges
        n = [(0,0), (3,0), (3,3)]
        e = [(0,1), (1,2)]
        layer = build_test_layer(n, e)
        assert layer.get_nearest_edge_with_orientation((0,3), 0) == e[0]
        assert layer.get_nearest_edge_with_orientation((3,3), 0) == e[0]
        assert layer.get_nearest_edge_with_orientation((3,3), math.radians(30)) == e[0]
        assert layer.get_nearest_edge_with_orientation((3,3), math.radians(45)) == None
        assert layer.get_nearest_edge_with_orientation((3,3), math.radians(60.01)) == e[1]
        assert layer.get_nearest_edge_with_orientation((0,0), math.radians(90)) == e[1]
        assert layer.get_nearest_edge_with_orientation((0,0), math.pi) == None
        assert layer.get_nearest_edge_with_orientation((0,0), -math.pi) == None

    def test_get_nearest_connected_edge(self):
        pass

    def test_pick_node_given_orientation(self):
        pass
    
    def test_is_position_within_edge_bounds(self):
        n = [(0,0), (3,0)]
        e = [(0,1)]
        layer = build_test_layer(n, e)
        assert layer.is_position_within_edge_bounds((0,5), e[0]) == True
        assert layer.is_position_within_edge_bounds((-1,5), e[0]) == False
        assert layer.is_position_within_edge_bounds((3.1,5), e[0]) == False
        assert layer.is_position_within_edge_bounds((3,5), e[0]) == True
        assert layer.is_position_within_edge_bounds((1,-5), e[0]) == True
        assert layer.is_position_within_edge_bounds((1,0), e[0]) == True
        assert layer.is_position_within_edge_bounds((1,0.1), e[0]) == True
        assert layer.is_position_within_edge_bounds((0,0), e[0]) == True
        assert layer.is_position_within_edge_bounds((3,0), e[0]) == True
        assert layer.is_position_within_edge_bounds((3.001,0), e[0]) == False
        e = [(1,0)]
        layer = build_test_layer(n, e)
        assert layer.is_position_within_edge_bounds((0,5), e[0]) == True
        assert layer.is_position_within_edge_bounds((-1,5), e[0]) == False
        assert layer.is_position_within_edge_bounds((3.1,5), e[0]) == False
        assert layer.is_position_within_edge_bounds((3,5), e[0]) == True
        assert layer.is_position_within_edge_bounds((1,-5), e[0]) == True
        assert layer.is_position_within_edge_bounds((1,0), e[0]) == True
        assert layer.is_position_within_edge_bounds((1,0.1), e[0]) == True
        n = [(0,0), (0,3)]
        e = [(0,1)]
        layer = build_test_layer(n, e)
        assert layer.is_position_within_edge_bounds((5,0), e[0]) == True
        assert layer.is_position_within_edge_bounds((5,1), e[0]) == True
        assert layer.is_position_within_edge_bounds((5,3), e[0]) == True
        assert layer.is_position_within_edge_bounds((5,4), e[0]) == False
        assert layer.is_position_within_edge_bounds((-5,0), e[0]) == True
        assert layer.is_position_within_edge_bounds((-5,1), e[0]) == True
        assert layer.is_position_within_edge_bounds((-5,-0.1), e[0]) == False
        assert layer.is_position_within_edge_bounds((0,0), e[0]) == True
        assert layer.is_position_within_edge_bounds((0,3), e[0]) == True
        n = [(0,0), (3,3)]
        e = [(0,1)]
        layer = build_test_layer(n, e)
        assert layer.is_position_within_edge_bounds((0,3), e[0]) == True
        assert layer.is_position_within_edge_bounds((3,0), e[0]) == True
        assert layer.is_position_within_edge_bounds((3,3), e[0]) == True
        assert layer.is_position_within_edge_bounds((0,0), e[0]) == True
        assert layer.is_position_within_edge_bounds((1,1), e[0]) == True
        assert layer.is_position_within_edge_bounds((-1,-1), e[0]) == False
        assert layer.is_position_within_edge_bounds((4,4), e[0]) == False




    def test_distance_to_edge(self):
        n = [(0,0), (3,0)]
        e = [(0,1)]
        layer = build_test_layer(n, e)
        assert layer.distance_to_edge((0,0), e[0]) == 0
        assert layer.distance_to_edge((2,1), e[0]) == -1
        assert layer.distance_to_edge((5,2), e[0]) == -2
        assert layer.distance_to_edge((5,-2), e[0]) == 2
        assert layer.distance_to_edge((-5,-2), e[0]) == 2
        e = [(1,0)]
        layer = build_test_layer(n, e)
        assert layer.distance_to_edge((0,0), e[0]) == 0
        assert layer.distance_to_edge((2,1), e[0]) == 1
        assert layer.distance_to_edge((5,2), e[0]) == 2
        assert layer.distance_to_edge((5,-2), e[0]) == -2
        assert layer.distance_to_edge((-5,-2), e[0]) == -2
        n = [(0,0), (0,3)]
        e = [(0,1)]
        layer = build_test_layer(n, e)
        assert layer.distance_to_edge((0,0), e[0]) == 0
        assert layer.distance_to_edge((1,2), e[0]) == 1
        assert layer.distance_to_edge((2,5), e[0]) == 2
        assert layer.distance_to_edge((-2,5), e[0]) == -2
        assert layer.distance_to_edge((-2,-5), e[0]) == -2
        e = [(1,0)]
        layer = build_test_layer(n, e)
        assert layer.distance_to_edge((0,0), e[0]) == 0
        assert layer.distance_to_edge((1,2), e[0]) == -1
        assert layer.distance_to_edge((2,5), e[0]) == -2
        assert layer.distance_to_edge((-2,5), e[0]) == 2
        assert layer.distance_to_edge((-2,-5), e[0]) == 2

    def test_distance_to_edge_at_angle(self):
        n = [(0,0), (3,3)]
        e = [(0,1)]
        layer = build_test_layer(n, e)
        assert_close(layer.distance_to_edge((0,3), e[0]), -length((3,3))/2)
        assert_close(layer.distance_to_edge((3,0), e[0]), length((3,3))/2)
        e = [(1,0)]
        layer = build_test_layer(n, e)
        assert_close(layer.distance_to_edge((0,3), e[0]), length((3,3))/2)
        assert_close(layer.distance_to_edge((3,0), e[0]), -length((3,3))/2)
        n = [(0,3), (3,0)]
        e = [(0,1)]
        layer = build_test_layer(n, e)
        assert_close(layer.distance_to_edge((0,0), e[0]), length((3,3))/2)
        assert_close(layer.distance_to_edge((3,3), e[0]), -length((3,3))/2)
        e = [(1,0)]
        layer = build_test_layer(n, e)  
        assert_close(layer.distance_to_edge((0,0), e[0]), -length((3,3))/2)
        assert_close(layer.distance_to_edge((3,3), e[0]), length((3,3))/2)



if __name__ == '__main__':
    unittest.main()