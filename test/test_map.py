import unittest

class MapTestCase(unittest.TestCase):
    def setUp(self):
        # Create an instance of the Map class and initialize nodes
        self.map = Map()
        self.map.nodes = {
            0: (0, 0),
            1: (1, 1),
            2: (2, 2)
        }

    def test_get_edge_coordinates(self):
        # Test case 1: Edge with nodes 0 and 1
        edge = (0, 1)
        expected_coordinates = ((0, 0), (1, 1))
        self.assertEqual(self.map.get_edge_coordinates(edge), expected_coordinates)

        # Test case 2: Edge with nodes 1 and 2
        edge = (1, 2)
        expected_coordinates = ((1, 1), (2, 2))
        self.assertEqual(self.map.get_edge_coordinates(edge), expected_coordinates)

if __name__ == '__main__':
    unittest.main()