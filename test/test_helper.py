import unittest
from tinycarlo.helper import clip_angle
import math

class HelperTestCase(unittest.TestCase):
    def test_clip_angle(self):
        assert clip_angle(0) == 0
        assert clip_angle(math.pi) == math.pi
        assert clip_angle(-math.pi) == -math.pi
        assert clip_angle(2*math.pi) == 0
        assert clip_angle(-2*math.pi) == 0
        assert clip_angle(3*math.pi) == math.pi
        assert clip_angle(-3*math.pi) == -math.pi
        assert clip_angle(- 3/2 * math.pi) == math.pi/2
        assert clip_angle(3/2 * math.pi) == -math.pi/2

if __name__ == '__main__':
    unittest.main()