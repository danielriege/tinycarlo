import os
import math

def getenv(key: str) -> bool:
    if os.environ.get(key) is not None:
        v = os.environ.get(key)
        if v is not None and v.lower() == '1':
            return True
    return False

def clip_angle(angle: float) -> float:
    """
    Clip angle to [-pi, pi] range
    """
    while angle > math.pi:
        angle -= 2*math.pi
    while angle < -math.pi:
        angle += 2*math.pi
    return angle

def angle(x: float, y: float) -> float:
    """
    Calculate the angle of a vector (x, y) in radians
    """
    return math.atan2(y, x)