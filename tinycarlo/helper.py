import os
import math

def getenv(key: str) -> bool:
    if os.environ.get(key) is not None:
        v = os.environ.get(key)
        if v.lower() == '1':
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