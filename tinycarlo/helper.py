import os

def getenv(key: str) -> bool:
    if os.environ.get(key) is not None:
        v = os.environ.get(key)
        if v.lower() == '1':
            return True
    return False