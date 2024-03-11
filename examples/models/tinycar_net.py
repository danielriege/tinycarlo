from tinygrad.tensor import Tensor
from tinygrad.helpers import fetch
import tinygrad.nn as nn
from typing import Union, Dict, Tuple

from tinygrad.dtype import dtypes

dtypes.default_float = dtypes.float16
    
model_urls: Dict[Tuple[int,int,int], str] = {   
    (3,80,200): "http://riege.com.de/tinycarlo/tinycar_combo.safetensors"
}

class TinycarEncoder:
    FEATURE_VEC_SIZE = 64

    def __init__(self, image_dim: Tuple[int, int, int]):
        """
        image_dim: (channels, height, width) of the input image
        """
        self.image_dim = image_dim

        self.conv1 = nn.Conv2d(image_dim[0], 16, 3, 2)
        self.bn1 = nn.InstanceNorm(16)
        self.conv2 = nn.Conv2d(16, 24, 3, 2)
        self.bn2 = nn.InstanceNorm(24)
        self.conv3 = nn.Conv2d(24, 32, 3, 2)
        self.bn3 = nn.InstanceNorm(32)
        self.conv4 = nn.Conv2d(32, 48, 3, 2)
        self.fc1 = nn.Linear(self.__calculate_conv_out_size(), 128)
        self.fc2 = nn.Linear(128, self.FEATURE_VEC_SIZE)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1(x).leakyrelu()
        out = self.bn1(out)
        out = self.conv2(out).leakyrelu()
        out = self.bn2(out)
        out = self.conv3(out).leakyrelu()
        out = self.bn3(out)
        out = self.conv4(out).leakyrelu().flatten(start_dim=1)
        out = self.fc1(out).leakyrelu()
        return self.fc2(out).leakyrelu()
    
    def __calculate_conv_out_size(self) -> int:
        x = Tensor.zeros(*self.image_dim).unsqueeze(0)
        out = x.sequential([self.conv1, self.conv2, self.conv3, self.conv4]).flatten(start_dim=1)
        return out.shape[1]
    
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)
    
class TinycarActor:
    def __init__(self, in_features: int = TinycarEncoder.FEATURE_VEC_SIZE, maneuver_dim: int = 4, action_dim: int = 1):
        self.fcm = nn.Linear(maneuver_dim, in_features)
        self.fc1 = nn.Linear(in_features, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, action_dim)

    def forward(self, f: Tensor, m: Tensor) -> Tensor:
        out = f + self.fcm(m).leakyrelu()
        out = self.fc1(out).leakyrelu()
        out = self.fc2(out).leakyrelu()
        return self.fc3(out).tanh()
    
    def __call__(self, f: Tensor, m: Tensor) -> Tensor:
        return self.forward(f, m)

class TinycarCritic:
    def __init__(self, maneuver_dim: int = 4, action_dim: int = 1):
        self.fca = nn.Linear(action_dim, 50)
        self.fcm = nn.Linear(maneuver_dim, 50)
        self.fc1 = nn.Linear(50 + 50 + TinycarEncoder.FEATURE_VEC_SIZE, 50)
        self.fc2 = nn.Linear(50, 1)
    
    def forward(self, f: Tensor, m: Tensor, a: Tensor):
        m = self.fcm(m).relu()
        a = self.fca(a).relu()
        out = f.cat(m, dim=1).cat(a, dim=1)
        out = self.fc1(out).relu()
        return self.fc2(out)
    
    def __call__(self, state: Tensor, maneuver: Tensor = 4, action: Tensor = 1) -> Tensor:
        return self.forward(state, maneuver, action)
    
class TinycarCombo:
    def __init__(self, image_dim: Tuple[int, int, int], maneuver_dim: int = 4, action_dim: int = 1):
        """
        image_dim: (channels, height, width) of the input image
        maneuver_dim: the number of maneuvers
        """
        self.image_dim, self.m_dim, self.a_dim = image_dim, maneuver_dim, action_dim
        self.encoder = TinycarEncoder(image_dim)
        self.actor = TinycarActor(maneuver_dim = maneuver_dim, action_dim = action_dim)

    def forward(self, x: Tensor, m: Tensor) -> Tensor:
        out = self.encoder(x)
        return self.actor(out, m)
    
    def __call__(self, image: Tensor, maneuver: Tensor) -> Tensor:
        return self.forward(image, maneuver)
    
    def load_pretrained(self) -> bool: 
        if self.image_dim in model_urls and self.m_dim == 4 and self.a_dim == 1:
            nn.state.load_state_dict(self, nn.state.safe_load(fetch(model_urls[self.image_dim])), strict=False, consume=True)
            return True
        print(f"No pretrained weights found for image_dim: {self.image_dim}, maneuver_dim: {self.m_dim}, action_dim: {self.a_dim}")
        return False




