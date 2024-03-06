from tinygrad.tensor import Tensor
import tinygrad.nn as nn
from typing import Union, List, Tuple

class Encoder:
    FEATURE_VEC_SIZE = 128

    def __init__(self, image_dim: Tuple[int, int, int]):
        """
        image_dim: (channels, height, width) of the input image
        """
        self.image_dim = image_dim

        self.conv1 = nn.Conv2d(image_dim[0], 16, 5, 2)
        self.bn1 = nn.InstanceNorm(16)
        self.conv2 = nn.Conv2d(16, 32, 5, 2)
        self.bn2 = nn.InstanceNorm(32)
        self.conv3 = nn.Conv2d(32, 48, 3, 2)
        self.bn3 = nn.InstanceNorm(48)
        self.conv4 = nn.Conv2d(48, 64, 3, 2)
        self.fc1 = nn.Linear(self.__calculate_conv_out_size(), 256)
        self.fc2 = nn.Linear(256, self.FEATURE_VEC_SIZE)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1(x).relu()
        out = self.bn1(out)
        out = self.conv2(out).relu()
        out = self.bn2(out)
        out = self.conv3(out).relu()
        out = self.bn3(out)
        out = self.conv4(out).relu().flatten(start_dim=1)
        out = self.fc1(out).relu()
        return self.fc2(out).relu()
    
    def __calculate_conv_out_size(self) -> int:
        x = Tensor.zeros(*self.image_dim).unsqueeze(0)
        out = x.sequential([self.conv1, self.conv2, self.conv3, self.conv4]).flatten(start_dim=1)
        return out.shape[1]
    
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)
    
class SteeringBlock:
    def __init__(self, in_features: int, maneuver_dim: int, action_dim: int):
        self.fcm = nn.Linear(maneuver_dim, 64)
        self.fc1 = nn.Linear(64+in_features, 128)
        self.fc2 = nn.Linear(128, action_dim)

    def forward(self, f: Tensor, m: Tensor) -> Tensor:
        out = f.cat(self.fcm(m).relu(), dim=1)
        out = self.fc1(out).relu()
        return self.fc2(out).tanh()
    
    def __call__(self, f: Tensor, m: Tensor) -> Tensor:
        return self.forward(f, m)
    
class TinycarCombo:
    def __init__(self, image_dim: Tuple[int, int, int], maneuver_dim: int, action_dim: int):
        """
        image_dim: (channels, height, width) of the input image
        maneuver_dim: the number of maneuvers
        """
        self.encoder = Encoder(image_dim)
        self.steering = SteeringBlock(Encoder.FEATURE_VEC_SIZE, maneuver_dim, action_dim)

    def forward(self, x: Tensor, m: Tensor) -> Tensor:
        out = self.encoder(x)
        return self.steering(out, m)
    
    def __call__(self, image: Tensor, maneuver: Tensor) -> Tensor:
        return self.forward(image, maneuver)

class TinycarEncoder:
    """
    Only a wrapper to copy weights from combo to encoder
    """
    def __init__(self, image_dim: Tuple[int, int, int]):
        self.encoder = Encoder(image_dim)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.encoder(x)
    
    def __call__(self, image: Tensor) -> Tensor:
        return self.forward(image)

class TinycarActor:
    def __init__(self, maneuver_dim: int, action_dim: int):
        self.steering = SteeringBlock(Encoder.FEATURE_VEC_SIZE, maneuver_dim, action_dim)

    def forward(self, f: Tensor, m: Tensor) -> Tensor:
        return self.steering(f, m)
    
    def __call__(self, state: Tensor, maneuver: Tensor) -> Tensor:
        return self.forward(state, maneuver)

class TinycarCritic:
    def __init__(self, maneuver_dim: int, action_dim: int):
        self.fca = nn.Linear(action_dim, 50)
        self.fcm = nn.Linear(maneuver_dim, 50)
        self.fc1 = nn.Linear(50 + 50 + Encoder.FEATURE_VEC_SIZE, 50)
        self.fc2 = nn.Linear(50, 1)
    
    def forward(self, f: Tensor, m: Tensor, a: Tensor):
        m = self.fcm(m).relu()
        a = self.fca(a).relu()
        out = f.cat(m, dim=1).cat(a, dim=1)
        out = self.fc1(out).relu()
        return self.fc2(out)
    
    def __call__(self, state: Tensor, maneuver: Tensor, action: Tensor) -> Tensor:
        return self.forward(state, maneuver, action)
    
if __name__ == "__main__":
    image_dim = (3, 80, 200)
    maneuver_dim = 3
    action_dim = 1
    image_tensor_batch = Tensor.randn(*image_dim).unsqueeze(0)
    maneuver_tensor_batch = Tensor.randn(3).unsqueeze(0)
    feature_vec_tensor_batch = Tensor.randn(Encoder.FEATURE_VEC_SIZE).unsqueeze(0)
    action_tensor_batch = Tensor.randn(1).unsqueeze(0)

    print("Testing Encoder")
    encoder = Encoder(image_dim)
    assert encoder(image_tensor_batch).shape == feature_vec_tensor_batch.shape
    print("Encoder test passed")
    print("Testing SteeringBlock")
    steering = SteeringBlock(Encoder.FEATURE_VEC_SIZE, maneuver_dim, action_dim)
    assert steering(feature_vec_tensor_batch, maneuver_tensor_batch).shape == action_tensor_batch.shape
    print("SteeringBlock test passed")
    print("Testing TinycarCombo")
    combo = TinycarCombo(image_dim, maneuver_dim, action_dim)
    assert combo(image_tensor_batch, maneuver_tensor_batch).shape == action_tensor_batch.shape
    test_image = Tensor.randn((32, 3, 80, 200))
    test_maneuver = Tensor.randn((32, 3))
    assert combo(test_image, test_maneuver).shape == (32, 1)
    print("TinycarCombo test passed")
    print("Testing TinycarActor")
    actor = TinycarActor(maneuver_dim, action_dim)
    assert actor(feature_vec_tensor_batch, maneuver_tensor_batch).shape == action_tensor_batch.shape
    print("TinycarActor test passed")
    print("Testing TinycarCritic")
    critic = TinycarCritic(maneuver_dim, action_dim)
    assert critic(feature_vec_tensor_batch, maneuver_tensor_batch, action_tensor_batch).shape == (1,1)
    print("TinycarCritic test passed")

    print("Test weight transfer from combo to actor")
    from tinygrad.nn.state import get_state_dict, load_state_dict
    actor_state_dict = get_state_dict(actor)
    combo_state_dict = get_state_dict(combo)
    
    print("Actor state dict example before transfer")
    print(actor_state_dict["steering.fcm.weight"].numpy()[0,0])
    print("Combo state dict example before transfer")
    print(combo_state_dict["steering.fcm.weight"].numpy()[0,0])

    load_state_dict(actor, combo_state_dict)
    print("Actor state dict example after transfer")
    print(actor_state_dict["steering.fcm.weight"].numpy()[0,0])




