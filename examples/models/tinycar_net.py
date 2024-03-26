import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict
import os

model_urls: Dict[Tuple[int,int,int], str] = {   
    (5,64,160): "http://riege.com.de/tinycarlo/tinycar_combo_5_64_160.pt",
}

DEFAULT_M_DIM = 4
DEFAULT_A_DIM = 1

class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 2, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x: torch.tensor) -> torch.tensor:
        return F.relu(self.bn1(self.conv1(x)))
    
class TinycarEncoder(nn.Module):
    FEATURE_VEC_SIZE = 256

    def __init__(self, image_dim: Tuple[int, int, int]):
        super(TinycarEncoder, self).__init__()
        self.image_dim = image_dim
        self.filters = [24, 36, 48, 64, 64]
        self.convs = nn.ModuleList([ConvBlock(image_dim[0] if i == 0 else self.filters[i-1], fts) for i, fts in enumerate(self.filters)])
        self.fc1 = nn.Linear(self.__calculate_conv_out_size(), self.FEATURE_VEC_SIZE)

    def forward(self, x: torch.tensor) -> torch.tensor:
        out = x
        for conv in self.convs:
            out = conv(out)
        out = out.flatten(start_dim=1)
        return F.relu(self.fc1(out))
    
    def __calculate_conv_out_size(self) -> int:
        x = torch.zeros(*self.image_dim).unsqueeze(0)
        out = x
        for conv in self.convs:
            out = conv(out)
        out = out.flatten(start_dim=1)
        return out.shape[1]
    
class TinycarActor(nn.Module):
    def __init__(self, in_features: int = TinycarEncoder.FEATURE_VEC_SIZE, maneuver_dim: int = DEFAULT_M_DIM, action_dim: int = DEFAULT_A_DIM):
        super(TinycarActor, self).__init__()
        self.fcm = nn.Linear(maneuver_dim, in_features)
        self.fc1 = nn.Linear(in_features, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.fc4 = nn.Linear(10, action_dim)

    def forward(self, f: torch.tensor, m: torch.tensor) -> torch.tensor:
        attention_weights = F.sigmoid(self.fcm(m))
        out = f * attention_weights
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        return F.tanh(self.fc4(out))
    
class TinycarActorLSTM(nn.Module):
    def __init__(self, in_features: int = TinycarEncoder.FEATURE_VEC_SIZE, maneuver_dim: int = DEFAULT_M_DIM, action_dim: int = DEFAULT_A_DIM):
        super(TinycarActorLSTM, self).__init__()
        self.lstm = nn.LSTM(in_features, in_features, 1, batch_first=True)
        self.fcm = nn.Linear(maneuver_dim, in_features)
        self.fc1 = nn.Linear(in_features, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.fc4 = nn.Linear(10, action_dim)

    def forward(self, f: torch.tensor, m: torch.tensor) -> torch.tensor:
        attention_weights = F.sigmoid(self.fcm(m))
        last_output = self.lstm(f)[0][:, -1, :]
        out = last_output * attention_weights
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        return F.tanh(self.fc4(out)) 
    
class TinycarCombo(nn.Module):
    def __init__(self, image_dim: Tuple[int, int, int], maneuver_dim: int = DEFAULT_M_DIM, action_dim: int = DEFAULT_A_DIM):
        super(TinycarCombo, self).__init__()
        self.image_dim, self.m_dim, self.a_dim = image_dim, maneuver_dim, action_dim
        self.encoder = TinycarEncoder(image_dim)
        self.actor = TinycarActor(maneuver_dim=maneuver_dim, action_dim=action_dim)

    def forward(self, x: torch.tensor, m: torch.tensor) -> torch.tensor:
        out = self.encoder(x)
        return self.actor(out, m)
    
    def load_pretrained(self) -> bool: 
        if self.image_dim in model_urls and self.m_dim == DEFAULT_M_DIM and self.a_dim == DEFAULT_A_DIM:
            model_url = model_urls[self.image_dim]
            cached_file = os.path.join("/tmp", os.path.basename(model_url))
            if not os.path.exists(cached_file):
                torch.hub.download_url_to_file(model_url, cached_file)
            self.load_state_dict(torch.load(cached_file))
            return True
        print(f"No pretrained weights found for image_dim: {self.image_dim}, maneuver_dim: {self.m_dim}, action_dim: {self.a_dim}")
        return False

class TinycarComboLSTM(nn.Module):
    def __init__(self, image_dim: Tuple[int, int, int], maneuver_dim: int = DEFAULT_M_DIM, action_dim: int = DEFAULT_A_DIM):
        super(TinycarComboLSTM, self).__init__()
        self.image_dim, self.m_dim, self.a_dim = image_dim, maneuver_dim, action_dim
        self.encoder = TinycarEncoder(image_dim)
        self.actor = TinycarActorLSTM(maneuver_dim=maneuver_dim, action_dim=action_dim)

    def forward(self, x: torch.tensor, m: torch.tensor) -> torch.tensor:
        out = self.encoder(x)
        return self.actor(out, m)

class TinycarCritic(nn.Module):
    def __init__(self, maneuver_dim: int = DEFAULT_M_DIM, action_dim: int = DEFAULT_A_DIM):
        super(TinycarCritic, self).__init__()
        self.fca = nn.Linear(action_dim, TinycarEncoder.FEATURE_VEC_SIZE)
        self.fcm = nn.Linear(maneuver_dim, TinycarEncoder.FEATURE_VEC_SIZE)
        self.fc1 = nn.Linear(TinycarEncoder.FEATURE_VEC_SIZE*3, 512)
        self.fc2 = nn.Linear(512, 1)
    
    def forward(self, f: torch.tensor, m: torch.tensor, a: torch.tensor) -> torch.tensor:
        m = F.relu(self.fcm(m))
        a = F.relu(self.fca(a))
        out = torch.cat([f, m, a], dim=1)
        out = F.relu(self.fc1(out))
        return self.fc2(out)
    
class TinycarCriticLSTM(nn.Module):
    def __init__(self, maneuver_dim: int = DEFAULT_M_DIM, action_dim: int = DEFAULT_A_DIM):
        super(TinycarCriticLSTM, self).__init__()
        self.fca = nn.Linear(action_dim, TinycarEncoder.FEATURE_VEC_SIZE)
        self.fcm = nn.Linear(maneuver_dim, TinycarEncoder.FEATURE_VEC_SIZE)
        self.lstm = nn.LSTM(TinycarEncoder.FEATURE_VEC_SIZE, TinycarEncoder.FEATURE_VEC_SIZE, 1, batch_first=True)
        self.fc1 = nn.Linear(TinycarEncoder.FEATURE_VEC_SIZE*3, 512)
        self.fc2 = nn.Linear(512, 1)
    
    def forward(self, f: torch.tensor, m: torch.tensor, a: torch.tensor) -> torch.tensor:
        m = F.relu(self.fcm(m))
        a = F.relu(self.fca(a))
        last_output = self.lstm(f)[0][:, -1, :]
        out = torch.cat([last_output, m, a], dim=1)
        out = F.relu(self.fc1(out))
        return self.fc2(out)