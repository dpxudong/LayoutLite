import torch
import torch.nn as nn

class VisionTokenScoreMLP(nn.Module):
    def __init__(self, hidden_dim=1024, mid_dim=512):
        super().__init__()
        
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, mid_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(mid_dim, 1)   
        
    def forward(self, x):
        """
        x: (B, N, 1024)
        return: (B, N, 1)
        """
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class MultiLayerVisionTokenScoreMLP(nn.Module):
    def __init__(self, hidden_dim=1024, mid_dim=512):
        super().__init__()
        
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, mid_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(mid_dim, 1)
        self.conv = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=4,
        )

        
    def forward(self, x: torch.Tensor):
        """
        x: (B, N, 1024)
        return: (B, N, 1)
        """
        # [L-1, N, D]
        # x = x[1:] - x[:-1]
        x = x.permute(1, 2, 0)
        x = self.conv(x)
        x = x.mean(dim=-1)
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

model = MultiLayerVisionTokenScoreMLP(hidden_dim=2560, mid_dim=1280)
nn.init.normal_(model.fc2.weight, mean=0.0, std=1e-3)
nn.init.constant_(model.fc2.bias, 0)

# 保存（只保存参数）
torch.save(model.state_dict(), "new_multi_layer_vision_token_score_mlp_2560_50.pt")