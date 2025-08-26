import torch.nn as nn

class CNN1D(nn.Module):
    def __init__(self, T_obs):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch, T_obs)
        x = x.unsqueeze(1)  # add channel dimension -> (batch, 1, T_obs)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)  # (batch, 32)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x
    
    def embed(self, x):
        # return embeddings before sigmoid, intended for UMAP visualisation
        x = x.unsqueeze(1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)  # (batch, 32)
        return x
    
class FFN(nn.Module):
    def __init__(self, T_obs, hidden_dim=64, embed_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(T_obs, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.fc_out = nn.Linear(embed_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch, T_obs)
        x = self.relu(self.fc1(x))        # (batch, hidden_dim)
        x = self.relu(self.fc2(x))        # (batch, embed_dim)
        x = self.fc_out(x)                # (batch, 1)
        x = self.sigmoid(x)
        return x
    
    def embed(self, x):
        # return embeddings before sigmoid, intended for UMAP visualisation
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))        # (batch, embed_dim)
        return x
    