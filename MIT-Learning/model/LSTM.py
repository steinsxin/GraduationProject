import torch
import torch.nn as nn


class ConvFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(16),
            nn.GELU(),
            nn.Conv1d(16, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        return self.layers(x)


class LSTM_Model(nn.Module):
    def __init__(self, input_size=64, hidden_size=64, num_layers=2, num_classes=1, dropout=0.2):
        super(LSTM_Model, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.feature_extractor = ConvFeatureExtractor()
        self.sequence_norm = nn.LayerNorm(input_size)

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

        feature_dim = hidden_size * 2 * 4
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, 192),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(192, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if x.dim() != 3:
            raise ValueError(f"Expected 2D or 3D input, got shape {tuple(x.shape)}")

        conv_features = self.feature_extractor(x)
        x = conv_features.transpose(1, 2)
        x = self.sequence_norm(x)

        out, (h_n, c_n) = self.lstm(x)

        last_hidden = torch.cat((h_n[-2], h_n[-1]), dim=1)
        mean_pool = torch.mean(out, dim=1)
        max_pool = torch.max(out, dim=1).values
        attn_scores = self.attention(out).squeeze(-1)
        attn_weights = torch.softmax(attn_scores, dim=1)
        attn_pool = torch.sum(out * attn_weights.unsqueeze(-1), dim=1)
        features = torch.cat((last_hidden, mean_pool, max_pool, attn_pool), dim=1)

        x = self.fc(features)
        return x

if __name__ == "__main__":
    # Test
    x = torch.randn(32, 1, 720)
    model = LSTM_Model()
    y = model(x)
    print(y.shape)
