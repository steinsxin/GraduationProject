import torch
import torch.nn as nn
import math


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()

        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            raise ValueError(f"Sequence length {seq_len} exceeds max positional length {self.pe.size(1)}")
        return x + self.pe[:, :seq_len, :]


class AttentionPooling(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x):
        weights = torch.softmax(self.score(x), dim=1)
        return torch.sum(x * weights, dim=1)


class Transformer_Model(nn.Module):
    def __init__(self, input_size=1, d_model=128, nhead=4, num_layers=2, num_classes=1, dropout=0.3):
        super(Transformer_Model, self).__init__()

        self.cnn_extractor = nn.Sequential(
            nn.Conv1d(input_size, 32, kernel_size=15, stride=1, padding=7),
            nn.BatchNorm1d(32),
            nn.GELU(),

            nn.Conv1d(32, 64, kernel_size=9, stride=1, padding=4),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(64, 128, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(128, d_model, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(d_model),
            nn.GELU()
        )

        self.pre_lstm_norm = nn.LayerNorm(d_model)
        self.bilstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.feature_fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.pos_encoder = SinusoidalPositionalEncoding(d_model=d_model, max_len=1024)
        self.input_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.sequence_norm = nn.LayerNorm(d_model)
        self.attention_pool = AttentionPooling(d_model)
        self.fc = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)

        x = self.cnn_extractor(x)
        x = x.permute(0, 2, 1)
        x = self.pre_lstm_norm(x)

        lstm_out, _ = self.bilstm(x)
        x = self.feature_fusion(torch.cat([x, lstm_out], dim=-1))

        x = self.pos_encoder(x)
        x = self.input_dropout(x)
        x = self.transformer_encoder(x)
        x = self.sequence_norm(x)
        x = self.attention_pool(x)
        return self.fc(x)

if __name__ == "__main__":
    x = torch.randn(32, 1, 400)  # (Batch, 1, 400)
    model = Transformer_Model()
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
