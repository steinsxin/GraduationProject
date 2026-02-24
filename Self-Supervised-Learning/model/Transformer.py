import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """位置编码模块"""
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class Transformer_Model(nn.Module):
    """
    用于ECG信号分类的Transformer模型
    输入: (Batch, 1, Length) - 与CNN/LSTM保持一致
    输出: (Batch, 1) - 二分类概率
    """
    def __init__(self, input_size=1, d_model=64, nhead=4, num_layers=2, 
                 dim_feedforward=128, num_classes=1, dropout=0.2, max_len=5000):
        super(Transformer_Model, self).__init__()
        
        self.d_model = d_model
        
        # 输入嵌入: 将1维信号映射到d_model维
        self.input_embedding = nn.Linear(input_size, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 分类头
        self.fc = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: (Batch, Channel=1, Length)
        # 转换为 (Batch, Length, 1)
        if x.dim() == 3 and x.shape[1] == 1:
            x = x.permute(0, 2, 1)
        
        # 嵌入到d_model维度: (Batch, Length, d_model)
        x = self.input_embedding(x)
        
        # 添加位置编码
        x = self.pos_encoder(x)
        
        # Transformer编码: (Batch, Length, d_model)
        x = self.transformer_encoder(x)
        
        # 全局平均池化: (Batch, d_model)
        x = torch.mean(x, dim=1)
        
        # 分类
        x = self.fc(x)
        return x


if __name__ == "__main__":
    # Test
    x = torch.randn(32, 1, 400)  # (Batch, 1, 400)
    model = Transformer_Model()
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
