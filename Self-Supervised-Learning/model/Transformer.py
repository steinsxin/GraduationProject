import torch
import torch.nn as nn
import math

class Transformer_Model(nn.Module):
    """
    优化的 CNN + Transformer (1D-ViT / Conformer-Lite) 架构
    加入 [CLS] Token 聚合全局特征，调整 CNN 降采样力度，使用可学习位置编码以比肩 LSTM。
    """
    def __init__(self, input_size=1, d_model=64, nhead=4, num_layers=2, num_classes=1, dropout=0.3):
        super(Transformer_Model, self).__init__()
        
        # 1. 局部特征提取层 (CNN) - 减轻降采样力度，保留更多时间步细节，避免特征被粗暴压缩
        self.cnn_extractor = nn.Sequential(
            nn.Conv1d(input_size, 16, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 长度 L / 2
            
            nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 长度 L / 4

            nn.Conv1d(32, d_model, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)   # 长度 L / 8 (400个点输入时，这里剩下50步)
        )
        
        # 2. [CLS] Token & 可学习位置编码 (Learnable Positional Encoding)
        # 仿照 ViT，专门引入一个全局表征 Token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # 充裕的位置编码矩阵最大长度（适配较长的输入心电信号被池化后的长度，比如输入4000长度，池化后约500，这里给够余量）
        self.pos_embedding = nn.Parameter(torch.randn(1, 1500, d_model))
        self.pos_dropout = nn.Dropout(dropout)
        
        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. 分类头 (Classification Head) - 只取 [CLS] 的输出
        self.fc = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape expected: (Batch, Channel=1, Length)
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        # 1. CNN 提取局部微小节律特征并降采
        x_cnn = self.cnn_extractor(x)  # (Batch, d_model, L')
        x_seq = x_cnn.permute(0, 2, 1) # (Batch, L', d_model)
        
        batch_size, seq_len, _ = x_seq.shape
        
        # 2. 拼接 [CLS] Token 在序列头部
        cls_tokens = self.cls_token.expand(batch_size, -1, -1) # (Batch, 1, d_model)
        x_seq = torch.cat((cls_tokens, x_seq), dim=1)          # (Batch, 1 + L', d_model)
        
        # 3. 添加绝对位置编码
        x_seq = x_seq + self.pos_embedding[:, :(seq_len + 1), :]
        x_seq = self.pos_dropout(x_seq)
        
        # 4. Transformer 全局自注意力计算长程关联
        x_trans = self.transformer_encoder(x_seq) # (Batch, 1 + L', d_model)
        
        # 5. 提取聚合了全局信息的 [CLS] 进行最终分类 (不使用全局均值池化，避免重要特征被稀释)
        cls_out = x_trans[:, 0, :] # 提取头部 Token
        
        # 6. 分类出概率
        out = self.fc(cls_out)
        return out

if __name__ == "__main__":
    # Test
    x = torch.randn(32, 1, 400)  # (Batch, 1, 400)
    model = Transformer_Model()
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
