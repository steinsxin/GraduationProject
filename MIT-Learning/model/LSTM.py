import torch
import torch.nn as nn

class LSTM_Model(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, num_classes=1, dropout=0.2):
        super(LSTM_Model, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True, 
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Last hidden state + temporal mean + temporal max.
        feature_dim = hidden_size * 2 * 3
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        if x.dim() == 3 and x.shape[1] == 1:
            x = x.permute(0, 2, 1)

        out, (h_n, c_n) = self.lstm(x)

        last_hidden = torch.cat((h_n[-2], h_n[-1]), dim=1)
        mean_pool = torch.mean(out, dim=1)
        max_pool = torch.max(out, dim=1).values
        features = torch.cat((last_hidden, mean_pool, max_pool), dim=1)

        x = self.fc(features)
        return x

if __name__ == "__main__":
    # Test
    x = torch.randn(32, 1, 400) # (Batch, 1, 400)
    model = LSTM_Model()
    y = model(x)
    print(y.shape)
