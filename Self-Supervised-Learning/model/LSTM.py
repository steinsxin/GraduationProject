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
        
        # FC Layer
        # Bidirectional -> hidden_size * 2
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: (Batch, Channel=1, Length)
        # LSTM needs: (Batch, Length, Input_Size=1)
        if x.dim() == 3 and x.shape[1] == 1:
            x = x.permute(0, 2, 1)

        # out: (Batch, Length, Hidden*2)
        # h_n, c_n: (Num_Layers*2, Batch, Hidden)
        out, (h_n, c_n) = self.lstm(x)
        
        # Use the output of the last time step? 
        # Or Global Average Pooling? Or the hidden state?
        # Standard approach for classification: Last time step output (forward) + First time step output (backward)
        # Or just Global Max/Avg Pooling over time.
        
        # Let's use the last hidden state from the last layer for both directions
        # h_n shape: (num_layers * num_directions, batch, hidden_size)
        # ordered: [layer_0_fwd, layer_0_bwd, layer_1_fwd, layer_1_bwd, ...]
        
        # Construct feature vector from last layer states
        # forward_hidden = h_n[-2, :, :]
        # backward_hidden = h_n[-1, :, :]
        # feat = torch.cat((forward_hidden, backward_hidden), dim=1)
        
        # Easier: Global Average Pooling over voltage time series features
        # out shape: (batch, seq_len, hidden*2)
        out = torch.mean(out, dim=1)
        
        x = self.fc(out)
        return x

if __name__ == "__main__":
    # Test
    x = torch.randn(32, 1, 400) # (Batch, 1, 400)
    model = LSTM_Model()
    y = model(x)
    print(y.shape)
