import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------
# (V) VAE: 観測モデル (あなたのコードをそのままペースト)
# ----------------------------------------------------------------------
class VAE(nn.Module):
    """
    (V) 観測モデル
    入力: (Batch, 1, 64, 64) の顔画像
    出力: 潜在変数 z (Batch, z_dim)
    """
    def __init__(self, z_dim=32):
        super(VAE, self).__init__()
        self.z_dim = z_dim
        self.image_channels = 1 
        self.encoder_last_dim = 128 * 8 * 8

        # --- 1. Encoder (CNN) ---
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(self.image_channels, 32, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(True),
            nn.Flatten() 
        )
        
        # --- 2. 潜在変数 (z) への変換 ---
        self.fc_mu = nn.Linear(self.encoder_last_dim, z_dim)
        self.fc_logvar = nn.Linear(self.encoder_last_dim, z_dim)

        # --- 3. Decoder (逆CNN) ---
        self.decoder_fc = nn.Linear(z_dim, self.encoder_last_dim) 
        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(True),
            nn.ConvTranspose2d(32, self.image_channels, 4, 2, 1),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder_cnn(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_fc(z)
        h = h.view(-1, 128, 8, 8)
        return self.decoder_cnn(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

# ----------------------------------------------------------------------
# (M) RNN: ダイナミクスモデル (あなたのコードをそのままペースト)
# ----------------------------------------------------------------------
class DynamicsModel(nn.Module):
    """
    (M) ダイナミクスモデル
    入力: z のシーケンス (Batch, Seq_Len, z_dim)
    出力: 次の z の予測シーケンス (Batch, Seq_Len, z_dim)
    """
    def __init__(self, z_dim=32, rnn_hidden_dim=256):
        super(DynamicsModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=z_dim,
            hidden_size=rnn_hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.fc_out = nn.Linear(rnn_hidden_dim, z_dim)

    def forward(self, z_sequence, hidden_state=None):
        lstm_out, next_hidden_state = self.lstm(z_sequence, hidden_state)
        predicted_z_sequence = self.fc_out(lstm_out)
        return predicted_z_sequence, next_hidden_state