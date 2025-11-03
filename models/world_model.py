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
    
    
# ----------------------------------------------------------------------
# ★★★ 損失関数 ★★★
# ----------------------------------------------------------------------

# 1. VAEの損失関数
def vae_loss_function(recon_x, x, mu, logvar, beta=1.0):
    """VAEの損失: 再構成誤差 (BCE) + KLダイバージェンス"""
    # BCE (Binary Cross Entropy): 再構成誤差 - ピクセルレベルで比較
    recon_loss = F.binary_cross_entropy(recon_x.view(recon_x.size(0), -1), x.view(x.size(0), -1), reduction='sum')
    
    # KLD (KL Divergence): 潜在空間の正則化 - 標準正規分布からの逸脱を罰する
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + beta * kld_loss
    
# (M) MDNRNN: ダイナミクスモデル (Dynamics Model)
class MDNRNN(nn.Module):
    def __init__(self, z_dim, rnn_hidden_dim, num_gaussians):
        super(MDNRNN, self).__init__()
        self.z_dim = z_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.num_gaussians = num_gaussians

        # 1. LSTMコア - 潜在変数zの時系列パターンを学習
        self.lstm = nn.LSTM(
            input_size=z_dim,
            hidden_size=rnn_hidden_dim,
            num_layers=1, 
            batch_first=True
        )
        
        # 2. MDN出力層 - 混合ガウス分布のパラメータ (μ, σ, π) を予測
        self.mdn_output_layer = nn.Linear(
            rnn_hidden_dim,
            num_gaussians * (2 * z_dim + 1)
        )

    def forward(self, z_sequence, hidden_state=None):
        lstm_out, next_hidden_state = self.lstm(z_sequence, hidden_state)
        mdn_params = self.mdn_output_layer(lstm_out)
        
        B, T, _ = mdn_params.shape
        
        # μ (平均) を抽出
        mus = mdn_params[..., :self.num_gaussians * self.z_dim].view(B, T, self.num_gaussians, self.z_dim)
        
        # σ (標準偏差) を抽出 (exp()により必ず正の値にする)
        sigmas = torch.exp(mdn_params[..., self.num_gaussians * self.z_dim : 2 * self.num_gaussians * self.z_dim].view(B, T, self.num_gaussians, self.z_dim))
        
        # log(π) (混合係数の対数) を抽出 (log_softmaxにより合計が1になることを保証)
        log_pi = F.log_softmax(mdn_params[..., 2 * self.num_gaussians * self.z_dim:].view(B, T, self.num_gaussians), dim=-1)
        
        return mus, sigmas, log_pi, next_hidden_state

# 2. MDN-RNNの損失関数 (NLL: 負の対数尤度)
def gmm_nll_loss(targets, mus, sigmas, log_pi):
    """GMM (混合ガウス分布) の負の対数尤度を計算"""    
    targets = targets.unsqueeze(2) # (B, T, 1, z_dim) に拡張
    z_dim = targets.shape[-1] # ターゲットからz_dimを取得

    # 各ガウス分布の対数確率密度を計算
    log_sigma = torch.log(sigmas)
    exponent = -0.5 * ((targets - mus) / sigmas) ** 2
    log_prob_const = -0.5 * z_dim * np.log(2 * np.pi)
    
    # z_dim全体で合計し、各ガウス分布の対数尤度を求める (B, T, num_gaussians)
    log_probs = log_prob_const - log_sigma.sum(dim=-1) + exponent.sum(dim=-1)

    # logsumexpトリックで混合分布の対数尤度 log(Σ π_k * N_k) を安定して計算
    log_likelihood = torch.logsumexp(log_pi + log_probs, dim=-1)
    
    # 負の対数尤度を最小化する
    return -torch.mean(log_likelihood)