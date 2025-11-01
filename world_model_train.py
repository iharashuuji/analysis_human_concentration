import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2  
import os
import numpy as np
from PIL import Image
import random
from tqdm import tqdm
import time # 処理時間計測用

# --- 1. デバイス設定 ---
# Vast.ai上のGPUを自動で利用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 2. モデルのクラス定義 (VAE and DynamicsModel) ---
# ※プロジェクトの構造上は models/world_model.py に分離すべきですが、
#   ここでは単一ファイルで完結させます。

class VAE(nn.Module):
    """(V) 観測モデル"""
    def __init__(self, z_dim=32):
        super(VAE, self).__init__()
        self.z_dim = z_dim
        self.image_channels = 1 
        self.encoder_last_dim = 128 * 8 * 8

        # Encoder (CNN)
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(self.image_channels, 32, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(True),
            nn.Flatten()
        )
        self.fc_mu = nn.Linear(self.encoder_last_dim, z_dim)
        self.fc_logvar = nn.Linear(self.encoder_last_dim, z_dim)
        
        # Decoder (逆CNN)
        self.decoder_fc = nn.Linear(z_dim, self.encoder_last_dim) 
        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(True),
            nn.ConvTranspose2d(32, self.image_channels, 4, 2, 1), nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder_cnn(x); return self.fc_mu(h), self.fc_logvar(h)
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar); eps = torch.randn_like(std); return mu + eps * std
    def decode(self, z):
        h = self.decoder_fc(z).view(-1, 128, 8, 8); return self.decoder_cnn(h)
    def forward(self, x):
        mu, logvar = self.encode(x); z = self.reparameterize(mu, logvar); return self.decode(z), mu, logvar

class DynamicsModel(nn.Module):
    """(M) ダイナミクスモデル"""
    def __init__(self, z_dim=32, rnn_hidden_dim=256):
        super(DynamicsModel, self).__init__()
        self.lstm = nn.LSTM(input_size=z_dim, hidden_size=rnn_hidden_dim, num_layers=1, batch_first=True)
        self.fc_out = nn.Linear(rnn_hidden_dim, z_dim)

    def forward(self, z_sequence, hidden_state=None):
        lstm_out, next_hidden_state = self.lstm(z_sequence, hidden_state)
        predicted_z_sequence = self.fc_out(lstm_out)
        return predicted_z_sequence, next_hidden_state

# --- 3. データセットクラス ---

class DaisEeNormalDataset(Dataset):
    """Engagement=3の動画のみをロード（動画フォルダ直接指定版）"""
    def __init__(self, video_root, mode='vae', seq_len=10, transform=None):
        self.video_root = video_root
        self.mode = mode
        self.seq_len = seq_len
        self.transform = transform
        
        # フォルダ内の動画ファイルをすべてリストアップ
        self.video_files = [
            os.path.join(video_root, f) 
            for f in os.listdir(video_root) 
            if f.endswith(('.mp4', '.avi', '.mov'))
        ]
        
        if not self.video_files:
            print(f"エラー: {video_root} に動画ファイルが見つかりません。")

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_path = self.video_files[idx]
        frames = []
        try:
            cap = cv2.VideoCapture(video_path)
            while True:
                ret, frame = cap.read()
                if not ret: break
                frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert('L')
                if self.transform:
                    frame_pil = self.transform(frame_pil)
                frames.append(frame_pil)
            cap.release()
        except Exception as e:
            # エラー時はダミーデータを返すことでdataloaderのクラッシュを防ぐ
            return torch.zeros(1, 64, 64) if self.mode == 'vae' else torch.zeros(self.seq_len + 1, 1, 64, 64)

        if not frames:
             return torch.zeros(1, 64, 64) if self.mode == 'vae' else torch.zeros(self.seq_len + 1, 1, 64, 64)

        # VAE: ランダムな単一フレームを返す
        if self.mode == 'vae':
            return random.choice(frames)
        # RNN: 連続 (seq_len + 1) フレームを返す
        elif self.mode == 'rnn':
            if len(frames) <= self.seq_len:
                clip = frames + [frames[-1]] * (self.seq_len + 1 - len(frames))
            else:
                start_idx = random.randint(0, len(frames) - self.seq_len - 1)
                clip = frames[start_idx : start_idx + self.seq_len + 1]
            return torch.stack(clip)

# --- VAE 損失関数 ---
def vae_loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x.view(-1, 64*64), x.view(-1, 64*64), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# ----------------------------------------------------------------------
# ★★★ パラメータ設定 ★★★
# ----------------------------------------------------------------------
# ★ VAST.AI コンテナ内の、動画フォルダパスを設定してください
NORMAL_VIDEO_ROOT = "/app/data/" 
# モデル保存パス
WEIGHTS_DIR = "weights"
os.makedirs(WEIGHTS_DIR, exist_ok=True)

# VAEパラメータ
Z_DIM = 32
VAE_LR = 1e-3
VAE_EPOCHS = 20  # GPU環境ならこの程度は回したい
VAE_BATCH_SIZE = 64

# RNNパラメータ
RNN_LR = 1e-4
RNN_EPOCHS = 50  # VAEより多めに回す
RNN_BATCH_SIZE = 32
RNN_SEQ_LEN = 10 
RNN_HIDDEN_DIM = 256

transform_base = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# --- 4. VAE訓練の実行 ---

def train_vae():
    print("\n" + "="*50)
    print(" (1) VAE (観測モデル) 訓練開始 ")
    print("="*50)
    
    start_time = time.time()
    
    vae_dataset = DaisEeNormalDataset(video_root=NORMAL_VIDEO_ROOT, mode='vae', transform=transform_base)
    vae_dataloader = DataLoader(vae_dataset, batch_size=VAE_BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    
    vae_model = VAE(z_dim=Z_DIM).to(device)
    optimizer_vae = Adam(vae_model.parameters(), lr=VAE_LR)
    
    model_path = os.path.join(WEIGHTS_DIR, 'vae_engage3_only.pth')

    for epoch in tqdm(range(VAE_EPOCHS), desc="VAE EPOCHS"):
        total_loss = 0
        for images in vae_dataloader:
            if images.dim() == 0: continue
            images = images.to(device)
            
            optimizer_vae.zero_grad()
            recon_images, mu, logvar = vae_model(images)
            loss = vae_loss_function(recon_images, images, mu, logvar)
            
            loss.backward()
            optimizer_vae.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / (len(vae_dataloader) + 1e-6)
        print(f"| VAE Epoch {epoch+1}/{VAE_EPOCHS}, Avg Loss: {avg_loss:.4f}")

    torch.save(vae_model.state_dict(), model_path)
    print(f"\n--- VAE訓練完了。経過時間: {time.time() - start_time:.1f}s ---")
    return vae_model

# --- 5. RNN訓練の実行 ---

def train_rnn(vae_model):
    print("\n" + "="*50)
    print(" (2) RNN (ダイナミクスモデル) 訓練開始 ")
    print("="*50)

    start_time = time.time()
    
    # RNN用データローダー
    rnn_dataset = DaisEeNormalDataset(video_root=NORMAL_VIDEO_ROOT, mode='rnn', seq_len=RNN_SEQ_LEN, transform=transform_base)
    rnn_dataloader = DataLoader(rnn_dataset, batch_size=RNN_BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    
    # RNNモデル
    rnn_model = DynamicsModel(z_dim=Z_DIM, rnn_hidden_dim=RNN_HIDDEN_DIM).to(device)
    optimizer_rnn = Adam(rnn_model.parameters(), lr=RNN_LR)
    loss_fn_rnn = nn.MSELoss() 
    model_path = os.path.join(WEIGHTS_DIR, 'rnn_engage3_only.pth')

    # VAEはエンコーダとして固定
    vae_model.eval()
    
    for epoch in tqdm(range(RNN_EPOCHS), desc="RNN EPOCHS"):
        total_loss = 0
        for frame_clips in rnn_dataloader:
            if frame_clips.dim() < 5: continue
            frame_clips = frame_clips.to(device)
            
            # --- VAEで z シーケンスに変換 ---
            with torch.no_grad():
                B, T, C, H, W = frame_clips.shape
                frame_clips_flat = frame_clips.view(B * T, C, H, W)
                mu, _ = vae_model.encode(frame_clips_flat)
                z_sequence = mu.view(B, T, Z_DIM)

            inputs_z = z_sequence[:, :-1, :] # t=0 から t=Seq_Len-1
            targets_z = z_sequence[:, 1:, :]  # t=1 から t=Seq_Len

            optimizer_rnn.zero_grad()
            predicted_z, _ = rnn_model(inputs_z)
            
            loss = loss_fn_rnn(predicted_z, targets_z)
            
            loss.backward()
            optimizer_rnn.step()
            total_loss += loss.item()

        avg_loss = total_loss / (len(rnn_dataloader) + 1e-6)
        print(f"| RNN Epoch {epoch+1}/{RNN_EPOCHS}, Avg Loss: {avg_loss:.6f}")

    torch.save(rnn_model.state_dict(), model_path)
    print(f"\n--- RNN訓練完了。経過時間: {time.time() - start_time:.1f}s ---")
    return rnn_model

# --- 6. メイン実行 ---

if __name__ == "__main__":
    # 訓練開始
    vae_model_trained = train_vae()
    
    # VAEが成功した場合のみRNNを訓練
    if vae_model_trained:
        train_rnn(vae_model_trained)
        print("\n★★★ 全てのモデル訓練が完了しました。推論アプリに接続してください。★★★")
    else:
        print("\n!!! VAE訓練が失敗したため、RNN訓練はスキップされました。データパスを確認してください。!!!")
