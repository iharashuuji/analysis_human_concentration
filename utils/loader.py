import streamlit as st
import torch
import os
from models.world_model import VAE, DynamicsModel

# --- パスとパラメータ ---
VAE_MODEL_PATH = "weights/vae_engage3_only.pth"
RNN_MODEL_PATH = "weights/rnn_engage3_only.pth"
Z_DIM = 32
RNN_HIDDEN_DIM = 256

@st.cache_resource
def load_models():
    """
    訓練済みのVAEとRNNモデルをロードする。
    CPU環境 (Docker内など) を想定し、map_location="cpu" を指定。
    """
    # ローカルPCのGPUで動かす場合は "cuda" も可
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading models to {device}...")

    # VAE (観測モデル)
    vae_model = VAE(z_dim=Z_DIM).to(device)
    if not os.path.exists(VAE_MODEL_PATH):
        st.error(f"VAEモデルファイルが見つかりません: {VAE_MODEL_PATH}")
        return None, None
    vae_model.load_state_dict(torch.load(VAE_MODEL_PATH, map_location=device))
    vae_model.eval()

    # RNN (ダイナミクスモデル)
    rnn_model = DynamicsModel(z_dim=Z_DIM, rnn_hidden_dim=RNN_HIDDEN_DIM).to(device)
    if not os.path.exists(RNN_MODEL_PATH):
        st.error(f"RNNモデルファイルが見つかりません: {RNN_MODEL_PATH}")
        return None, None
    rnn_model.load_state_dict(torch.load(RNN_MODEL_PATH, map_location=device))
    rnn_model.eval()

    print("Models loaded successfully.")
    return vae_model, rnn_model, device