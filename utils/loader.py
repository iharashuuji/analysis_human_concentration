import streamlit as st
import torch
import os
from models.world_model import VAE, MDNRNN # DynamicsModelの代わりにMDNRNNをインポート

from utils.logger import setup_logger

logger = setup_logger()

@st.cache_resource
def load_models(vae_path, rnn_path, z_dim, rnn_hidden_dim):
    """
    訓練済みのVAEとRNNモデルをロードする。
    CPU環境 (Docker内など) を想定し、map_location="cpu" を指定。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading models to device: {device}")

    # VAE (観測モデル)
    vae_model = VAE(z_dim=z_dim).to(device)
    logger.info(f"Attempting to load VAE model from: {vae_path}")
    if not os.path.exists(vae_path):
        st.error(f"VAEモデルファイルが見つかりません: {vae_path}")
        return None, None, None
    vae_model.load_state_dict(torch.load(vae_path, map_location=device))
    vae_model.eval()

    # MDN-RNN (ダイナミクスモデル)
    # num_gaussiansは学習時と合わせる
    rnn_model = MDNRNN(z_dim=z_dim, rnn_hidden_dim=rnn_hidden_dim, num_gaussians=5).to(device)
    logger.info(f"Attempting to load RNN model from: {rnn_path}")
    if not os.path.exists(rnn_path):
        st.error(f"RNNモデルファイルが見つかりません: {rnn_path}")
        return None, None, None
    rnn_model.load_state_dict(torch.load(rnn_path, map_location=device))
    rnn_model.eval()

    return vae_model, rnn_model, device