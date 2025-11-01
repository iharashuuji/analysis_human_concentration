import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image
import os
import tempfile
import logging

logger = logging.getLogger(__name__)

class RealtimeProcessor:
    """
    リアルタイムでフレームを処理し、異常スコアを計算するためのクラス。
    RNNの隠れ状態などを内部で保持する。
    """
    def __init__(self, vae_model, rnn_model, device):
        self.vae_model = vae_model
        self.rnn_model = rnn_model
        self.device = device
        
        # 前処理
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])
        
        # 損失関数
        self.loss_fn_rnn = torch.nn.MSELoss()
        
        # 状態を初期化
        self.reset()

    def reset(self):
        """推論の状態をリセットする"""
        self.hidden_state = None
        self.last_z = None

    def process_frame(self, frame):
        """
        単一のフレーム (OpenCV BGR形式) を処理し、異常スコアを返す。
        """
        with torch.no_grad():
            # 1. 前処理 (OpenCV -> PIL -> Grayscale -> Tensor)
            logger.debug("Processing a new frame.")
            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert('L')
            frame_tensor = self.transform(frame_pil).unsqueeze(0).to(self.device)

            # 2. VAEで現在の状態 z_t を抽出
            mu, _ = self.vae_model.encode(frame_tensor)
            current_z = mu.unsqueeze(1) # (1, 1, Z_DIM)

            # 3. RNNで予測誤差（異常スコア）を計算
            score = 0.0
            if self.last_z is not None:
                predicted_z, self.hidden_state = self.rnn_model(self.last_z, self.hidden_state)
                score = self.loss_fn_rnn(predicted_z, current_z).item()
            
            self.last_z = current_z # 状態を更新
            logger.debug(f"Calculated score: {score:.4f}")
            return score

def analyze_video(video_bytes, vae_model, rnn_model, device):
    """
    アップロードされた動画を分析し、異常スコア（予測誤差）のリストを返す。
    内部でRealtimeProcessorを利用してコードの重複を避ける。
    """
    anomaly_scores = []
    # RealtimeProcessorのインスタンスを作成
    processor = RealtimeProcessor(vae_model, rnn_model, device)

    # StreamlitのUploadFileをOpenCVで読めるように一時ファイルに保存
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(video_bytes)
    tfile.close()

    cap = None
    try:
        cap = cv2.VideoCapture(tfile.name)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            st.error("動画が読み込めません。")
            return []
        
        # Streamlitのプログレスバー
        progress_bar = st.progress(0, text="分析を開始します...")

        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            
            # RealtimeProcessorを使って1フレームずつ処理
            score = processor.process_frame(frame)
            anomaly_scores.append(score)
            
            # プログレスバーを更新
            progress_bar.progress((i + 1) / total_frames, text=f"フレーム {i+1}/{total_frames} を処理中...")

        progress_bar.empty() # 完了したらバーを消す

    except Exception as e:
        st.error(f"動画分析中にエラーが発生しました: {e}")
    finally:
        if cap:
            cap.release()
        os.remove(tfile.name) # 一時ファイルを削除

    return anomaly_scores