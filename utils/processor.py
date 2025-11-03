import streamlit as st
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image
import os
import tempfile
import time
import logging

logger = logging.getLogger(__name__)

# --- 訓練コードからの定数 ---
Z_DIM = 32
# 異常スコアの重み係数 (調整が必要なハイパーパラメータ)
# 予測誤差 (NLL) を重視し、再構成誤差 (MSE) を補助的に使う想定
ALPHA = 0.7  # NLL (予測誤差) の重み
BETA = 0.3   # MSE (再構成誤差) の重み 

class RealtimeProcessor:
    """
    WebRTCから受信したフレームに対して、顔検出、VAEエンコード、
    MDN-RNN予測を行い、**予測誤差と再構成誤差を組み合わせた異常スコア**を計算する。
    """
    def __init__(self, vae_model, rnn_model, face_cascade, device, num_gaussians=5):
        self.vae_model = vae_model
        self.rnn_model = rnn_model
        self.face_cascade = face_cascade # 新規に追加
        self.device = device
        self.num_gaussians = num_gaussians
        
        self.reset()
        
        # VAE用の画像前処理
        self.transform = transforms.Compose([
            transforms.Resize((64, 64), antialias=True),
            transforms.ToTensor(),
        ])
        
        # MDN-RNNの損失関数 (予測誤差計算用)
        self.gmm_nll_loss = self._gmm_nll_loss_func
        self.mse_loss = nn.MSELoss(reduction='none') # 再構成誤差計算用

        # 連続した異常スコア計算を抑制するためのカウンタ (毎フレーム実行しない)
        self.inference_skip_count = 0
        self.INFERENCE_SKIP = 3 # 3フレームに1回推論を実行

        # 異常度を正規化するための平均値 (学習データから事前に計算・設定すべき)
        # 訓練中に得られた正常時の平均値を暫定値として設定
        self.NORMAL_NLL_MEAN = 10.0  # (暫定値 - 実際の訓練後に要調整)
        self.NORMAL_MSE_MEAN = 0.005 # (暫定値 - 実際の訓練後に要調整)

    def reset(self):
        """RNNの状態をリセットする"""
        logger.info("Resetting RealtimeProcessor state.")
        self.last_z = None
        self.hidden_state = None
        self.anomaly_scores = []
        self.last_face_rect = None
        self.last_face_detected = False

        
    def _gmm_nll_loss_func(self, target, mus, sigmas, log_pi):
        """MDN-RNNのNLL損失関数を異常スコア計算用に流用 (NLL: 予測誤差)"""
        # (以前のgmm_nll_loss関数をそのまま使用)
        z_dim = self.vae_model.z_dim
        target = target.unsqueeze(2) 
        log_sigma = torch.log(sigmas)
        exponent = -0.5 * ((target - mus) / sigmas) ** 2
        log_prob_const = -0.5 * z_dim * np.log(2 * np.pi)
        log_probs = log_prob_const - log_sigma.sum(dim=-1) + exponent.sum(dim=-1)
        log_likelihood = torch.logsumexp(log_pi + log_probs, dim=-1)
        return -log_likelihood.item()

    def process_frame(self, img_bgr):
        """
        単一のフレームを処理し、最終異常スコアを返す
        """
        # 1. 処理頻度の制御
        self.inference_skip_count += 1
        if self.inference_skip_count < self.INFERENCE_SKIP:
            # 推論をスキップする場合、直前の結果を返す
            last_score = self.anomaly_scores[-1] if self.anomaly_scores else 0.0
            return last_score, self.last_face_rect, self.last_face_detected

        self.inference_skip_count = 0 
        
        # 2. 顔検出
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        nll_score = self.NORMAL_NLL_MEAN * 3 # NLL初期値 (異常と見なすために高く設定)
        mse_score = self.NORMAL_MSE_MEAN * 5 # MSE初期値
        
        # 処理対象の顔が検出されたかどうかのフラグ
        face_detected = False
        # 描画用の顔領域
        face_rect_for_drawing = None
        
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            face_roi = gray[y:y+h, x:x+w]
            face_pil = Image.fromarray(face_roi)
            face_tensor = self.transform(face_pil).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                # VAEの順伝播 (再構成画像とzを取得)
                recon_x, mu, logvar = self.vae_model(face_tensor) # ★recon_xを取得★
                current_z = mu.unsqueeze(1) # (1, 1, z_dim)
                
                # --- 予測誤差 (NLL) の計算 ---
                if self.last_z is not None and self.hidden_state is not None:
                    try:
                        mus, sigmas, log_pi, next_hidden = self.rnn_model(self.last_z, self.hidden_state)
                        current_nll = self.gmm_nll_loss(current_z, mus, sigmas, log_pi)
                        self.hidden_state = next_hidden
                    except Exception as e:
                        # rnn_modelがMDNRNNでない場合にエラーになる可能性
                        logger.error(f"Error during RNN prediction. Is the loaded model an MDNRNN? Error: {e}")
                        # エラー時は暫定値を使用
                        current_nll = self.NORMAL_NLL_MEAN
                else:
                    current_nll = self.NORMAL_NLL_MEAN # 初期フレームは平均値を使用
                    # MDNRNNの隠れ状態は(h, c)のタプル
                    self.hidden_state = (torch.zeros(1, 1, self.rnn_model.rnn_hidden_dim).to(self.device),
                                         torch.zeros(1, 1, self.rnn_model.rnn_hidden_dim).to(self.device))
                
                self.last_z = current_z
                nll_score = current_nll

                # --- 再構成誤差 (MSE) の計算 ---
                # 元画像(input)と再構成画像(recon_x)のMSE (ピクセル値は0〜1)
                # MSE = (1/N) * sum((x - recon_x)^2)
                mse_result = self.mse_loss(recon_x, face_tensor).mean().item() # 平均をとる
                mse_score = mse_result
            
            face_detected = True
            face_rect_for_drawing = (x, y, w, h)
        
        # 5. 最終異常スコアの計算 (2つの誤差を組み合わせる)
        # スコアを正規化（オプション：標準偏差で割るなど）した上で、加重平均を取る
        
        # (簡略化のため、ここでは単純な加重平均を使用)
        # NLLスコアとMSEスコアを、それぞれの正常平均値で割って「正規化」する
        normalized_nll = nll_score / self.NORMAL_NLL_MEAN
        normalized_mse = mse_score / self.NORMAL_MSE_MEAN

        # 直前の結果を保存
        self.last_face_rect = face_rect_for_drawing
        self.last_face_detected = face_detected
        
        final_anomaly_score = (ALPHA * normalized_nll) + (BETA * normalized_mse)
        
        # 最終スコアを履歴に追加
        self.anomaly_scores.append(final_anomaly_score)
        
        # スコアと、描画用の顔領域、検出フラグを返す
        return final_anomaly_score, face_rect_for_drawing, face_detected



def analyze_video(video_bytes, vae_model, rnn_model, device):
    """
    アップロードされた動画を分析し、異常スコア（予測誤差）のリストを返す。
    """
    # 顔検出器をロード
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    anomaly_scores = []
    # RealtimeProcessorのインスタンスを作成
    # num_gaussiansはモデル学習時の値に合わせる
    processor = RealtimeProcessor(vae_model, rnn_model, face_cascade, device, num_gaussians=5)

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
            score, _, _ = processor.process_frame(frame) # スコアのみ取得
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