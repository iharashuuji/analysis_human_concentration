import streamlit as st
import cv2
from streamlit_webrtc import webrtc_streamer,  WebRtcMode
import av
import queue
import pandas as pd
import time

# ユーティリティからロガー、ローダー、プロセッサーをインポート
from utils.logger import setup_logger
from utils.loader import load_models
from utils.processor import RealtimeProcessor

# --- 0. ログ設定 ---
logger = setup_logger()

st.set_page_config(layout="wide")
st.title("リアルタイム集中度分析デモ")

# --- 1. モデルをロード ---
# st.cache_resourceでモデルのロードをキャッシュし、再実行時の高速化を図る
logger.info("Loading models...")
vae_model, rnn_model, device = load_models()

if vae_model and rnn_model:
    logger.info("Models loaded successfully.")
    st.success(f"モデルのロードが完了しました (使用デバイス: {device})")

    # --- 2. リアルタイム処理用のプロセッサをインスタンス化 ---
    processor = RealtimeProcessor(vae_model, rnn_model, device)

    # 結果（スコア）をWebRTCコールバックとメインスレッドで共有するためのキュー
    result_queue: "queue.Queue[float]" = queue.Queue()

    def video_frame_callback(frame: av.VideoFrame, processor: RealtimeProcessor) -> av.VideoFrame:
        """WebRTCからフレームを受け取るたびに呼ばれるコールバック関数"""
        logger.debug("video_frame_callback called.")
        img = frame.to_ndarray(format="bgr24")
        # RealtimeProcessorを使って1フレームを処理し、異常スコア（集中度の低さ）を取得
        score = processor.process_frame(img)

        # 結果をキューに入れてメインスレッドに渡す
        logger.debug(f"Putting score to queue: {score:.4f}")
        result_queue.put(score)

        # 映像にスコアを描画
        text = f"Anomaly Score: {score:.4f}"
        cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

    # video_frame_callback に processor インスタンスを束縛(bind)する
    bound_video_frame_callback = lambda frame: video_frame_callback(frame, processor)

    # --- 3. WebRTCでカメラ映像を表示 ---
    webrtc_ctx = webrtc_streamer(
        key="realtime-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
        # 束縛したコールバックを渡す
        video_frame_callback=bound_video_frame_callback,
        async_processing=True,
    )

    # --- 4. UI要素の配置 ---
    col1, col2 = st.columns([1, 2])
    with col1:
        if st.button("RNNの状態をリセット"):
            processor.reset()
            st.info("プロセッサの状態がリセットされました。")

    # --- 5. 異常スコアをリアルタイムでプロット ---
    if webrtc_ctx.state.playing:
        st.subheader("集中度スコアの推移 (スコアが低いほど集中)")
        chart_placeholder = st.empty()
        scores_history = []

        while webrtc_ctx.state.playing:
            try:
                score = result_queue.get(timeout=1.0)
                logger.debug(f"Got score from queue: {score:.4f}")
                scores_history.append(score)
                # グラフを更新
                chart_placeholder.line_chart(pd.DataFrame({'score': scores_history}))
                time.sleep(0.01) # 描画のための短い待機
            except queue.Empty:
                logger.debug("Queue is empty, continuing...")
                continue # ループを終了させずに続ける
else:
    logger.error("Failed to load models. Please run the training script first.")
    st.error("モデルのロードに失敗しました。先に `world_model_train.py` を実行してモデルファイルを生成してください。")