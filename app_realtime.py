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
logger.info("--- Streamlit App Started ---")

st.set_page_config(layout="wide")
st.title("リアルタイム集中度分析デモ")
logger.info("Streamlit UI configured.")

# --- モデルパラメータ ---
VAE_MODEL_PATH = "weights/vae_engage3_only.pth"
RNN_MODEL_PATH = "weights/rnn_engage3_only.pth"
Z_DIM = 32
RNN_HIDDEN_DIM = 256

# --- 1. モデルをロード ---
# st.cache_resourceでモデルのロードをキャッシュし、再実行時の高速化を図る
logger.info("Loading models...")
st.info("モデルをロード中です... (初回起動時は時間がかかります)")

vae_model, rnn_model, device = load_models(VAE_MODEL_PATH, RNN_MODEL_PATH, Z_DIM, RNN_HIDDEN_DIM)

if vae_model and rnn_model:
    logger.info("Models loaded successfully.")
    st.success(f"モデルのロードが完了しました (使用デバイス: {device})")

    # --- 顔検出器のロード ---
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # --- 2. セッション状態でプロセッサと状態を管理 ---
    if "processor" not in st.session_state:
        st.session_state.processor = RealtimeProcessor(vae_model, rnn_model, face_cascade, device, num_gaussians=5)
        st.session_state.scores_history = []
        logger.info("RealtimeProcessor and scores_history initialized in st.session_state.")

    processor = st.session_state.processor

    # 結果（スコア）をWebRTCコールバックとメインスレッドで共有するためのキュー
    result_queue: "queue.Queue[float]" = queue.Queue()

    def video_frame_callback(frame: av.VideoFrame, processor: RealtimeProcessor) -> av.VideoFrame:
        """WebRTCからフレームを受け取るたびに呼ばれるコールバック関数"""
        # このログは非常に頻繁に出力されるため、デバッグ時のみ有効化します
        logger.debug("video_frame_callback called.")
        img = frame.to_ndarray(format="bgr24")
        # RealtimeProcessorを使って1フレームを処理
        score, face_rect, face_detected = processor.process_frame(img)

        # 結果をキューに入れてメインスレッドに渡す
        logger.debug(f"Putting score to queue: {score:.4f}")
        result_queue.put(score)

        # 顔が検出されたら四角で囲む
        if face_detected and face_rect:
            (x, y, w, h) = face_rect
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # 映像にスコアを描画（常に表示）
        text = f"Anomaly Score: {score:.4f}"
        cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

    # video_frame_callback に processor インスタンスを束縛(bind)する
    bound_video_frame_callback = lambda frame: video_frame_callback(frame, processor)

    # --- 3. WebRTCでカメラ映像を表示 ---
    logger.info("Setting up webrtc_streamer...")
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
            logger.info("Reset button clicked.")
            st.session_state.scores_history = [] # グラフのデータもリセット
            processor.reset()
            st.info("プロセッサの状態がリセットされました。")

    # --- 5. 異常スコアをリアルタイムでプロット ---
    if webrtc_ctx.state.playing:
        logger.info("WebRTC connection is playing. Starting to plot scores.")
        st.subheader("集中度スコアの推移 (スコアが高いほど集中度が低い)")
        chart_placeholder = st.line_chart(pd.DataFrame({'score': st.session_state.scores_history}))

        while webrtc_ctx.state.playing:
            try:
                # logger.debug("Attempting to get score from queue...")
                score = result_queue.get(timeout=1.0)
                logger.debug(f"Got score from queue: {score:.4f}")
                st.session_state.scores_history.append(score)
                # グラフに新しいデータを追加 (より効率的)
                chart_placeholder.add_rows(pd.DataFrame({'score': [score]}))
                time.sleep(0.01) # 描画のための短い待機
            except queue.Empty:
                # このログも頻繁に出力される可能性があるため、必要に応じてコメントアウトします
                # logger.debug("Queue is empty, continuing...")
                continue # ループを終了させずに続ける
        logger.info("WebRTC connection stopped.")
else:
    logger.error("Failed to load models. Please run the training script first.")
    st.error("モデルのロードに失敗しました。先に `world_model_train.py` を実行してモデルファイルを生成してください。")

logger.info("--- Streamlit App Script Execution Finished ---")