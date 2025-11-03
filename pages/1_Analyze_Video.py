import streamlit as st
import pandas as pd
from utils.loader import load_models
from utils.processor import analyze_video

st.set_page_config(page_title="動画分析", page_icon="📈", layout="wide")
st.title("📈 動画セッションの分析")

# --- モデルパラメータ ---
VAE_MODEL_PATH = "weights/vae_engage3_only.pth"
RNN_MODEL_PATH = "weights/rnn_engage3_only.pth"
Z_DIM = 32
RNN_HIDDEN_DIM = 256

# 1. モデルの読み込み (キャッシュされる)
with st.spinner("世界モデル（VAE+RNN）をロード中..."):
    vae_model, rnn_model, device = load_models(VAE_MODEL_PATH, RNN_MODEL_PATH, Z_DIM, RNN_HIDDEN_DIM)

if vae_model is None or rnn_model is None:
    st.error("モデルのロードに失敗しました。weights/ フォルダを確認してください。")
else:
    st.success(f"モデルのロード完了 (Device: {device})")

    # 2. 動画ファイルのアップロード
    st.header("1. 分析する動画をアップロード")
    uploaded_file = st.file_uploader(
        "分析したい動画ファイル（あくびやよそ見を含むもの）を選択してください",
        type=["mp4", "avi", "mov", "mkv"]
    )

    if uploaded_file is not None:
        st.video(uploaded_file)
        
        # 3. 分析の実行
        st.header("2. 分析の実行")
        if st.button("📈 分析を開始する", type="primary"):
            
            # utils/processor.py の関数を呼び出す
            with st.spinner(f"「{uploaded_file.name}」を分析中... (数分かかることがあります)"):
                anomaly_scores = analyze_video(uploaded_file.getvalue(), vae_model, rnn_model, device)

            if anomaly_scores:
                st.success("分析が完了しました。")
                
                # 4. 結果の表示
                st.header("3. 分析結果")
                st.subheader("異常スコアの時系列グラフ")
                st.write("スコアが高いほど、モデルの「正常（集中）パターン」の予測から逸脱しています。")
                
                # データをPandas DataFrameにするとst.line_chartが使いやすい
                chart_data = pd.DataFrame(
                    anomaly_scores,
                    columns=["Anomaly Score (予測誤差)"]
                )
                st.line_chart(chart_data)
                
                st.subheader("生データ")
                st.dataframe(chart_data)
                
