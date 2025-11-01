# -----------------------------------------------------------------
# ベースイメージ: NVIDIA CUDA + Python (GPU対応)
# -----------------------------------------------------------------
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# CUDAベースの環境にPythonをセットアップ
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# python3.10をデフォルトのpythonに設定（シンボリックリンクの調整）
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# -----------------------------------------------------------------
# 環境変数とワーキングディレクトリ
# -----------------------------------------------------------------
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
WORKDIR /app

# -----------------------------------------------------------------
# 依存ライブラリのインストール
# -----------------------------------------------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# -----------------------------------------------------------------
# グラフィック関連の依存関係 (OpenCV用)
# -----------------------------------------------------------------
# libgl1-mesa-glx のエラー対策として libgl1 をインストール
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------
# アプリケーションコードのコピー
# -----------------------------------------------------------------
COPY . .

EXPOSE 8501

# ★★★ 修正点：ENTRYPOINTをコメントアウトし、CMDでオーバーライド ★★★
# Docker Composeで "command: tail -f /dev/null" を上書きするため、
# ここでは明示的な起動コマンドは指定せず、コンテナ起動状態を維持する
# -----------------------------------------------------------------
# ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
CMD ["tail", "-f", "/dev/null"] 
