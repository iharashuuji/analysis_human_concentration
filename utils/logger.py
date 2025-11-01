import logging
import sys

LOG_FILE = "app.log"

def setup_logger():
    """
    アプリケーション全体で使用するロガーを設定する。
    コンソールとログファイルの両方に出力する。
    """
    # ルートロガーを取得
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 既存のハンドラをクリア
    if logger.hasHandlers():
        logger.handlers.clear()

    # フォーマッターの定義
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # ファイルハンドラ
    file_handler = logging.FileHandler(LOG_FILE, mode='w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # コンソールハンドラ
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger