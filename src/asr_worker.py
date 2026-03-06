#!/usr/bin/env python3
# ==============================================================================
# 檔案：asr_worker.py
# 功能：ASR Worker 語音辨識工作模組 - Qwen3-ASR 模型整合
# 描述：
#     此模組負責將音訊資料轉換為文字 (Speech-to-Text / ASR)。
#     採用非同步工作佇列模式，在獨立執行緒中處理語音辨識任務，
#     避免阻塞主執行緒，確保音訊處理的流暢性。
#
# 設計概念：
#     - 生產者-消費者模式：AudioInput 生產任務，ASRWorker 消費任務
#     - 執行緒安全：使用 queue.Queue 進行執行緒間的安全通訊
#     - 模型載入分離：模型載入與推論分離，優化啟動時間
# ==============================================================================

import threading
import queue
import time
from typing import Optional, Callable
from dataclasses import dataclass
import numpy as np


# ==============================================================================
# ASR 結果資料類別 (ASRResult)
# ==============================================================================
@dataclass
class ASRResult:
    """
    語音辨識結果資料類別

    說明：
        封裝語音辨識的結果，包含識別出的文字及其他相關資訊。

    屬性：
        transcript: str，識別出的文字內容
            - 這是最主要的輸出，通常是使用者說的話
            - 可能為空字串表示無法識別
        language: Optional[str]，語言代碼 (如 "zh", "en")
            - 可選，用於多語言辨識場景
        confidence: float，信心度分數
            - 範圍：0.0 到 1.0
            - 1.0 = 最高信心
            - 預設值：0.0
        duration_ms: int，處理耗時 (毫秒)
            - 從收到音訊到完成辨識的時間

    使用範例：
        result = ASRResult(
            transcript="你好世界",
            language="zh",
            confidence=0.95,
            duration_ms=150
        )
    """

    transcript: str
    language: Optional[str] = None
    confidence: float = 0.0
    duration_ms: int = 0


# ==============================================================================
# ASRWorker 類別
# ==============================================================================
class ASRWorker:
    """
    語音辨識工作者

    說明：
        負責將音訊資料轉換為文字的 worker 模組。
        使用執行緒和佇列實現非同步處理：
        - 主執行緒透過 process() 方法提交音訊任務
        - Worker 執行緒在背景執行 _worker_loop()
        - 辨識完成後透過 on_result 回調通知上層

    設計重點：
        - 非同步處理：不阻塞音訊擷取
        - 任務佇列：使用 queue.Queue 緩衝多個任務
        - 可擴展性：預留 model_path 參數支援多種 ASR 模型

    依賴模型 (預留)：
        - Qwen3-ASR：阿里巴巴的離線語音辨識模型
        - 若未載入模型，會回傳錯誤訊息

    使用流程：
        1. 建立 ASRWorker 實例，設定 on_result 回調
        2. 呼叫 load_model() 載入模型
        3. 呼叫 start() 啟動 worker 執行緒
        4. 透過 process() 提交音訊任務
        5. 在 on_result 回調中取得辨識結果
        6. 呼叫 stop() 停止 worker
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        on_result: Optional[Callable[[ASRResult], None]] = None,
    ):
        """
        建構函式 - 建立 ASRWorker 實例

        說明：
            初始化 ASR Worker，設定模型路徑和回調函式。

        參數：
            model_path: Optional[str]，模型檔案路徑
                - 目前預設為 None，待實作 Qwen3-ASR 時填入
                - 可為相對路徑或絕對路徑
            on_result: Optional[Callable[[ASRResult], None]]，辨識結果回調
                - 函式簽名：callback(result: ASRResult) -> None
                - 辨識完成後會呼叫此函式

        內部變數：
            self._input_queue: queue.Queue，輸入任務佇列
            self._worker_thread: threading.Thread，worker 執行緒
            self._is_running: bool，執行狀態標記
            self._model: 載入的模型物件 (目前為 None)
            self._model_loaded: bool，模型是否已載入
        """
        self.model_path = model_path
        self.on_result = on_result

        # 建立任務佇列，容量無上限
        # 說明：使用 queue.Queue 實現執行緒安全的任務傳遞
        self._input_queue: queue.Queue = queue.Queue()

        # Worker 執行緒 (延遲初始化)
        self._worker_thread: Optional[threading.Thread] = None

        # 執行狀態標記
        self._is_running = False

        # 模型相關變數
        self._model = None
        self._processor = None
        self._model_loaded = False
        self._whisper_model = None

    def load_model(self):
        print("[ASR] 載入 Faster-Whisper 模型...")
        try:
            from faster_whisper import WhisperModel

            model_size = "base"
            print(f"[ASR] 下載模型: faster-whisper-{model_size} (約 140MB)")

            self._whisper_model = WhisperModel(
                model_size, device="cpu", compute_type="int8"
            )
            self._model_loaded = True
            print("[ASR] 模型載入成功！")
        except Exception as e:
            print(f"[ASR] 模型載入失敗: {e}")
            self._model_loaded = False

    def start(self):
        """
        啟動 ASR Worker 執行緒

        說明：
            建立並啟動 worker 執行緒，開始處理任務佇列中的音訊任務。

        參數：
            無

        回傳：
            無

        設計考量：
            - 可安全地多次呼叫 (idempotent)
            - 若已運行則不做任何事
        """
        if self._is_running:
            return

        self._is_running = True

        # 建立守護執行緒 (daemon=True)
        # 說明：守護執行緒會在主程式結束時自動終止，
        #       不會阻擋程式結束
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

    def stop(self):
        """
        停止 ASR Worker

        說明：
            優雅地停止 worker 執行緒：
            1. 設定停止標記
            2. 傳送 None 到佇列，觸發 worker 結束
            3. 等待執行緒結束 (最多 2 秒)

        參數：
            無
        """
        self._is_running = False

        if self._worker_thread:
            # 傳送 None 作為結束訊號
            self._input_queue.put(None)
            # 等待執行緒結束
            self._worker_thread.join(timeout=2.0)
            self._worker_thread = None

    def process(self, audio: np.ndarray) -> bool:
        """
        提交音訊進行辨識

        說明：
            將音訊資料加入任務佇列，等待 worker 執行緒處理。
            這是非同步操作，函式會立即返回。

        參數：
            audio: numpy.ndarray，音訊資料
                - 浮點數陣列，範圍 -1.0 到 1.0
                - 任意長度 (從單字到完整句子)

        回傳：
            bool:
                - True: 成功加入佇列
                - False: worker 未運行

        使用範例：
            # 當 VAD 檢測到完整語句時
            asr_worker.process(utterance.audio)
        """
        if not self._is_running:
            return False

        # 加入任務佇列
        self._input_queue.put(audio)
        return True

    def _worker_loop(self):
        """
        Worker 執行緒主迴圈

        說明：
            在獨立執行緒中運行的主要工作迴圈：
            1. 從輸入佇列取出音訊任務
            2. 若收到 None，則結束迴圈
            3. 呼叫 _recognize() 進行辨識
            4. 若有結果，呼叫 on_result 回調
            5. 標記任務完成

        參數：
            無

        設計考量：
            - 使用 try-except 包裹主要邏輯，確保穩定性
            - 使用 queue.Empty 例外處理逾時
        """
        while self._is_running:
            try:
                # 從佇列取出任務 (最多等待 0.1 秒)
                audio = self._input_queue.get(timeout=0.1)

                # 收到結束訊號
                if audio is None:
                    break

                # 執行語音辨識
                result = self._recognize(audio)

                # 若有結果，回調上層
                if result and self.on_result:
                    self.on_result(result)

                # 標記任務完成
                self._input_queue.task_done()

            except queue.Empty:
                # 佇列空閒，繼續迴圈
                continue
            except Exception as e:
                # 任何其他例外，印出錯誤但繼續執行
                print(f"[ASR] Error: {e}")

    def _recognize(self, audio: np.ndarray) -> Optional[ASRResult]:
        """
        執行實際的語音辨識

        說明：
            這是核心的辨識函式，目前為預留實作：
            - 若模型未載入，回傳錯誤訊息
            - 若已載入模型，應調用模型進行推論

        參數：
            audio: numpy.ndarray，音訊資料

        回傳：
            Optional[ASRResult]:
                - 辨識結果物件
                - 若模型未載入，回傳帶錯誤訊息的 ASRResult
                - 若發生錯誤，回傳 None

        TODO：
            實作 Faster-Whisper 推論邏輯
        """
        # 檢查模型是否已載入
        if not self._model_loaded:
            return ASRResult(transcript="[ASR model not loaded]")

        # 執行 ASR 推論
        try:
            import numpy as np

            # 確保音訊是正確的格式
            if not isinstance(audio, np.ndarray):
                return ASRResult(transcript="[ASR 音訊格式錯誤]")

            # 使用 faster-whisper
            import tempfile
            import soundfile as sf

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                sf.write(tmp.name, audio, 16000)
                segments, info = self._whisper_model.transcribe(
                    tmp.name, language="zh", beam_size=5
                )
                transcript = "".join([seg.text for seg in segments]).strip()

            return ASRResult(transcript=transcript)

        except Exception as e:
            print(f"[ASR] 推論錯誤: {e}")
            return ASRResult(transcript=f"[ASR error: {e}]")
