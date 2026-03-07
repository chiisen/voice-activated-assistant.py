#!/usr/bin/env python3
# ==============================================================================
# 檔案：tts_worker.py
# 功能：TTS Worker 文字轉語音工作模組 - 使用 espeak-ng 進行語音合成
# 描述：
#     此模組負責將文字轉換為語音輸出 (Text-to-Speech / TTS)。
#     目前使用 espeak-ng 文字轉語音引擎，未來可擴展支援其他 TTS 引擎
#     (如 Coqui TTS、VITS、Azure TTS 等)。
#
# 設計概念：
#     - 非同步處理：使用執行緒和任務佇列實現非同步朗讀
#     - 事件通知：使用 threading.Event 通知其他模組當前是否正在說話
#     - 流量控制：限制任務佇列大小，防止任務堆積
#     - 打斷支援：透過 speaking_event 支援打斷式對話
# ==============================================================================

import threading
import queue
import time
import subprocess
import torch
import numpy as np
import sounddevice as sd
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass
from .rule_engine import TTSJob


# ==============================================================================
# TTS 結果資料類別 (TTSResult)
# ==============================================================================
@dataclass
class TTSResult:
    """
    TTS 結果資料類別

    說明：
        封裝 TTS 任務的執行結果。

    屬性：
        job_id: str，對應的規則 ID
            - 用於日誌和追蹤
        success: bool，是否成功播放
            - True = 成功
            - False = 失敗
        duration_ms: int，播放持續時間 (毫秒)
            - 從開始朗讀到結束的時間
        error: Optional[str]，錯誤訊息
            - 若成功為 None
            - 若失敗則包含錯誤描述
    """

    job_id: str
    success: bool
    duration_ms: int = 0
    error: Optional[str] = None


# ==============================================================================
# TTSWorker 類別
# ==============================================================================
class TTSWorker:
    """
    文字轉語音工作者

    說明：
        負責將文字轉換為語音並播放的 worker 模組。
        使用執行緒和任務佇列實現非同步處理：
        - 主執行緒透過 speak() 方法提交朗讀任務
        - Worker 執行緒在背景執行 _worker_loop()
        - 朗讀完成後透過 on_complete 回調通知上層

    設計重點：
        - 非同步處理：不阻塞主執行緒
        - 任務佇列：使用 queue.Queue 緩衝多個任務
        - 打斷支援：透過 speaking_event 讓其他模組知道是否正在說話
        - 流量控制：限制最大佇列大小

    依賴工具：
        - espeak-ng：开源的文字轉語音引擎
            - 安裝：brew install espeak-ng (macOS) 或 apt-get install espeak-ng (Linux)
            - 參數：-v 語言代碼 -s 語速 要朗讀的文字

    使用流程：
        1. 建立 TTSWorker 實例，設定 on_complete 回調
        2. 呼叫 load_model() 初始化
        3. 呼叫 start() 啟動 worker 執行緒
        4. 透過 speak() 提交朗讀任務
        5. 在 on_complete 回調中處理完成事件
        6. 呼叫 stop() 停止 worker
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        on_complete: Optional[Callable[[TTSResult], None]] = None,
        device: Optional[int] = None,
        device_type: str = "auto",
        default_voice: str = "vivian"
    ):
        """
        建構函式 - 建立 TTSWorker 實例

        說明：
            初始化 TTS Worker，設定模型路徑和回調函式。

        參數：
            model_path: Optional[str]，模型/配置路徑
                - 目前使用 espeak-ng，預留給其他 TTS 引擎
            on_complete: Optional[Callable[[TTSResult], None]]，完成回調
                - 函式簽名：callback(result: TTSResult) -> None
                - 朗讀完成後會呼叫此函式

        內部變數：
            self._job_queue: queue.Queue，任務佇列
            self._worker_thread: threading.Thread，worker 執行緒
            self._is_running: bool，執行狀態標記
            self._is_speaking: bool，是否正在說話
            self._max_queue_size: int，最大佇列大小
            self._speaking_event: threading.Event，說話事件 (用於打斷)
        """
        self.model_path = model_path
        self.on_complete = on_complete
        self.device = device
        self.device_type = device_type
        self.default_voice = default_voice

        # 建立任務佇列
        self._job_queue: queue.Queue = queue.Queue()

        # Worker 執行緒
        self._worker_thread: Optional[threading.Thread] = None

        # 執行狀態
        self._is_running = False
        self._is_speaking = False

        # TTS 引擎
        self._engine = None

        # 任務佇列最大容量 (超過此數量會拒絕新任務)
        self._max_queue_size = 10

        # 建立說話事件
        # 說明：此 Event 用於通知其他模組當前是否正在說話
        #       Orchestrator 會檢查此事件來決定是否要忽略新的音訊輸入
        self._speaking_event = threading.Event()

    # =========================================================================
    # 屬性存取器
    # =========================================================================

    @property
    def is_speaking(self) -> bool:
        """
        檢查是否正在說話

        回傳：
            bool: True = 正在朗讀
        """
        return self._is_speaking

    @property
    def speaking_event(self) -> threading.Event:
        """
        取得說話事件物件

        說明：
            此 Event 會在開始說話時設為 True，說完後設為 False。
            其他模組 (如 Orchestrator) 可以透過檢查此事件
            來判斷是否要打斷當前說話。

        回傳：
            threading.Event: 說話事件物件
                - is_set() = True 表示正在說話
        """
        return self._speaking_event

    def load_model(self):
        print(f"[TTS] 載入 Qwen3-TTS 模型自: {self.model_path}...")
        try:
            import torch
            from qwen_tts import Qwen3TTSModel

            # 設定裝置與資料型別
            if self.device_type == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = self.device_type
            
            # 針對 Ampere (30系列) 以上顯卡啟用 TF32 加速
            if torch.cuda.is_available():
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

            # 使用 bfloat16 以符合 Qwen3 原生型別效能與穩定性 (針對 RTX 30+ 最佳化)
            torch_dtype = torch.bfloat16 if "cuda" in device else torch.float32
            
            print(f"[TTS] 準備載入 Qwen3-TTS 模型 (DType: {torch_dtype}, 裝置: {device})...", flush=True)

            # 載入 Qwen3-TTS 模型
            self._engine = Qwen3TTSModel.from_pretrained(
                self.model_path, 
                device_map=device,
                dtype=torch_dtype,
                attn_implementation="sdpa",
                trust_remote_code=True
            )
            
            # 強制檢查設備屬性
            actual_device = next(self._engine.model.parameters()).device
            print(f"[TTS] 模型載入完畢 | 物理裝置: {actual_device} | 型別: {next(self._engine.model.parameters()).dtype}")
        except Exception as e:
            print(f"[TTS] 模型載入失敗: {e}, 將使用系統預設 TTS 作為備援")
            self._load_fallback_engine()
    
    def _load_fallback_engine(self):
        """載入系統內建的 TTS 引擎作為備援"""
        import platform
        system = platform.system()
        if system == "Windows":
            try:
                import pyttsx3
                self._fallback_engine = pyttsx3.init()
            except ImportError:
                self._fallback_engine = None
        else:
            self._fallback_engine = None

    def start(self):
        """
        啟動 TTS Worker 執行緒

        說明：
            建立並啟動 worker 執行緒，開始處理任務佇列中的朗讀任務。

        參數：
            無

        回傳：
            無
        """
        if self._is_running:
            return

        self._is_running = True

        # 建立守護執行緒
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

    def stop(self):
        """
        停止 TTS Worker

        說明：
            優雅地停止 worker 執行緒。

        參數：
            無
        """
        self._is_running = False

        if self._worker_thread:
            # 傳送 None 作為結束訊號
            self._job_queue.put(None)
            # 等待執行緒結束
            self._worker_thread.join(timeout=2.0)
            self._worker_thread = None

    def speak(self, job: TTSJob) -> bool:
        """
        提交朗讀任務

        說明：
            將 TTS 任務加入佇列，等待 worker 執行緒處理。

        參數：
            job: TTSJob，要朗讀的任務

        回傳：
            bool:
                - True: 成功加入佇列
                - False: 佇列已滿 (超過 _max_queue_size)

        流量控制：
            - 若佇列已滿，會回傳 False
            - 這是為了防止任務堆積導致延遲過大
        """
        # 檢查佇列是否已滿
        if self._job_queue.qsize() >= self._max_queue_size:
            return False

        # 加入任務佇列
        print(f"[TTS] 提交朗讀任務到佇列: 「{job.text}」", flush=True)
        self._job_queue.put(job)
        return True

    def _worker_loop(self):
        """
        Worker 執行緒主迴圈

        說明：
            1. 從任務佇列取出朗讀任務
            2. 將任務拆分為多個子句 (Streaming 基礎)
            3. 依序產生並播放，實現「邊生邊播」的效果
        """
        print("[TTS] Worker 執行緒已啟動", flush=True)
        while self._is_running:
            try:
                # 從佇列取出任務 (最多等待 1.0 秒)
                job = self._job_queue.get(timeout=1.0)

                # 收到結束訊號
                if job is None:
                    break

                # 執行朗讀 (內部已支援串流化處理)
                self._speak_streaming(job)

                # 標記任務完成
                self._job_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                print(f"[TTS] Worker Error: {e}")

    def _split_into_sentences(self, text: str) -> list[str]:
        """將長句拆分為短句以支援串流播放"""
        # 使用正則表達式根據標點符號拆分，並保留標點
        import re
        # 匹配非標點內容 + 後隨的標點
        pattern = r'([^，。！？,;.!?]+[，。！？,;.!?]*)'
        sentences = re.findall(pattern, text)
        
        # 若沒匹配到(可能沒標點)，直接回傳原句
        if not sentences:
            return [text]
        
        # 過濾空字串並去除首尾空白
        return [s.strip() for s in sentences if s.strip()]

    def _speak_streaming(self, job: TTSJob):
        """串流化朗讀實作：拆分 -> 生一段 -> 播一段"""
        print(f"[TTS] 開始處理任務: 「{job.text}」", flush=True)
        self._is_speaking = True
        self._speaking_event.set()
        start_time = time.time()

        # 1. 拆分句子
        sentences = self._split_into_sentences(job.text)
        print(f"[TTS] 偵測到 {len(sentences)} 個語句段落，進入串流模式...", flush=True)

        # 2. 建立播放佇列與播放執行緒 (本任務專用)
        playback_queue = queue.Queue()
        playback_stop_event = threading.Event()

        def playback_worker():
            import sounddevice as sd
            while not playback_stop_event.is_set() or not playback_queue.empty():
                try:
                    # 從佇列取得音訊片段
                    chunk = playback_queue.get(timeout=0.1)
                    if chunk is None: break
                    
                    audio_data, sr = chunk
                    # 播放
                    sd.play(audio_data, sr, device=self.device)
                    sd.wait()
                    playback_queue.task_done()
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"[TTS] Playback Thread Error: {e}")
                    break

        playback_thread = threading.Thread(target=playback_worker, daemon=True)
        playback_thread.start()

        # 3. 逐句生成並餵入播放佇列
        try:
            import torch
            import numpy as np

            for i, sent in enumerate(sentences):
                # 檢查是否被外部打斷 (例如 Orchestrator 清除了事件)
                if not self._speaking_event.is_set():
                    print("[TTS] 偵測到打斷，停止後續段落生成", flush=True)
                    break

                print(f"[TTS] 串流生成中 ({i+1}/{len(sentences)}): {sent}", flush=True)

                if hasattr(self._engine, 'generate_custom_voice'):
                    speaker = job.voice or self.default_voice
                    with torch.inference_mode():
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        
                        wavs, sr = self._engine.generate_custom_voice(
                            sent, 
                            speaker=speaker,
                            do_sample=False,
                            max_new_tokens=512
                        )
                    
                    # 將結果放入播放佇列
                    playback_queue.put((wavs[0], sr))
                else:
                    # 備援模式不支援細粒度串流，一次跑完
                    self._speak_fallback(TTSJob(rule_id=job.rule_id, text=sent))

        except Exception as e:
            print(f"[TTS] Streaming Generation Error: {e}")
        finally:
            # 結束播放執行緒
            playback_queue.put(None)
            playback_stop_event.set()
            playback_thread.join(timeout=2.0)

        # 4. 結束處理
        duration_ms = int((time.time() - start_time) * 1000)
        result = TTSResult(job_id=job.rule_id, success=True, duration_ms=duration_ms)
        self._is_speaking = False
        self._speaking_event.clear()
        if self.on_complete:
            self.on_complete(result)

    def _speak_fallback(self, job: TTSJob):
        """備援的朗讀實作"""
        if hasattr(self, '_fallback_engine') and self._fallback_engine:
             self._fallback_engine.say(job.text)
             self._fallback_engine.runAndWait()
        else:
            # 最後一線：使用系統指令
            import subprocess
            try:
                subprocess.run(["espeak-ng", "-v", "zh", job.text], capture_output=True)
            except:
                print(f"[TTS] 無法播放語音: {job.text}")
