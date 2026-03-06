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
from typing import Optional, Callable
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

        # 建立任務佇列
        self._job_queue: queue.Queue = queue.Queue()
        
        # Worker 執行緒
        self._worker_thread: Optional[threading.Thread] = None
        
        # 執行狀態
        self._is_running = False
        self._is_speaking = False

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
        """
        初始化 TTS 引擎
        
        說明：
            目前的實作使用 espeak-ng (系統命令)，無需額外模型載入。
            此函式預留給其他 TTS 引擎使用。
        
        參數：
            無
        
        回傳：
            無
        """
        print("[TTS] Using espeak-ng for speech synthesis")

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
        self._worker_thread = threading.Thread(
            target=self._worker_loop, 
            daemon=True
        )
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
        self._job_queue.put(job)
        return True

    def _worker_loop(self):
        """
        Worker 執行緒主迴圈
        
        說明：
            在獨立執行緒中運行：
            1. 從輸入佇列取出朗讀任務
            2. 若收到 None，則結束迴圈
            3. 呼叫 _speak() 進行朗讀
            4. 標記任務完成
        
        參數：
            無
        """
        while self._is_running:
            try:
                # 從佇列取出任務 (最多等待 0.1 秒)
                job = self._job_queue.get(timeout=0.1)

                # 收到結束訊號
                if job is None:
                    break

                # 執行朗讀
                self._speak(job)

                # 標記任務完成
                self._job_queue.task_done()

            except queue.Empty:
                # 佇列空閒，繼續迴圈
                continue
            except Exception as e:
                # 例外處理
                print(f"[TTS] Error: {e}")

    def _speak(self, job: TTSJob):
        """
        執行文字朗讀
        
        說明：
            使用 espeak-ng 將文字轉換為語音並播放。
            流程：
            1. 設定說話狀態 (設定 speaking_event)
            2. 記錄開始時間
            3. 執行 espeak-ng 命令
            4. 計算耗時
            5. 建立 TTSResult
            6. 呼叫 on_complete 回調
            7. 清除說話狀態
        
        參數：
            job: TTSJob，要朗讀的任務
        
        espeak-ng 參數說明：
            -v zh: 使用中文聲音
            -s 140: 語速 140 WPM (每分鐘字數)
            job.text: 要朗讀的文字
        """
        # 設定說話狀態
        self._is_speaking = True
        self._speaking_event.set()

        # 記錄開始時間
        start_time = time.time()

        try:
            # 執行 espeak-ng 命令
            # 說明：subprocess.run() 會等待命令執行完成
            #       capture_output=True 隱藏命令輸出
            #       timeout=10 防止命令卡住
            subprocess.run(
                ["espeak-ng", "-v", "zh", "-s", "140", job.text],
                capture_output=True,
                timeout=10,
            )
        except FileNotFoundError:
            # espeak-ng 未安裝
            print("[TTS] espeak not found, skipping audio output")
        except Exception as e:
            # 其他錯誤
            print(f"[TTS] Speak error: {e}")

        # 計算耗時
        duration_ms = int((time.time() - start_time) * 1000)

        # 建立結果物件
        result = TTSResult(job_id=job.rule_id, success=True, duration_ms=duration_ms)

        # 呼叫完成回調
        if self.on_complete:
            self.on_complete(result)

        # 清除說話狀態
        self._is_speaking = False
        self._speaking_event.clear()
