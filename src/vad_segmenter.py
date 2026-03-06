#!/usr/bin/env python3
# ==============================================================================
# 檔案：vad_segmenter.py
# 功能：VAD Segmenter 語音活動檢測模組 - 語音偵測與語句分段
# 描述：
#     此模組負責語音活動檢測 (Voice Activity Detection, VAD) 和語句分段。
#     它的主要任務是：
#     1. 判斷音訊中是否包含語音
#     2. 偵測說話的開始和結束
#     3. 將連續的語音片段組合成完整語句
#     4. 過濾掉過短或過長的語句
#
# 設計概念：
#     - 雙模式偵測：支援 Silero VAD (高精確度) 和 Simple VAD (簡單能量偵測)
#     - 緩衝區管理：使用串列儲存語音片段，最後合併為完整語句
#     - 執行緒安全：使用 Lock 保護緩衝區
#     - 可調參數：支援靜音閾值、最小/最大語句長度等配置
#
# VAD 概念說明：
#     VAD (Voice Activity Detection) 是一種用於偵測音訊中是否包含人聲的技術。
#     常用的 VAD 方法包括：
#     - 能量閾值法：計算音訊能量，超過閾值則視為語音 (Simple VAD)
#     - 機器學習法：使用訓練好的模型預測是否為語音 (Silero VAD)
# ==============================================================================

import numpy as np
from typing import Optional, Callable
from dataclasses import dataclass, field
import threading
import time


# ==============================================================================
# VAD 組態資料類別 (VADConfig)
# ==============================================================================
@dataclass
class VADConfig:
    """
    VAD 模組的組態資料類別
    
    說明：
        定義語音活動檢測的所有相關參數。
    
    屬性：
        silence_threshold: float，靜音閾值
            - 用於 Silero VAD：語音機率閾值 (0.0-1.0)
            - 用於 Simple VAD：能量閾值
            - 預設值：0.5
        min_silence_duration: float，最小靜音持續時間 (秒)
            - 當說話停止後，需要持續這麼長時間的靜音才會結束語句
            - 預設值：1.0 秒
        min_utterance_ms: int，最小語句長度 (毫秒)
            - 小於此長度的語句會被過濾掉
            - 預設值：300 ms
        max_utterance_s: float，最大語句長度 (秒)
            - 超過此長度的語句會被截斷
            - 預設值：15 秒
        sample_rate: int，音訊取樣率
            - 預設值：16000 Hz
    """
    silence_threshold: float = 0.5
    min_silence_duration: float = 1.0
    min_utterance_ms: int = 300
    max_utterance_s: int = 15
    sample_rate: int = 16000


# ==============================================================================
# 語句資料類別 (Utterance)
# ==============================================================================
@dataclass
class Utterance:
    """
    語句資料類別
    
    說明：
        封裝一個完整語句的所有資訊。
    
    屬性：
        audio: numpy.ndarray，完整語句的音訊資料
            - 浮點數陣列，範圍 -1.0 到 1.0
            - 多個片段已經過合併
        start_time: float，語句開始時間戳
            - time.time() 的回傳值
        end_time: float，語句結束時間戳
        duration_ms: int，語句持續時間 (毫秒)
    """
    audio: np.ndarray
    start_time: float
    end_time: float
    duration_ms: int


# ==============================================================================
# VADSegmenter 類別
# ==============================================================================
class VADSegmenter:
    """
    語音活動檢測與語句分段器
    
    說明：
        負責從連續的音訊流中偵測語音並組合成完整語句。
        
        工作流程：
        1. 接收每個音訊框架 (frame)
        2. 判斷該框架是否為語音
        3. 若為語音，加入緩衝區
        4. 若為靜音，累積靜音計時
        5. 當靜音超過閾值，觸發語句完成
        6. 呼叫 on_utterance 回調，傳遞完整語句
    
    設計重點：
        - 雙模式支援：Silero VAD (高精確度) 或 Simple VAD (快速fallback)
        - 執行緒安全：使用 Lock 保護共享資料
        - 可配置過濾：支援最小/最大語句長度過濾
    
    依賴套件 (可選)：
        - silero-vad：Google 的預訓練 VAD 模型
        - torch, torchaudio：PyTorch 深度學習框架
    
    使用流程：
        1. 建立 VADSegmenter 實例
        2. 呼叫 load_vad() 載入 VAD 模型
        3. 對每個收到的音訊框架呼叫 process_frame()
        4. 在 on_utterance 回調中處理完整語句
    """
    
    def __init__(
        self,
        config: Optional[VADConfig] = None,
        on_utterance: Optional[Callable[[Utterance], None]] = None,
    ):
        """
        建構函式 - 建立 VADSegmenter 實例
        
        說明：
            初始化 VAD 模組，設定組態和回調函式。
        
        參數：
            config: Optional[VADConfig]，組態物件
            on_utterance: Optional[Callable[[Utterance], None]]，語句完成回調
                - 函式簽名：callback(utterance: Utterance) -> None
                - 當檢測到完整語句時呼叫
        
        內部變數：
            self._buffer: list[numpy.ndarray]，音訊片段緩衝區
            self._buffer_lock: threading.Lock，緩衝區鎖
            self._silence_duration: float，當前靜音持續時間
            self._speech_duration: float，當前說話持續時間
            self._is_speaking: bool，是否正在說話
            self._utterance_start_time: Optional[float]，語句開始時間
            self._vad_model: VAD 模型物件
            self._vad_loaded: bool，模型是否已載入
        """
        self.config = config or VADConfig()
        self.on_utterance = on_utterance

        # 音訊片段緩衝區 (用於儲存語音片段)
        self._buffer: list[np.ndarray] = []
        
        # 緩衝區鎖 (確保執行緒安全)
        self._buffer_lock = threading.Lock()

        # 狀態變數
        self._silence_duration = 0.0
        self._speech_duration = 0.0
        self._is_speaking = False
        self._utterance_start_time: Optional[float] = None

        # VAD 模型相關
        self._vad_model = None
        self._vad_loaded = False

    def load_vad(self):
        """
        載入 VAD 模型
        
        說明：
            嘗試載入 Silero VAD 模型。
            若載入失敗 (缺少依賴套件)，會自動切換到 Simple VAD 模式。
        
        流程：
            1. 嘗試匯入 torch, torchaudio, silero_vad
            2. 若成功，載入 Silero VAD 模型
            3. 若失敗，標記為未載入，使用 Simple VAD
        
        參數：
            無
        
        回傳：
            無
        """
        try:
            # 嘗試匯入必要的套件
            import torch
            import torchaudio
            from silero_vad import load_silero_vad

            # 載入 Silero VAD 模型
            # 說明：Silero Vad 是一個高品質的預訓練 VAD 模型
            #       由 Silero AI 提供，專為即時語音應用設計
            self._vad_model = load_silero_vad()
            self._vad_loaded = True
            
        except ImportError:
            # 缺少必要的套件，使用 Simple VAD
            self._vad_loaded = False
        except Exception:
            # 其他錯誤，同樣使用 Simple VAD
            self._vad_loaded = False

    def process_frame(self, audio: np.ndarray) -> bool:
        """
        處理單個音訊框架
        
        說明：
            這是 VAD 模組的核心函式，每次收到新的音訊資料時呼叫。
            職責：
            1. 判斷是否為語音
            2. 更新緩衝區和狀態
            3. 檢查是否應該結束語句
        
        參數：
            audio: numpy.ndarray，音訊資料
                - 浮點數陣列，範圍 -1.0 到 1.0
                - 長度由 frame_samples 決定
        
        回傳：
            bool: 
                - True: 這個框架包含語音
                - False: 這個框架是靜音
        
        執行緒安全：
            - 此函式使用 _buffer_lock 保護共享狀態
        """
        current_time = time.time()

        # Step 1: 判斷是否為語音
        # 優先使用 Silero VAD，若未載入則使用 Simple VAD
        if not self._vad_loaded:
            is_speech = self._simple_vad(audio)
        else:
            is_speech = self._silero_vad(audio)

        # Step 2: 更新狀態和緩衝區
        with self._buffer_lock:
            if is_speech:
                # ========== 語音區塊 ==========
                if not self._is_speaking:
                    # 新的語句開始
                    self._is_speaking = True
                    self._utterance_start_time = current_time
                    self._buffer = []  # 清空緩衝區
                    print("[VAD] Speech detected, recording...")

                # 加入緩衝區
                self._buffer.append(audio.copy())
                
                # 更新說話持續時間
                self._speech_duration += len(audio) / self.config.sample_rate
                
                # 重設靜音計時
                self._silence_duration = 0.0
                
            else:
                # ========== 靜音區塊 ==========
                if self._is_speaking:
                    # 正在說話中遇到靜音，累積靜音時間
                    self._silence_duration += len(audio) / self.config.sample_rate

                    # 檢查是否應該結束語句
                    if self._silence_duration >= self.config.silence_threshold:
                        print(
                            f"[VAD] Silence for {self._silence_duration:.1f}s - finalizing..."
                        )
                        # 觸發語句完成處理
                        self._finalize_utterance()

        return is_speech

    def _simple_vad(self, audio: np.ndarray) -> bool:
        """
        簡單能量閾值 VAD
        
        說明：
            使用簡單的能量計算來判斷是否為語音。
            計算音訊的 RMS (Root Mean Square) 能量，
            若超過閾值則視為語音。
        
        原理：
            - RMS = sqrt(mean(x^2))
            - RMS 代表訊號的平均功率
        
        參數：
            audio: numpy.ndarray，音訊資料
        
        回傳：
            bool: 
                - True: 能量超過閾值，視為語音
                - False: 能量低於閾值，視為靜音
        
        優點：
            - 計算快速，無需額外依賴
            - 適合簡單場景
        
        缺點：
            - 對噪聲敏感
            - 閾值需要根據環境調整
        """
        # 計算 RMS 能量
        # 說明：np.mean(audio**2) 計算平方的平均值
        #       sqrt() 開根號得到 RMS
        energy = np.sqrt(np.mean(audio**2))
        
        # 能量閾值判斷
        return energy > 0.01

    def _silero_vad(self, audio: np.ndarray) -> bool:
        """
        Silero VAD 語音活動檢測
        
        說明：
            使用 Silero AI 的預訓練 VAD 模型進行語音偵測。
            這是一個深度學習模型，比簡單能量法更精確。
        
        原理：
            - 將音訊轉換為 PyTorch Tensor
            - 輸入 VAD 模型取得語音機率
            - 若機率超過閾值則視為語音
        
        參數：
            audio: numpy.ndarray，音訊資料
        
        回傳：
            bool: 
                - True: 模型的語音機率 > 閾值
                - False: 語音機率 <= 閾值
        
        優點：
            - 高精確度
            - 對噪聲有較好的抵抗力
            - 可偵測多種語言
        
        缺點：
            - 需要額外依賴 (torch, silero-vad)
            - 計算量較大
        """
        if self._vad_model is None:
            # 模型未正確載入，回退到 Simple VAD
            return self._simple_vad(audio)

        try:
            import torch

            # 轉換為 PyTorch Tensor
            tensor = torch.from_numpy(audio).float()
            
            # 取得語音機率
            # 說明：模型輸出 0.0-1.0 的機率值
            #       1.0 = 確認是語音，0.0 = 確認不是語音
            speech_prob = self._vad_model(tensor, self.config.sample_rate).item()
            
            # 與閾值比較
            return speech_prob > self.config.silence_threshold
            
        except Exception:
            # 若發生錯誤，回退到 Simple VAD
            return self._simple_vad(audio)

    def _finalize_utterance(self):
        """
        完成語句處理
        
        說明：
            當偵測到語句結束時呼叫此函式。
            職責：
            1. 檢查語句長度是否符合要求
            2. 合併緩衝區中的所有片段
            3. 建立 Utterance 物件
            4. 呼叫 on_utterance 回調
            5. 重設狀態
        
        參數：
            無
        
        過濾邏輯：
            - 若 duration_ms < min_utterance_ms，丟棄
            - 若 speech_duration > max_utterance_s，丟棄
        """
        # 檢查緩衝區
        if not self._buffer or self._utterance_start_time is None:
            return

        # 計算語句持續時間
        duration_ms = int(self._speech_duration * 1000)

        # 過濾：檢查最小長度
        if duration_ms < self.config.min_utterance_ms:
            self._reset_state()
            return

        # 過濾：檢查最大長度
        if self._speech_duration > self.config.max_utterance_s:
            self._reset_state()
            return

        # 合併所有音訊片段
        audio_data = np.concatenate(self._buffer)

        # 建立 Utterance 物件
        utterance = Utterance(
            audio=audio_data,
            start_time=self._utterance_start_time,
            end_time=time.time(),
            duration_ms=duration_ms,
        )

        # 呼叫回調
        if self.on_utterance:
            self.on_utterance(utterance)

        # 重設狀態
        self._reset_state()

    def _reset_state(self):
        """
        重設內部狀態
        
        說明：
            清空緩衝區並重設所有計時器和狀態變數。
            用於語句完成後或需要重新開始時。
        
        參數：
            無
        """
        self._buffer = []
        self._silence_duration = 0.0
        self._speech_duration = 0.0
        self._is_speaking = False
        self._utterance_start_time = None

    def reset(self):
        """
        公開的重設函式
        
        說明：
            提供給外部呼叫的重設接口。
            會先取得鎖再重設，確保執行緒安全。
        
        參數：
            無
        """
        with self._buffer_lock:
            self._reset_state()
