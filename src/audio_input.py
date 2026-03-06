#!/usr/bin/env python3
# ==============================================================================
# 檔案：audio_input.py
# 功能：AudioInput 音訊輸入模組 - 負責從麥克風擷取音訊資料
# 描述：
#     此模組使用 sounddevice 函式庫與系統的音訊驅動程式互動，
#     從指定的音訊輸入裝置 (麥克風) 即時擷取音訊資料，並透過回調函式
#     將資料傳給上層模組 (Orchestrator -> VAD) 進行處理。
#
# 設計概念：
#     - Callback 模式：使用 sounddevice 的回調機制，非同步處理音訊
#     - 區塊式擷取：每次擷取固定大小的音訊區塊 (frames per buffer)
#     - 浮點數標準：輸出標準化的浮點數音訊 (範圍 -1.0 到 1.0)
# ==============================================================================

import numpy as np
import sounddevice as sd
from typing import Callable, Optional
from dataclasses import dataclass


# ==============================================================================
# 音訊組態資料類別 (AudioConfig)
# ==============================================================================
@dataclass
class AudioConfig:
    """
    音訊輸入的組態資料類別
    
    說明：
        定義音訊擷取的所有相關參數，包含取樣率、聲道數、資料類型等。
        使用 dataclass 提供型別安全且易於擴展的組態管理。
    
    屬性：
        sample_rate: 音訊取樣率 (Hz)，預設 16000 Hz
            - 較低的取樣率可以減少運算量，但會降低高頻品質
            - 16000 Hz 是語音處理的標準取樣率
        frame_duration_ms: 每個音訊區塊的持續時間 (毫秒)，預設 30ms
            - 影響音訊處理的延遲和顆粒度
            - 30ms 是語音處理的常用區塊大小
        channels: 聲道數，預設 1 (單聲道)
            - 1 = 單聲道 (Mono)
            - 2 = 立体聲 (Stereo)
        dtype: 資料類型，預設 "float32"
            - float32 是標準的浮點數表示，範圍 -1.0 到 1.0
        device: 音訊裝置索引，None = 使用系統預設裝置
            - 可透過 sd.query_devices() 查詢可用裝置
    
    衍生屬性：
        frame_samples: 根據取樣率和區塊持續時間計算的樣本數
            - 計算公式：sample_rate * frame_duration_ms / 1000
    
    使用範例：
        config = AudioConfig(
            sample_rate=16000,
            frame_duration_ms=30,
            channels=1,
            device=0  # 使用第一個音訊輸入裝置
        )
    """
    sample_rate: int = 16000
    frame_duration_ms: int = 30
    channels: int = 1
    dtype: str = "float32"
    device: Optional[int] = None

    @property
    def frame_samples(self) -> int:
        """
        計算每個音訊區塊的樣本數
        
        說明：
            根據取樣率和區塊持續時間計算每次 Callback 應該處理的樣本數。
            例如：16000 Hz * 30 ms / 1000 = 480 個樣本
        
        參數：
            無
        
        回傳：
            int: 每個區塊的樣本數
        """
        return int(self.sample_rate * self.frame_duration_ms / 1000)


# ==============================================================================
# AudioInput 類別
# ==============================================================================
class AudioInput:
    """
    音訊輸入管理器
    
    說明：
        負責與系統音訊驅動互動，從麥克風即時擷取音訊資料。
        使用 sounddevice 的串流 (Stream) 機制實現低延遲的音訊處理。
    
    設計重點：
        - 非同步處理：音訊擷取在背景執行，不阻塞主執行緒
        - 即時回調：每次收到新的音訊區塊時呼叫回調函式
        - 執行緒安全：sounddevice 內部處理執行緒安全問題
    
    使用流程：
        1. 建立 AudioInput 實例
        2. 呼叫 start() 開始擷取
        3. 在回調函式中處理音訊資料
        4. 呼叫 stop() 停止擷取
    """
    
    def __init__(
        self,
        config: Optional[AudioConfig] = None,
        callback: Optional[Callable[[np.ndarray], None]] = None,
    ):
        """
        建構函式 - 建立 AudioInput 實例
        
        說明：
            初始化音訊輸入管理器，設定組態和回調函式。
            注意：此時尚未啟動音訊串流，必須呼叫 start() 才會開始擷取。
        
        參數：
            config: AudioConfig 組態物件，若為 None則使用預設值
            callback: 回調函式，用於處理每個音訊區塊
                - 函式簽名：callback(audio: np.ndarray) -> None
                - audio 參數為浮點數陣列，範圍 -1.0 到 1.0
        
        內部變數：
            self._stream: sounddevice InputStream 物件 (啟動後有效)
            self._is_running: 執行狀態標記
        """
        self.config = config or AudioConfig()
        self.callback = callback
        self._stream: Optional[sd.InputStream] = None
        self._is_running = False

    def list_devices(self):
        """
        列出所有可用的音訊裝置
        
        說明：
            查詢並顯示系統中所有可用的音訊輸入和輸出裝置。
            此函式用於帮助使用者選擇正確的音訊裝置索引。
        
        參數：
           無
        
        回傳：
            無 (直接輸出到標準輸出)
        
        使用範例：
            $ python main.py --list-devices
            Available audio devices:
            ...
        """
        print("Available audio devices:")
        print(sd.query_devices())

    def start(self) -> bool:
        """
        啟動音訊串流，開始從麥克風擷取音訊
        
        說明：
            建立 sounddevice InputStream 並開始錄音。此函式會：
            1. 檢查是否有可用的音訊輸入裝置
            2. 建立 InputStream 物件
            3. 啟動串流
        
        參數：
            無
        
        回傳：
            bool: 
                - True: 起動成功，串流已開始
                - False: 起動失敗 (可能是無可用裝置或權限問題)
        
        例外：
            若已處於執行狀態，回傳 False
        """
        if self._is_running:
            return

        # Step 1: 檢查是否有可用的音訊輸入裝置
        try:
            devices = sd.query_devices()
            if devices is None or (
                isinstance(devices, dict) and devices.get("max_input_channels", 0) == 0
            ):
                print("[AUDIO] No input devices available!")
                print("Run with --list-devices to see available devices")
                return False
        except Exception as e:
            print(f"[AUDIO] Error querying devices: {e}")
            return False

        # Step 2: 建立並啟動音訊串流
        try:
            # 建立 InputStream
            # 參數說明：
            # - samplerate: 取樣率
            # - channels: 聲道數
            # - dtype: 資料類型 (float32 = 標準化的浮點數)
            # - blocksize: 每個區塊的樣本數
            # - device: 指定的裝置索引 (None = 預設)
            # - callback: 回調函式
            self._stream = sd.InputStream(
                samplerate=self.config.sample_rate,
                channels=self.config.channels,
                dtype=self.config.dtype,
                blocksize=self.config.frame_samples,
                device=self.config.device,
                callback=self._audio_callback,
            )
            
            # 啟動串流
            self._stream.start()
            self._is_running = True
            return True
            
        except Exception as e:
            print(f"[AUDIO] Failed to start audio stream: {e}")
            return False

    def stop(self):
        """
        停止音訊串流
        
        說明：
            優雅地停止音訊串流並釋放資源。
            此函式會：
            1. 檢查是否正在執行
            2. 停止串流
            3. 關閉串流物件
            4. 重設執行狀態
        
        參數：
            無
        
        設計考量：
            - 可安全地多次呼叫 ( idempotent )
            - 確保資源正確釋放
        """
        if not self._is_running:
            return

        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._is_running = False

    def _audio_callback(
        self, indata: np.ndarray, frames: int, time, status: sd.CallbackFlags
    ):
        """
        音訊資料回調 - sounddevice 每次收到新音訊時呼叫
        
        說明：
            此函式由 sounddevice 內部執行緒呼叫，每次有新的音訊區塊時觸發。
            它的職責是：
            1. 處理任何狀態標記 (如 overflow)
            2. 提取單聲道音訊資料
            3. 呼叫上層的回調函式
        
        重要：此函式在 sounddevice 的內部執行緒執行，必須快速返回！
        
        參數：
            indata: numpy.ndarray，輸入音訊資料
                - 形狀：(frames, channels)
                - 類型：float32
            frames: int，這次收到的樣本數
            time: C 結構，時間資訊 (未使用)
            status: sd.CallbackFlags，狀態標記 (如 overflow、input_overflow)
        
        回傳：
            無
        """
        # 處理狀態標記
        if status:
            print(f"[AUDIO] Status: {status}")

        # 提取第一個聲道 (單聲道)
        # 說明：indata 可能包含多聲道，我們只需要第一個聲道
        # 使用 .copy() 避免資料被後續處理修改
        audio_data = indata[:, 0].copy()

        # 呼叫上層回調函式
        if self.callback:
            self.callback(audio_data)

    def is_running(self) -> bool:
        """
        檢查音訊串流是否正在執行
        
        參數：
            無
        
        回傳：
            bool: 
                - True: 正在擷取音訊
                - False: 已停止
        """
        return self._is_running
