#!/usr/bin/env python3
# ==============================================================================
# 檔案：orchestrator.py
# 功能：Orchestrator 協調器 - 語音助理的核心狀態機與執行緒協調中心
# 描述：
#     此模組是語音助理的大腦，負責協調 AudioInput、VADSegmenter、ASRWorker、
#     RuleEngine 和 TTSWorker 等子模組之間的工作流程。它實作了一個狀態機
#     (State Machine) 來管理語音助理的不同狀態 (Listening, Processing, Speaking 等)，
#     確保各模組之間的互動是有序且執行緒安全的。
#
# 設計概念：
#     - 狀態機 (State Machine)：使用列舉 (Enum) 定義系統狀態，確保狀態轉換是可預測的
#     - 執行緒安全 (Thread Safety)：使用 threading.Lock 保護共享狀態，防止競爭條件
#     - 事件驅動 (Event-Driven)：各子模組透過回調函式 (Callback) 通知 Orchestrator
# ==============================================================================

import threading
import time
from enum import Enum
from typing import Optional, Callable
from dataclasses import dataclass

from .audio_input import AudioInput, AudioConfig
from .vad_segmenter import VADSegmenter, VADConfig, Utterance
from .asr_worker import ASRWorker, ASRResult
from .rule_engine import RuleEngine, TTSJob
from .tts_worker import TTSWorker


# ==============================================================================
# 狀態列舉 (State Enum)
# ==============================================================================
# 說明：定義語音助理的所有可能狀態，用於追蹤系統目前的工作模式
class State(Enum):
    """
    語音助理狀態列舉
    
    成員說明：
        - LISTENING: 等待語音輸入狀態，系統正在監聽麥克風
        - ASR_PROCESSING: 語音辨識中，音訊正在被轉換為文字
        - SPEAKING: 說話中，TTS 正在輸出語音
        - STOPPED: 程式已停止，所有執行緒已結束
        - ERROR: 發生錯誤，需要人工介入或重置
    """
    LISTENING = "LISTENING"
    ASR_PROCESSING = "ASR_PROCESSING"
    SPEAKING = "SPEAKING"
    STOPPED = "STOPPED"
    ERROR = "ERROR"


# ==============================================================================
# 組態資料類別 (OrchestratorConfig)
# ==============================================================================
@dataclass
class OrchestratorConfig:
    """
    Orchestrator 協調器的組態資料類別
    
    說明：
        使用 Python 的 dataclass 定義所有可設定的參數，
        提供型別安全且具有預設值的組態選項。
    
    屬性：
        sample_rate: 音訊取樣率，預設 16000 Hz (每秒 16000 個樣本)
        frame_duration_ms: 音訊框架持續時間 (毫秒)，預設 30ms
        silence_threshold: 靜音閾值，用於 VAD 判斷是否為語音，預設 0.5
        silence_duration: 最小靜音持續時間 (秒)，預設 1.0 秒
        min_utterance_ms: 最小語句持續時間 (毫秒)，預設 300ms
        max_utterance_s: 最大語句持續時間 (秒)，預設 15 秒
        resume_grace_s: TTS 結束後的緩衝時間 (秒)，預設 0.2 秒
        rules_path: 規則檔案路徑，預設 "config/rules.json"
        debug: 是否啟用除錯模式，預設 False
        audio_device: 音訊裝置索引，None 表示使用預設裝置
        mock_mode: 是否使用模擬模式 (無需實際音訊輸入)，預設 False
    
    使用範例：
        config = OrchestratorConfig(
            sample_rate=16000,
            rules_path="config/rules.json",
            debug=True,
            mock_mode=True
        )
    """
    sample_rate: int = 16000
    frame_duration_ms: int = 30
    silence_threshold: float = 0.5
    silence_duration: float = 1.0
    min_utterance_ms: int = 300
    max_utterance_s: int = 15
    resume_grace_s: float = 0.2
    rules_path: str = "config/rules.json"
    debug: bool = False
    audio_device: Optional[int] = None
    mock_mode: bool = False


# ==============================================================================
# Orchestrator 協調器類別
# ==============================================================================
class Orchestrator:
    """
    語音助理的核心協調器
    
    說明：
        Orchestrator 是整個語音助理的心臟，負責：
        1. 管理所有子模組的生命週期 (初始化、起動、停止)
        2. 維護系統狀態機，確保狀態轉換的正確性
        3. 處理各子模組之間的事件傳遞 (Callback 模式)
        4. 協調音訊流程：音訊輸入 -> VAD -> ASR -> 規則匹配 -> TTS -> 說話
    
    設計重點：
        - 執行緒安全：所有狀態存取都透過 Lock 保護
        - 非同步處理：各子模組在獨立執行緒中運作
        - 錯誤復原：支援在 TTS 播放期間繼續監聽 (打斷式對話)
    """
    
    def __init__(self, config: Optional[OrchestratorConfig] = None):
        """
        建構函式 - 初始化 Orchestrator 和所有子模組
        
        說明：
            建立 Orchestrator 實例，初始化所有子模組並設定回調函式。
            注意：此時尚未啟動任何執行緒，僅建立物件關聯。
        
        參數：
            config: OrchestratorConfig 組態物件，若為 None則使用預設值
        
        內部元件：
            - AudioInput: 負責從麥克風擷取音訊
            - VADSegmenter: 負責語音活動檢測和語句分段
            - ASRWorker: 負責將音訊轉換為文字
            - RuleEngine: 負責根據文字匹配對應規則
            - TTSWorker: 負責將文字轉換為語音輸出
        """
        self.config = config or OrchestratorConfig()

        # 初始化執行緒鎖，確保狀態變數的執行緒安全
        self._state = State.STOPPED
        self._state_lock = threading.Lock()

        # 建立 AudioInput 實例 (音訊輸入模組)
        # 說明：callback 指向 _on_audio_frame，用於處理每個音訊框架
        self._audio_input = AudioInput(
            config=AudioConfig(
                sample_rate=self.config.sample_rate,
                frame_duration_ms=self.config.frame_duration_ms,
                device=self.config.audio_device,
            ),
            callback=self._on_audio_frame,
        )

        # 建立 VADSegmenter 實例 (語音活動檢測)
        # 說明：on_utterance 指向 _on_utterance，用於處理完整的語句
        self._vad = VADSegmenter(
            config=VADConfig(
                silence_threshold=self.config.silence_threshold,
                min_silence_duration=self.config.silence_duration,
                min_utterance_ms=self.config.min_utterance_ms,
                max_utterance_s=self.config.max_utterance_s,
                sample_rate=self.config.sample_rate,
            ),
            on_utterance=self._on_utterance,
        )

        # 建立各 Worker 實例
        # ASR: 語音辨識 Worker
        self._asr = ASRWorker(on_result=self._on_asr_result)
        
        # RuleEngine: 規則引擎，載入 rules.json 檔案
        self._rule_engine = RuleEngine(rules_path=self.config.rules_path)
        
        # TTS: 文字轉語音 Worker
        self._tts = TTSWorker(on_complete=self._on_tts_complete)

        # 取得 TTS 的 speaking Event，用於打斷式對話
        self._speaking_event = self._tts.speaking_event

        # 執行緒執行狀態標記
        self._is_running = False

    # =========================================================================
    # 屬性存取器 (Property Accessors)
    # =========================================================================
    
    @property
    def state(self) -> State:
        """
        取得目前系統狀態 (執行緒安全)
        
        說明：
            使用 Lock 保護 _state 變數，確保在多執行緒環境下
            讀取狀態不會取得不一致的結果。
        
        參數：
            無
        
        回傳：
            State: 目前的系統狀態
        """
        with self._state_lock:
            return self._state

    def set_state(self, new_state: State):
        """
        設定系統狀態 (執行緒安全)
        
        說明：
            使用 Lock 保護 _state 變數，確保在多執行緒環境下
            設定狀態不會產生競爭條件。
        
        參數：
            new_state: State 列舉的新狀態值
        """
        with self._state_lock:
            self._state = new_state

    # =========================================================================
    # 生命週期管理 (Lifecycle Management)
    # =========================================================================
    
    def start(self) -> bool:
        """
        啟動 Orchestrator 和所有子模組
        
        說明：
            此函式負責起動所有子模組並將系統狀態設為 LISTENING。
            執行順序：
            1. 載入規則檔 (RuleEngine)
            2. 載入 VAD 模型 (非 Mock 模式)
            3. 起動 ASR Worker
            4. 起動 TTS Worker
            5. 起動 AudioInput (非 Mock 模式)
            6. 設定狀態為 LISTENING
        
        參數：
            無
        
        回傳：
            bool: 
                - True: 起動成功
                - False: 起動失敗 (可能是無可用音訊裝置)
        
        例外：
            若已處於執行狀態，回傳 False 而不重複起動
        """
        # 防止重複起動
        if self._is_running:
            return False

        # Step 1: 載入規則檔
        # 說明：RuleEngine 會解析 rules.json 並建立規則列表
        self._rule_engine.load_rules()

        # Step 2: 載入 VAD 模型
        # 說明：非 Mock 模式下才需要載入 Silero VAD 模型
        if not self.config.mock_mode:
            self._vad.load_vad()

        # Step 3: 起動 ASR Worker
        # 說明：ASR 會在獨立執行緒中執行，不會阻塞主執行緒
        self._asr.load_model()
        self._asr.start()

        # Step 4: 起動 TTS Worker
        # 說明：TTS 會在獨立執行緒中等待任務
        self._tts.load_model()
        self._tts.start()

        # Step 5: 起動 AudioInput 或進入 Mock 模式
        if self.config.mock_mode:
            # Mock 模式：無需音訊輸入，用於測試
            print("[ORCHESTRATOR] Running in MOCK mode - no audio input required")
            self.set_state(State.LISTENING)
            self._is_running = True
            print("[STATE] -> LISTENING (waiting for speech...)")
            return True

        # 起動實際的音訊輸入
        if not self._audio_input.start():
            print("[ORCHESTRATOR] Failed to start audio input")
            return False

        # Step 6: 設定狀態為 LISTENING，開始監聽
        self.set_state(State.LISTENING)
        print("[STATE] -> LISTENING (waiting for speech...)")
        self._is_running = True
        return True

    def stop(self):
        """
        停止 Orchestrator 和所有子模組
        
        說明：
            此函式負責優雅地停止所有子模組：
            1. 停止 AudioInput (停止擷取音訊)
            2. 停止 ASR Worker (停止語音辨識)
            3. 停止 TTS Worker (停止語音輸出)
            4. 設定狀態為 STOPPED
        
        參數：
            無
        
        注意事項：
            - 此函式會等待執行緒結束 (最多數秒)
            - 若程式正在說話，會等待說話完成後才停止
        """
        if not self._is_running:
            return

        # 停止所有子模組
        self._audio_input.stop()
        self._asr.stop()
        self._tts.stop()

        # 更新狀態
        self.set_state(State.STOPPED)
        self._is_running = False

    # =========================================================================
    # 回調函式 (Callback Functions)
    # =========================================================================
    # 說明：以下函式作為 Callback 供子模組呼叫，
    #       當子模組發生特定事件時，會通知 Orchestrator 處理
    
    def _on_audio_frame(self, audio):
        """
        音訊框架回調 - 處理每個新的音訊資料塊
        
        說明：
            此函式由 AudioInput 在收到新的音訊框架時呼叫。
            它會：
            1. 檢查 TTS 是否正在說話 (打斷式對話支援)
            2. 檢查系統狀態
            3. 將音訊框架傳給 VADSegmenter 進行語音檢測
        
        參數：
            audio: numpy.ndarray，音訊資料 (浮點數陣列，範圍 -1.0 到 1.0)
        
        回傳：
            無 (此函式不應阻塞，否則會影響音訊擷取)
        
        設計考量：
            - 此函式在 AudioInput 的執行緒中執行，必須快速返回
            - 透過 VAD 的非同步處理來實現流暢的音訊處理
        """
        # 若 TTS 正在說話，忽略新的音訊框架 (打斷式對話)
        if self._speaking_event.is_set():
            return

        # 若系統處於說話狀態，也忽略音訊
        if self.state == State.SPEAKING:
            return

        # 將音訊傳給 VAD 進行語音活動檢測
        is_speech = self._vad.process_frame(audio)

    def _on_utterance(self, utterance: Utterance):
        """
        語句完成回調 - 當 VAD 檢測到完整語句時呼叫
        
        說明：
            此函式由 VADSegmenter 在檢測到完整語句時呼叫。
            它會：
            1. 顯示偵測到的語句資訊
            2. 將狀態改為 ASR_PROCESSING
            3. 將音訊資料傳給 ASR 進行文字辨識
        
        參數：
            utterance: Utterance 資料類別，包含：
                - audio: 完整語句的音訊資料
                - start_time: 語句開始時間戳
                - end_time: 語句結束時間戳
                - duration_ms: 語句持續時間 (毫秒)
        
        回傳：
            無
        """
        print(f"[VAD] Utterance detected: {utterance.duration_ms}ms")
        
        # 設定狀態為處理中
        self.set_state(State.ASR_PROCESSING)

        # 將音訊傳給 ASR 進行辨識
        self._asr.process(utterance.audio)

    def _on_asr_result(self, result: ASRResult):
        """
        語音辨識結果回調 - 當 ASR 完成文字辨識時呼叫
        
        說明：
            此函式由 ASRWorker 在完成語音辨識後呼叫。
            處理流程：
            1. 取得辨識出的文字 (transcript)
            2. 若文字為空，回到 LISTENING 狀態
            3. 若 TTS 正在說話，跳過此結果
            4. 呼叫 RuleEngine 匹配規則
            5. 若匹配成功，建立 TTSJob 並播放
            6. 若無匹配，回到 LISTENING 狀態
        
        參數：
            result: ASRResult 資料類別，包含：
                - transcript: 辨識出的文字
                - language: 語言代碼 (可選)
                - confidence: 信心度 (0.0 到 1.0)
                - duration_ms: 處理時間
        
        回傳：
            無
        """
        # 取得辨識文字並去除首尾空白
        transcript = result.transcript.strip()

        # 顯示辨識結果
        print(f"[ASR] Result: {transcript!r}")

        # Case 1: 辨識結果為空，回覆 LISTENING
        if not transcript:
            print("[STATE] -> LISTENING (empty transcript)")
            self.set_state(State.LISTENING)
            return

        # Case 2: TTS 正在說話，跳過此結果
        if self._speaking_event.is_set():
            print("[STATE] -> LISTENING (TTS speaking)")
            self.set_state(State.LISTENING)
            return

        # Case 3: 進行規則匹配
        job = self._rule_engine.match(transcript)

        if job:
            # 匹配成功！顯示匹配資訊
            print(f"┌─────────────────────────────────┐")
            print(f"│ 🎯 觸發規則: {job.rule_id:<20} │")
            print(f"│ 📝 辨識文字: {transcript:<20} │")
            print(f"│ 🔊 朗讀內容: {job.text[:20]:<20} │")
            print(f"└─────────────────────────────────┘")
            
            # 建立 TTS Job 並說話
            success = self._tts.speak(job)
            if success:
                self.set_state(State.SPEAKING)
            else:
                # 佇列已滿，等待下次機會
                print("[STATE] -> LISTENING (TTS queue full)")
                self.set_state(State.LISTENING)
        else:
            # 無匹配規則
            print(f"[RULE] No match for: {transcript!r}")
            print("[STATE] -> LISTENING")
            self.set_state(State.LISTENING)

    def _on_tts_complete(self, result):
        """
        TTS 完成回調 - 當 TTS 播放完畢時呼叫
        
        說明：
            此函式由 TTSWorker 在語音播放完成後呼叫。
            它會：
            1. 顯示播放完成的資訊
            2. 等待一小段時間 (resume_grace_s) 讓音訊完全結束
            3. 將狀態改回 LISTENING
        
        參數：
            result: TTSResult 資料類別，包含：
                - job_id: 工作的 ID (對應 rule_id)
                - success: 是否成功播放
                - duration_ms: 播放持續時間
                - error: 錯誤訊息 (若有)
        
        回傳：
            無
        
        設計考量：
            - 加入 resume_grace_s 緩衝時間，避免狀態轉換太快導致問題
        """
        print(f"[TTS] Finished: {result.duration_ms}ms")
        
        # 等待緩衝時間
        time.sleep(self.config.resume_grace_s)
        
        print("[STATE] -> LISTENING (waiting for speech...)")
        self.set_state(State.LISTENING)

    # =========================================================================
    # 測試與除錯辅助函式
    # =========================================================================
    
    def simulate_utterance(self, text: str):
        """
        模擬語句 - 用於測試規則匹配和 TTS
        
        說明：
            此函式用於 Mock 模式下模擬語音輸入。
            它會直接建立一個 ASRResult 並呼叫 _on_asr_result，
            繞過實際的音訊擷取和語音辨識流程。
        
        參數：
            text: str，要模擬的文字內容
        
        使用範例：
            orchestrator.simulate_utterance("你好")
        
        注意事項：
            - 此函式僅用於測試目的
            - 在 Mock 模式下非常有用
        """
        print(f"[MOCK] Simulating utterance: {text}")
        
        # 直接建立 ASRResult 並處理結果
        result = ASRResult(transcript=text)
        self._on_asr_result(result)
