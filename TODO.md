# TODO.md - Voice Activated Assistant

## 專案概述
- **目標**: 建立 Python 語音助理，使用 Faster-Whisper + TTS
- **平台**: Windows 11 + Python (GPU/CPU)
- **技術堆疊**: uv, Silero VAD, Faster-Whisper, pyttsx3/espeak-ng

---

## Phase 1: 基礎建設 (Infrastructure)

### 1.1 初始化專案
- [x] 1.1.1 使用 uv 初始化 Python 專案 (`uv init`)
- [x] 1.1.2 建立 pyproject.toml 依賴配置
- [x] 1.1.3 安裝核心依賴套件

### 1.2 建立專案結構
- [x] 1.2.1 建立目錄結構 (src/, config/, tests/)
- [x] 1.2.2 建立 __init__.py 檔案
- [x] 1.2.3 建立 config/config.yaml 範例
- [x] 1.2.4 建立 config/rules.json 範例

### 1.3 Logging 設定
- [x] 1.3.1 建立 logging 配置模組
- [x] 1.3.2 實作中文 log 格式

---

## Phase 2: 音訊輸入與 VAD (Audio Pipeline)

### 2.1 音訊輸入模組
- [x] 2.1.1 實作 audio_input.py - 麥克風收音
- [x] 2.1.2 設定 frame size (30ms)
- [x] 2.1.3 實作音訊 buffer (ring buffer)

### 2.2 VAD 語音偵測
- [x] 2.2.1 實作 vad_segmenter.py - Silero VAD 整合
- [x] 2.2.2 實作停頓偵測 finalize 演算法 (調整為 1.5s)
- [x] 2.2.3 實作最短 utterance 過濾 (<300ms)
- [x] 2.2.4 實作最長 utterance 強制切段 (>15s)

---

## Phase 3: ASR 轉寫 (Speech Recognition)

### 3.1 ASR Worker
- [x] 3.1.1 實作 asr_worker.py - 框架結構
- [x] 3.1.2 實作多執行緒處理
- [x] 3.1.3 實作 Faster-Whisper 模型載入與推論
- [x] 3.1.4 使用 Faster-Whisper base 模型 (140MB)
- [ ] 3.1.5 Qwen3-ASR 模型整合 (網路問題，暫時使用 fallback)

### 3.2 Utterance 處理
- [x] 3.2.1 實作 utterance 合併策略 (gap < 0.4s)
- [x] 3.2.2 實作記憶體釋放機制

---

## Phase 4: 規則引擎 (Rule Engine)

### 4.1 規則系統
- [x] 4.1.1 設計 JSON rules schema
- [x] 4.1.2 實作 rule_engine.py - 關鍵字匹配
- [x] 4.1.3 實作 match_mode (contains/regex/exact)
- [x] 4.1.4 實作 priority 排序

### 4.2 冷卻與熱更新
- [x] 4.2.1 實作 cooldown 機制 (RAM)
- [x] 4.2.2 實作 hot reload (偵測 mtime 變更)

---

## Phase 5: TTS 播放 (Text-to-Speech)

### 5.1 TTS Worker
- [x] 5.1.1 實作 tts_worker.py - 框架結構
- [x] 5.1.2 實作 pyttsx3/espeak-ng 輸出 (跨平台)
- [x] 5.1.3 實作 TTS voice 設定 (中文)
- [ ] 5.1.4 Qwen3-TTS 模型整合 (進階)

### 5.2 Queue 機制
- [x] 5.2.1 實作 TTS queue (排隊播放)
- [x] 5.2.2 實作 max_queue_size 上限

---

## Phase 6: 狀態機與協調 (Orchestrator)

### 6.1 狀態機
- [x] 6.1.1 實作 orchestrator.py - 狀態機協調
- [x] 6.1.2 實作狀態: LISTENING → ASR_PROCESSING → SPEAKING → LISTENING
- [x] 6.1.3 實作 speaking_event 同步 (threading.Event)

### 6.2 ASR/TTS 互斥
- [x] 6.2.1 實作 TTS 期間音訊丟棄
- [x] 6.2.2 實作 resume_grace_s 延遲 (0.2s)
- [x] 6.2.3 實作 audio buffer 清空

---

## Phase 7: 測試與驗收 (Testing)

### 7.1 測試
- [ ] 7.1.1 單元測試 - 各模組獨立測試
- [ ] 7.1.2 整合測試 - 端到端流程
- [ ] 7.1.3 壓力測試 - 30 次連續觸發

### 7.2 Debug 模式
- [x] 7.2.1 實作 debug 開關
- [x] 7.2.2 實作 mock 測試模式 (`--mock-mode`)
- [x] 7.2.3 實作偵錯訊息輸出 (VAD/ASR/RULE/TTS 狀態)
- [x] 7.2.4 實作命令列測試 (`--test "關鍵字"`)
- [x] 7.2.5 實作音訊設備列表 (`--list-devices`)

---

## Phase 8: 文件與交付 (Documentation)

### 8.1 文件
- [x] 8.1.1 更新 README.md
- [ ] 8.1.2 建立 API 文件 (如有需要)

---

## 驗收標準 (Acceptance Criteria)

1. ✅ 停頓 1.5 秒才輸出文字 (VAD 偵測)
2. ✅ 命中規則必播 TTS (規則引擎)
3. ✅ TTS 期間 ASR 暫停 (狀態機互斥)
4. ✅ 記憶體釋放 (不落盤)
5. ✅ 多執行緒穩定
6. ✅ ASR 語音辨識 (Faster-Whisper base)
