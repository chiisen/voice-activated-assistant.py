# 語音互動助理 (Voice-Activated Assistant) 產品需求文件 (PRD)
本文檔詳述一套基於本地端 ASR 與 TTS 技術的語音代理系統規範，包含高效的記憶體緩存管理，多執行緒併發處理與秒級停頓偵測判定邏輯。

---

## ⚠️ 實作狀態 (Implementation Status)

### 已完成 ✅
| 功能 | 狀態 | 備註 |
|------|------|------|
| 音訊輸入 (audio_input.py) | ✅ | 使用 sounddevice |
| VAD 語音偵測 (vad_segmenter.py) | ✅ | Silero VAD + 簡單能量偵測 |
| 停頓偵測 (FR-3) | ✅ | 調整為 1.5 秒 |
| ASR 轉寫 (asr_worker.py) | ✅ | 使用 Faster-Whisper base |
| 規則引擎 (rule_engine.py) | ✅ | contains/regex/exact, priority, cooldown |
| TTS 播放 (tts_worker.py) | ✅ | pyttsx3 (Windows) / espeak-ng (Linux) |
| 狀態機 (orchestrator.py) | ✅ | LISTENING → ASR → SPEAKING |
| ASR/TTS 互斥 | ✅ | speaking_event 同步 |
| 記憶體管理 (FR-6) | ✅ | 不落盤，僅 RAM |
| JSON 熱更新 | ✅ | mtime 偵測 |

### 待完成 ⚠️
| 功能 | 狀態 | 原因 |
|------|------|------|
| Qwen3-ASR 模型整合 | ⚠️ | 網路下載問題，改用 Faster-Whisper |
| Qwen3-TTS 模型整合 | ⚠️ | 延後，先用 pyttsx3/espeak-ng |
| 單元測試 | ⚠️ | 待補 |

### 技術變更
- **ASR**：從 Qwen3-ASR 改為 Faster-Whisper base (140MB)
- **TTS**：從 Qwen3-TTS 改為 pyttsx3 (Windows) / espeak-ng (Linux)
- **VAD 停頓**：從 1.0 秒調整為 1.5 秒

---

## 推理脈絡（給 AI Agent / 團隊對齊用）


## 推理脈絡（給 AI Agent / 團隊對齊用）

你要的是一個「本地（或可本地）語音代理」：**持續收音 → VAD/停頓判斷 → ASR 轉文字 → 關鍵字規則引擎匹配 JSON → 觸發 TTS 朗讀 JSON 內容**。其中最重要的互斥規則是：**一旦開始 TTS，必須暫停 ASR 的判斷與輸出，直到 TTS 播放完畢才恢復**。另外你要求「ASR 文字只存在記憶體，程式結束即釋放」，以及「ASR / TTS 多執行緒分工」與「停頓 1 秒才輸出文字」等需演算的細節。

技術參考上，Qwen3 的能力面向：Qwen3-ASR 支援多語言語音識別，而 Qwen3-TTS 支援串流（streaming）與低首包延遲（例如 12Hz tokenizer 可達極低 first-packet latency）[[4](https://arxiv.org/html/2601.15621v1?ref=hackernoon.com)]；而在工具鏈上，常見做法會引入 VAD（Voice Activity Detection, 語音活動偵測）來判定說話段落，像 Silero VAD 以 32ms chunks 進行串流偵測是典型組件之一[[1](https://github.com/ivan-digital/qwen3-asr-swift)]。這些都能支撐你想要的「停頓判斷 + 多執行緒」設計。

---

## PRD：Qwen3 ASR + 關鍵字觸發 TTS 語音回應（本地版）

### **1. 產品目標 (Goal)**
建立一個 Python 應用，能在麥克風持續收音下：

- 透過 **語音轉文字 ASR（Automatic Speech Recognition，自動語音識別）**將人聲轉為文字
- 把 ASR 文字在記憶體（RAM）中暫存（不落盤，程式結束即釋放）
- 依據 JSON 規則檔做 **關鍵字匹配 (Keyword Matching, 關鍵字比對)**  
- 匹配時觸發 **文字轉語音 TTS（Text-to-Speech，語音合成）**：朗讀 JSON 中指定的 key/value 或模板內容
- **TTS 播放期間必須暫停 ASR 事件輸出與關鍵字判斷**，播放完畢才恢復監聽流程

---

### **2. 範圍 (Scope)**

#### **2.1 In Scope**
- Windows 11 + Python（可選 GPU / CPU）本機執行
- **Faster-Whisper** 做離線 ASR（替代 Qwen3-ASR，網路問題）
- **pyttsx3/espeak-ng** 做本地 TTS（替代 Qwen3-TTS）
- 以 VAD/停頓判斷切段（停頓 1.5 秒視為一句結束）
- JSON 規則檔驅動：關鍵字、優先序、冷卻時間、輸出內容、TTS voice 設定
- 多執行緒：ASR 與 TTS 分工，主流程用狀態機協調

#### **2.2 Out of Scope（先不做）**
- 語者分離 (Diarization) / 多人聲辨識（可留擴充點；工具鏈中常見，但 PRD v1 不要求）[[1](https://github.com/ivan-digital/qwen3-asr-swift)]
- 長期記憶（磁碟、資料庫）與跨程序保存
- LLM 回答生成（本 PRD 僅做規則觸發式回應）

---

### **3. 使用者故事 (User Stories)**
- 作為使用者，我說出「天氣」後停頓 1 秒，系統應朗讀 JSON 設定的回覆內容，並在朗讀時不再誤觸發其他關鍵字。
- 作為開發者，我只要修改 JSON，就能新增/調整關鍵字與 TTS 輸出，不需改程式碼。
- 作為維運者，我希望能看到每次 ASR 轉寫結果、觸發規則、TTS 播放開始/結束時間，以及是否因冷卻時間而忽略觸發。

---

### **4. 功能需求 (Functional Requirements)**

#### **4.1 音訊輸入與切段**
**FR-1**：麥克風以固定 frame size 收音（例如 20ms/32ms/50ms 一個 frame）。  
**FR-2**：使用 **VAD（Voice Activity Detection，語音活動偵測）**判斷「有人在說話」或「靜音/背景」；可參考以 32ms chunks 的串流 VAD 方式[[1](https://github.com/ivan-digital/qwen3-asr-swift)]。  
**FR-3**：停頓規則（必要）  
- **靜音連續 >= 1.0 秒** ⇒ 視為一段 utterance 結束，觸發 ASR 對該段做轉寫輸出（你指定的規則）。  
- 若 utterance 時長 < `min_utterance_ms`（例如 300ms）則丟棄避免誤觸。  
- 若 utterance 時長 > `max_utterance_s`（例如 15s）則強制切段（避免無限累積記憶體）。

**FR-4**：ASR 用的音訊 buffer 只存在記憶體（ring buffer / list），ASR 完成後立即釋放（清空引用）。

#### **4.2 ASR 轉寫流程**
**FR-5**：當 utterance 結束事件發生（停頓 1 秒）時：
- 把該段音訊（PCM float32/int16）送入 ASR thread
- ASR thread 產生 `transcript`（字串）+（可選）語言識別結果

**FR-6**：ASR 文字記憶體策略（符合你要求）
- `transcript` 與 `history` 僅存於 RAM
- 不寫入檔案、不寫 DB
- 程式結束即由 OS 回收；程式運行中採「有限長度」保留：例如只留最近 N 句（`history_max_turns`，預設 20）
- 超出即丟棄最舊（FIFO）

**FR-7**：去抖動/合併策略（避免停頓切得太碎）
- 若兩次 utterance 間隔 < `merge_gap_s`（例如 0.4s）且前一句字數 < `min_chars_to_finalize`（例如 4）  
  ⇒ 合併成同一輪輸入（把文字串接，中間加空白或標點）

#### **4.3 規則引擎（JSON 關鍵字）**
**FR-8**：JSON 規則檔結構（建議）
- `rules[]`：每條規則包含：
  - `id`
  - `keywords`：可多關鍵字（支援 any / all）
  - `match_mode`：`contains` / `regex` / `exact`
  - `priority`：數字越小越優先
  - `cooldown_s`：同規則冷卻時間
  - `response`：
    - `type`: `speak_kv` / `speak_text`
    - `text_template`（可選）
    - `kv`: `{ "key": "...", "value": "..." }` 或多筆陣列
  - `tts`：speaker/language/style（視 Qwen3-TTS model 能力決定）[[4](https://arxiv.org/html/2601.15621v1?ref=hackernoon.com)]

**FR-9**：匹配流程
- 每次取得 `transcript_final` 後，進入規則引擎：
  - 收集所有命中規則
  - 過濾冷卻中規則
  - 依 `priority` 選第一個（或允許多個但需排隊，見 FR-12）
- 命中後生成 `tts_job`（要朗讀的文字）

**FR-10**：冷卻時間計算
- 每條 rule 記錄 `last_triggered_at`（只存在 RAM）
- 若 `now - last_triggered_at < cooldown_s` ⇒ 忽略本次命中（log 必須記錄 ignored）

#### **4.4 TTS 播放與 ASR 暫停（關鍵互斥規則）**
**FR-11**：狀態機（State Machine, 狀態機制）
- `LISTENING`：收音/VAD/切段啟用
- `ASR_PROCESSING`：ASR thread 正在推理（LISTENING 可繼續收音或暫停收音，見 NFR-2）
- `SPEAKING`：TTS thread 播放中（此時 **必須暫停** 觸發 ASR 文字輸出與規則判斷；避免自我回授）
- `STOPPED/ERROR`

**FR-12**：TTS 排程
- 命中規則後，將 `tts_job` 丟進 `tts_queue`
- 若目前在 `SPEAKING`，新 job 先排隊（或依需求丟棄/覆蓋；預設排隊）
- TTS thread 每次只處理一個 job，播放完才取下一個

**FR-13**：ASR 暫停定義（務必明確）
- 在 `SPEAKING` 期間：
  - VAD 可仍然收音（可選），但**不得 finalize utterance**、不得送 ASR 推理、不得觸發規則  
  - 建議直接「丟棄」這段時間的音訊 frame，避免把自己播出的聲音收進去（最簡 KISS）
- `SPEAKING` 結束後：
  - 清空 audio buffer（避免殘留 TTS 尾音）
  - 延遲 `resume_grace_s`（例如 0.2s）再恢復 LISTENING

> Qwen3-TTS 支援串流輸出與低延遲特性，意味著播放開始可能非常快，狀態切換必須在提交播放前就完成，以免短暫 race condition 造成 ASR 誤收[[4](https://arxiv.org/html/2601.15621v1?ref=hackernoon.com)]。

---

### **5. 非功能需求 (Non-Functional Requirements)**

#### **5.1 效能與延遲**
- 目標：從停頓 1 秒到 ASR 結果輸出 < 1.5s（視硬體）
- TTS 需可串流或至少快速出聲；Qwen3-TTS 設計為串流，具有極低 first-packet latency 的能力[[4](https://arxiv.org/html/2601.15621v1?ref=hackernoon.com)]

#### **5.2 可維護性（KISS）**
- 模組拆分：
  - `audio_input`（麥克風、frame 事件）
  - `vad_segmenter`（VAD + 停頓/切段演算法）
  - `asr_worker`（Qwen3-ASR）
  - `rule_engine`（JSON）
  - `tts_worker`（Qwen3-TTS）
  - `orchestrator`（狀態機、thread 協調）

#### **5.3 隱私與資料生命週期**
- 預設不落盤（音訊、文字皆不存檔）
- Debug 模式才允許把 wav / transcript 輸出到檔案（需明確開關）

---

### **6. 演算法/判斷細節（你要求逐條補齊）**

#### **6.1 停頓 1 秒 finalize（核心）**
定義：
- 每個 frame 計算能量 $E = mean(|x|)$ 或 RMS
- VAD 給出 `speech_prob` 或 `is_speech`
- 連續 `is_speech=False` 的累積時間 `silence_s`

流程：
- `is_speech=True`：把 frame append 到 `current_utterance_buffer`；`silence_s=0`
- `is_speech=False`：
  - 若 `current_utterance_buffer` 非空：`silence_s += frame_duration`
  - 若 `silence_s >= 1.0`：finalize utterance（送 ASR）

邊界：
- utterance 時長太短丟棄
- finalize 後 `current_utterance_buffer.clear()` 釋放引用（記憶體可回收）

#### **6.2 ASR/TTS 多執行緒同步**
- 使用 `threading.Event` / `Condition`
- `speaking_event`：
  - TTS 開始播放前 `speaking_event.set()`
  - 播放完 `speaking_event.clear()`
- Orchestrator 在 `speaking_event.is_set()` 時：
  - audio frames 直接丟棄
  - 不累積 utterance buffer

#### **6.3 避免 TTS 自我回授**
最低限度策略（KISS）：
- `SPEAKING` 期間丟棄所有麥克風輸入（不做回聲消除）
進階（可列為 v2）：
- AEC（Acoustic Echo Cancellation, 聲學回聲消除）
- 或把 TTS 輸出作為參考訊號做回授消除（較複雜，v1 不做）

#### **6.4 規則命中衝突處理**
- 若同一句話同時命中多規則：
  - 先依 `priority` 排序
  - 若 `allow_multiple=false` ⇒ 只播第一條
  - 若 `allow_multiple=true` ⇒ 依序排入 `tts_queue`（需設 `max_queue_size` 防爆）

---

### **7. 介面與輸出規格**

#### **7.1 Console Log（必備）**
- `[VAD] start_speech / end_speech`
- `[ASR] transcript_final="..."`
- `[RULE] matched=rule_id, keyword=..., cooldown_ignored=...`
- `[TTS] start job_id=... text="..."`
- `[TTS] end job_id=... duration_ms=...`

#### **7.2 設定檔（JSON）**
- 支援熱更新（可選）：每次 finalize 後檢查檔案 mtime，變更則 reload（RAM 中替換）

---

### **8. 驗收標準 (Acceptance Criteria)**

1. ✅ **停頓 1.5 秒才輸出文字**：使用者說一句話，中間短停頓不會立刻輸出；靜音滿 1.5 秒才 finalize 並印出轉寫。
2. ✅ **命中規則必播 TTS**：說出 JSON 設定關鍵字後，會朗讀指定 key/value 或模板文字。
3. ✅ **TTS 期間 ASR 暫停**：播放時不會再輸出 ASR 轉寫，也不會再觸發任何規則；播放結束後恢復。
4. ✅ **記憶體釋放**：不產生任何自動落盤檔案；停止程式後不殘留文字歷史（僅 RAM）。
5. ✅ **多執行緒穩定**：連續觸發多次關鍵字，無 deadlock、無無限 queue 成長（queue 有上限）。
6. ⚠️ **ASR 語音辨識**：使用 Faster-Whisper 代替 Qwen3-ASR（網路問題）

---

### **9. 實作備註（給 AI Agent 的任務拆解）**
- 先做最小可用版本（MVP）：
  - VAD/能量門檻 + 停頓 1 秒 finalize
  - ASR thread 產出 transcript
  - contains 規則命中 + TTS job
  - speaking_event 確保互斥
- 再擴充：
  - regex match、優先序、冷卻時間
  - history 合併策略
  - queue 上限與丟棄策略

---

如果你願意，我可以在這份 PRD 基礎上再補兩份附錄讓 AI Agent 更容易落地：  
1) **JSON Schema（含欄位型別/必填/預設值）**  
2) **狀態機轉移表（State Transition Table）+ 競態條件（Race Conditions）清單與防護策略**。



1. [ivan-digital/qwen3-asr-swift: AI speech toolkit for Apple ...](https://github.com/ivan-digital/qwen3-asr-swift)
2. [可以在你的筆記型電腦上本地測試Qwen3-TTS 和 ...](https://www.reddit.com/r/artificial/comments/1qs6ibp/i_built_a_way_to_test_qwen3tts_and_qwen3asr/?tl=zh-hant)
3. [Arxiv今日论文| 2026-02-27](http://lonepatient.top/2026/02/27/arxiv_papers_2026-02-27.html)
4. [Qwen3-TTS Technical Report](https://arxiv.org/html/2601.15621v1?ref=hackernoon.com)
5. [qwen3-asr speech recognition models support 52 languages](https://www.facebook.com/groups/AIUGM/posts/4358162187798076/)
6. [Qwen3 TTS 聲音設計和多角色對話: r/comfyui](https://www.reddit.com/r/comfyui/comments/1qn2wfm/qwen3_tts_voice_design_and_multicharacter_dialogue/?tl=zh-hant)
7. [Qwen3-TTS: The Complete 2026 Guide to Open-Source ...](https://medium.com/@zh.milo/qwen3-tts-the-complete-2026-guide-to-open-source-voice-cloning-and-ai-speech-generation-1a2efca05cd6)
8. [我用代码搞了个Qwen3-TTS 和Qwen3-ASR 的本地音频推理 ...](https://www.reddit.com/r/LocalLLaMA/comments/1qqhoyo/i_vibe_coded_a_local_audio_inference_engine_for/?tl=zh-hans)
9. [AI核心知识33——大语言模型之ASR（简洁且通俗易懂版）](https://adg.csdn.net/695229295b9f5f31781b2a70.html)
10. [Qwen3-TTS:2026年开源语音克隆与AI语音生成完全指南原创](https://blog.csdn.net/daiziguizhong/article/details/157290355)
11. [Qwen Releases Open-Source TTS Models for Commercial ...](https://www.linkedin.com/posts/andrewanokhin_qwen-qwen3-tts-activity-7420149619701071872-FFBS)
12. [這集陪你一起讀最新研究《Memory in the Age of AI Agents ...](https://www.facebook.com/seokpn/videos/ai-%E4%B8%8D%E5%86%8D%E5%81%A5%E5%BF%98%E4%B8%80%E6%AC%A1%E6%90%9E%E6%87%82%E6%9C%89%E8%A8%98%E6%86%B6%E7%9A%84-ai-%E4%BB%A3%E7%90%86%E4%BA%BA%E6%80%8E%E9%BA%BC%E6%94%B9%E5%AF%AB%E8%A1%8C%E9%8A%B7%E7%8E%A9%E6%B3%95/897241152744528/)
13. [Qwen3-TTS Family is Now Open Sourced: Voice Design, ...](https://qwen.ai/blog?id=qwen3tts-0115)
14. [阶跃星辰开源多模态模型Step3‑VL‑10B， ...](https://blog.csdn.net/agora_cloud/article/details/157251581)
15. [Qwen have open-sourced the full family of Qwen3-TTS](https://www.reddit.com/r/LocalLLaMA/comments/1qjul5t/qwen_have_opensourced_the_full_family_of_qwen3tts/)
16. [halsay/ASR-TTS-paper-daily: Update ASR paper everyday](https://github.com/halsay/ASR-TTS-paper-daily)
17. [从语音识别到智能助手：Voice Agent 的技术进化与交互变革 ...](https://zhuanlan.zhihu.com/p/1927014763390560095)
18. [语音Agent 的全面思考--万字长文](https://zhuanlan.zhihu.com/p/1952022345238709986)
19. [Qwen TTS，聲音設計師的穩定性: r/TextToSpeech](https://www.reddit.com/r/TextToSpeech/comments/1rg9yqs/qwen_tts_voice_designer_consistency/?tl=zh-hant)
20. [每日AI简报- 野湃AI](https://www.yepaisz.com/260.html)
21. [Qwen3-TTS全面开源：支持超低延迟流式合成的多语言语音 ...](https://zhuanlan.zhihu.com/p/1999111858230150545)
22. [自然语言处理2026_3_3](https://www.arxivdaily.com/thread/77258)
23. [some-stars/README.md at main](https://github.com/rcy1314/some-stars/blob/main/README.md)
24. [自然语言处理2026_2_27](http://43.153.52.135/thread/77089)
25. [51c大模型~合集114](https://blog.51cto.com/whaosoft143/13767557)
26. [Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)
27. [RTC 文档](https://bce-cdn.bj.bcebos.com/p3m/pdf/bce-doc/online/RTC/RTC.pdf?timeStamp=1750377600081)
28. [Best open source speech-to-text (STT) model in 2026 (with ...](https://northflank.com/blog/best-open-source-speech-to-text-stt-model-in-2026-benchmarks)
29. [Voice AI Agent 知识库：打造你自己的语音智能体！ 原创](https://blog.csdn.net/agora_cloud/article/details/149612874)
30. [解锁AI 语音交互的「灵魂秘籍」丨Voice Agent 学习笔记](https://blog.csdn.net/agora_cloud/article/details/148931365)
31. [Qwen3-TTS语音设计世界入门指南：'跳跃精准'滑块对语音 ...](https://blog.csdn.net/weixin_30653091/article/details/158312135)

