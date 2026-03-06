#!/usr/bin/env python3
# ==============================================================================
# 檔案：rule_engine.py
# 功能：Rule Engine 規則引擎 - 基於 JSON 的關鍵字匹配與回應生成
# 描述：
#     此模組是語音助理的「指令解析器」，負責：
#     1. 從 JSON 檔案載入規則定義 (關鍵字、匹配模式、回應內容)
#     2. 根據 ASR 識別出的文字匹配對應規則
#     3. 生成回應內容 (TTS Job)
#     4. 支援熱重載 (Hot Reload)，修改規則檔案無需重啟程式
#
# 設計概念：
#     - 規則優先級：支援 priority 欄位，高優先級規則優先匹配
#     - 冷卻機制：支援 cooldown_s 防止規則短時間內重複觸發
#     - 匹配模式：支援 contains (包含)、exact (完全匹配)、regex (正則表達式)
#     - 歷史記錄：維護最近的辨識歷史供調試使用
# ==============================================================================

import re
import json
import time
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from pathlib import Path


# ==============================================================================
# 規則資料類別 (Rule)
# ==============================================================================
@dataclass
class Rule:
    """
    規則資料類別
    
    說明：
        封裝單一規則的所有屬性，包含關鍵字、匹配模式、優先級、冷卻時間等。
    
    屬性：
        id: str，規則的唯一識別碼
            - 用於日誌和追蹤
            - 建議使用有意義的名稱，如 "greeting", "stop_music"
        keywords: List[str]，關鍵字列表
            - 用於匹配使用者輸入
            - 可包含多個關鍵字，任一匹配即觸發
        match_mode: str，匹配模式
            - "contains": 關鍵字包含在文字中 (預設)
            - "exact": 完全匹配
            - "regex": 正則表達式匹配
        priority: int，優先級
            - 數字越小越優先
            - 預設值：100
        cooldown_s: float，冷卻時間 (秒)
            - 規則觸發後需要等待的時間
            - 預設值：0.0 (無冷卻)
        response_type: str，回應類型
            - "speak_text": 朗讀指定文字
            - "speech_kv": 朗讀鍵值對
        text_template: Optional[str]，回應文字範本
            - 用於 "speak_text" 類型
        kv: Optional[Dict[str, str]]，鍵值對
            - 用於 "speak_kv" 類型
        tts_voice: Optional[str]，TTS 聲音
            - 指定 TTS 使用的聲音
        tts_language: Optional[str]，TTS 語言
            - 指定 TTS 使用的語言
    
    JSON 格式範例：
        {
            "id": "greeting",
            "keywords": ["你好", "嗨", "哈囉"],
            "match_mode": "contains",
            "priority": 10,
            "cooldown_s": 2.0,
            "response": {
                "type": "speak_text",
                "text_template": "你好！請問有什麼可以幫您？"
            },
            "tts": {
                "voice": "zh",
                "language": "zh-TW"
            }
        }
    """
    id: str
    keywords: List[str]
    match_mode: str = "contains"
    priority: int = 100
    cooldown_s: float = 0.0
    response_type: str = "speak_text"
    text_template: Optional[str] = None
    kv: Optional[Dict[str, str]] = None
    tts_voice: Optional[str] = None
    tts_language: Optional[str] = None


# ==============================================================================
# TTS 任務資料類別 (TTSJob)
# ==============================================================================
@dataclass
class TTSJob:
    """
    TTS 任務資料類別
    
    說明：
        封裝要送給 TTS Worker 的任務資料。
        由 RuleEngine 根據匹配的規則生成。
    
    屬性：
        rule_id: str，對應的規則 ID
            - 用於日誌和追蹤
        text: str，要朗讀的文字
            - 已經過文字生成處理
        voice: Optional[str]，指定的 TTS 聲音
            - 若為 None，使用預設聲音
        language: Optional[str]，指定的 TTS 語言
            - 若為 None，使用預設語言
    
    使用範例：
        job = TTSJob(
            rule_id="greeting",
            text="你好！請問有什麼可以幫您？",
            voice="zh",
            language="zh-TW"
        )
    """
    rule_id: str
    text: str
    voice: Optional[str] = None
    language: Optional[str] = None


# ==============================================================================
# RuleEngine 類別
# ==============================================================================
class RuleEngine:
    """
    規則引擎
    
    說明：
        負責管理所有規則的生命週期和匹配邏輯。
        核心功能：
        1. 從 JSON 檔案載入規則
        2. 根據輸入文字匹配對應規則
        3. 支援熱重載 (修改 JSON 無需重啟)
        4. 維護歷史記錄和冷卻狀態
    
    設計重點：
        - 非同步安全：match() 可從多個執行緒呼叫
        - 效率優化：規則按優先級排序，匹配到高優先級即返回
        - 靈活性：支援多種匹配模式和回應類型
    
    使用流程：
        1. 建立 RuleEngine 實例
        2. 呼叫 load_rules() 載入規則檔
        3. 每次 ASR 有結果時呼叫 match()
        4. 根據回傳的 TTSJob 進行 TTS 輸出
    """
    
    def __init__(self, rules_path: Optional[str] = None):
        """
        建構函式 - 建立 RuleEngine 實例
        
        說明：
            初始化規則引擎，設定規則檔路徑。
        
        參數：
            rules_path: Optional[str]，規則 JSON 檔案路徑
                - 可為相對路徑或絕對路徑
                - 若為 None，稍後可在 load_rules() 指定
        
        內部變數：
            self._rules: List[Rule]，已載入的規則列表
            self._last_triggered: Dict[str, float]，每個規則的最後觸發時間
            self._history: List[str]，辨識歷史 (用於調試)
            self._history_max: int，歷史記錄最大數量
            self._rules_mtime: Optional[float]，規則檔的最后修改時間
        """
        self.rules_path = rules_path
        self._rules: List[Rule] = []
        self._last_triggered: Dict[str, float] = {}
        self._history: List[str] = []
        self._history_max = 20
        self._rules_mtime: Optional[float] = None

    def load_rules(self, path: Optional[str] = None):
        """
        載入規則檔
        
        說明：
            從 JSON 檔案讀取規則定義，並轉換為 Rule 物件列表。
            載入後會按照 priority 欄位排序。
        
        參數：
            path: Optional[str]，規則檔路徑
                - 若未指定，使用建構時的 self.rules_path
        
        流程：
            1. 檢查檔案是否存在
            2. 讀取並解析 JSON
            3. 轉換為 Rule 物件列表
            4. 按優先級排序
            5. 記錄檔案修改時間 (用於熱重載)
        
        JSON 格式要求：
            {
                "rules": [
                    { ...rule1... },
                    { ...rule2... }
                ]
            }
        """
        path = path or self.rules_path
        if not path:
            return

        file_path = Path(path)
        if not file_path.exists():
            return

        # 記錄檔案修改時間
        self._rules_mtime = file_path.stat().st_mtime

        # 讀取並解析 JSON
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 轉換為 Rule 物件
        self._rules = []
        for item in data.get("rules", []):
            rule = Rule(
                id=item.get("id", ""),
                keywords=item.get("keywords", []),
                match_mode=item.get("match_mode", "contains"),
                priority=item.get("priority", 100),
                cooldown_s=item.get("cooldown_s", 0.0),
                response_type=item.get("response", {}).get("type", "speak_text"),
                text_template=item.get("response", {}).get("text_template"),
                kv=item.get("response", {}).get("kv"),
                tts_voice=item.get("tts", {}).get("voice"),
                tts_language=item.get("tts", {}).get("language"),
            )
            self._rules.append(rule)

        # 按優先級排序 (數字越小越優先)
        self._rules.sort(key=lambda r: r.priority)

    def check_hot_reload(self):
        """
        檢查是否需要熱重載
        
        說明：
            檢查規則檔是否被修改過，若是則自動重新載入。
            這使得使用者可以在程式運行時修改規則而無需重啟。
        
        參數：
            無
        
        回傳：
            bool: 
                - True: 已執行熱重載
                - False: 無需重載
        """
        if not self.rules_path:
            return False

        path = Path(self.rules_path)
        if not path.exists():
            return False

        # 取得目前修改時間
        current_mtime = path.stat().st_mtime

        # 若有更新，則重新載入
        if self._rules_mtime and current_mtime > self._rules_mtime:
            self.load_rules()
            return True

        return False

    def match(self, transcript: str) -> Optional[TTSJob]:
        """
        匹配規則
        
        說明：
            根據輸入的文字匹配對應的規則。
            匹配流程：
            1. 先檢查熱重載
            2. 將文字加入歷史記錄
            3. 遍歷所有規則，檢查關鍵字匹配
            4. 檢查冷卻時間
            5. 匹配成功則生成 TTSJob
        
        參數：
            transcript: str，ASR 識別出的文字
        
        回傳：
            Optional[TTSJob]: 
                - 匹配的 TTSJob (包含要朗讀的文字)
                - 若無匹配，回傳 None
        
        匹配優先級：
            - 按 self._rules 的順序 (已按 priority 排序)
            - 第一個符合條件的規則獲勝
        """
        # Step 1: 檢查熱重載
        self.check_hot_reload()

        # Step 2: 加入歷史記錄
        self._history.append(transcript)
        if len(self._history) > self._history_max:
            self._history.pop(0)

        # 取得目前時間
        current_time = time.time()
        matched_rules = []

        # Step 3: 遍歷規則，檢查匹配
        for rule in self._rules:
            # 檢查關鍵字
            if not self._check_keywords(transcript, rule):
                continue

            # Step 4: 檢查冷卻時間
            if rule.cooldown_s > 0:
                last_time = self._last_triggered.get(rule.id, 0)
                if current_time - last_time < rule.cooldown_s:
                    # 仍在冷卻中，跳過此規則
                    continue

            matched_rules.append(rule)

        # Step 5: 若無匹配，回傳 None
        if not matched_rules:
            return None

        # 取得第一個匹配的規則
        rule = matched_rules[0]
        self._last_triggered[rule.id] = current_time

        # 生成回應文字
        text = self._generate_response(rule, transcript)

        # 建立並回傳 TTSJob
        return TTSJob(
            rule_id=rule.id, 
            text=text, 
            voice=rule.tts_voice, 
            language=rule.tts_language
        )

    def _check_keywords(self, transcript: str, rule: Rule) -> bool:
        """
        檢查關鍵字是否匹配
        
        說明：
            根據規則的 match_mode 欄位，選擇合適的匹配方式：
            - "contains": 檢查關鍵字是否包含在文字中
            - "exact": 檢查是否完全相等
            - "regex": 使用正則表達式匹配
        
        參數：
            transcript: str，要檢查的文字
            rule: Rule，規則物件
        
        回傳：
            bool: 
                - True: 匹配成功
                - False: 不匹配
        """
        # 轉小寫進行大小寫不敏感匹配
        transcript_lower = transcript.lower()

        # 遍歷所有關鍵字
        for keyword in rule.keywords:
            keyword_lower = keyword.lower()

            if rule.match_mode == "contains":
                # 包含模式：關鍵字出現在文字中即可
                if keyword_lower in transcript_lower:
                    return True

            elif rule.match_mode == "exact":
                # 完全匹配模式：必須完全相等
                if keyword_lower == transcript_lower:
                    return True

            elif rule.match_mode == "regex":
                # 正則表達式模式
                try:
                    if re.search(keyword, transcript, re.IGNORECASE):
                        return True
                except re.error:
                    # 正則表達式語法錯誤，忽略此關鍵字
                    pass

        return False

    def _generate_response(self, rule: Rule, transcript: str) -> str:
        """
        生成回應文字
        
        說明：
            根據規則的回應類型生成要朗讀的文字：
            - "speak_text": 回傳 text_template
            - "speak_kv": 將 kv 字典轉換為文字
            - 其他：回傳原始輸入
        
        參數：
            rule: Rule，匹配的規則
            transcript: str，原始輸入文字
        
        回傳：
            str，要朗讀的文字
        """
        if rule.response_type == "speak_text" and rule.text_template:
            return rule.text_template

        if rule.response_type == "speak_kv" and rule.kv:
            parts = []
            for key, value in rule.kv.items():
                parts.append(f"{key}: {value}")
            return ", ".join(parts)

        # 預設：回傳原始輸入
        return transcript

    def get_history(self) -> List[str]:
        """
        取得辨識歷史
        
        說明：
            回傳最近的辨識記錄列表，用於調試和分析。
        
        參數：
            無
        
        回傳：
            List[str]: 文字列表，最新的在最後面
        """
        return self._history.copy()
