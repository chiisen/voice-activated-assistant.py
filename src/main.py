#!/usr/bin/env python3
# ==============================================================================
# 檔案：main.py
# 功能：Voice Activated Assistant 語音助理的主入口點
# 描述：
#     此程式作為語音助理的起動點，負責解析命令列參數、初始化日誌系統、
#     建立協調器 (Orchestrator) 實例，並處理系統訊號 (如 Ctrl+C) 以優雅地
#     關閉應用程式。
# ==============================================================================

import sys
import signal
import argparse
from pathlib import Path

# 將 src 目錄加入 Python 模組搜尋路徑，以便匯入相對模組
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.orchestrator import Orchestrator, OrchestratorConfig
from src.logging_config import setup_logging
from src.audio_input import AudioInput, AudioConfig


def parse_args():
    """
    解析命令列參數
    
    說明：
        此函式使用 argparse 模組解析命令列參數，讓使用者可以自訂程式行為，
        包括設定配置檔路徑、除錯模式、音訊裝置選擇等。
    
    參數：
        無
    
    回傳：
        argparse.Namespace: 包含所有解析後的參數物件
            - config: 組態檔路徑 (預設 "config/config.yaml")
            - rules: 規則 JSON 檔案路徑 (預設 "config/rules.json")
            - debug: 是否啟用除錯模式
            - list_devices: 是否列出可用音訊裝置
            - device: 指定音訊裝置索引
            - mock_mode: 是否使用模擬模式 (無需麥克風)
            - test: 測試用的語句 (自動啟用 mock_mode)
    
    使用範例：
        python main.py --rules config/rules.json --device 0 --debug
    """
    parser = argparse.ArgumentParser(description="Voice Activated Assistant")
    
    # -------------------------------------------------------------------------
    # 設定檔相關參數
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/config.yaml", 
        help="Path to config file"
    )
    parser.add_argument(
        "--rules", 
        type=str, 
        default="config/rules.json", 
        help="Path to rules JSON file"
    )
    
    # -------------------------------------------------------------------------
    # 偵錯與測試相關參數
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--debug", 
        action="store_true", 
        help="Enable debug mode"
    )
    parser.add_argument(
        "--list-devices", 
        action="store_true", 
        help="List available audio devices"
    )
    parser.add_argument(
        "--device", 
        type=int, 
        default=None, 
        help="Audio device index"
    )
    parser.add_argument(
        "--mock-mode", 
        action="store_true", 
        help="Run without microphone for testing"
    )
    parser.add_argument(
        "--test", 
        type=str, 
        help="Test with a specific phrase (implies --mock-mode)"
    )
    
    return parser.parse_args()


def main():
    """
    主程式進入點
    
    說明：
        此函式是程式的執行起點，負責以下任務：
        1. 解析命令列參數
        2. 若有 --list-devices 參數，列出可用音訊裝置後結束
        3. 初始化日誌系統
        4. 建立 Orchestrator 協調器實例
        5. 註冊系統訊號處理器 (SIGINT, SIGTERM) 以支援優雅關機
        6. 啟動協調器並進入主迴圈
    
    參數：
        無
    
    回傳：
        無 (若失敗則以 sys.exit(1) 結束程式)
    
    流程說明：
        1. 解析命令列參數
        2. 檢查是否需要列出音訊裝置
        3. 設定日誌系統 (根據 debug 參數決定日誌等級)
        4. 建立 OrchestratorConfig 組態物件
        5. 建立 Orchestrator 協調器實例
        6. 設定 Ctrl+C (SIGINT) 和終端 (SIGTERM) 訊號處理
        7. 顯示啟動訊息
        8. 呼叫 orchestrator.start() 起動語音助理
        9. 若有 --test 參數，模擬一段語句
        10. 進入無限迴圈等待訊號 (signal.pause())
    """
    # Step 1: 解析命令列參數
    args = parse_args()

    # Step 2: 檢查是否需要列出可用音訊裝置
    if args.list_devices:
        audio = AudioInput()
        audio.list_devices()
        return

    # Step 3: 初始化日誌系統
    # 根據 debug 參數決定是否顯示詳細除錯資訊
    logger = setup_logging(debug=args.debug)

    # Step 4: 建立 Orchestrator 協調器組態
    # 說明：將命令列參數轉換為 OrchestratorConfig 資料類別
    config = OrchestratorConfig(
        rules_path=args.rules,          # 規則檔案路徑
        debug=args.debug,               # 除錯模式
        audio_device=args.device,       # 音訊裝置索引
        # 若有 --test 參數或 --mock-mode，則啟用模擬模式
        mock_mode=args.mock_mode or args.test is not None,
    )

    # Step 5: 建立 Orchestrator 協調器實例
    # 說明：協調器負責管理所有子模組 (音訊輸入、VAD、ASR、TTS、規則引擎)
    orchestrator = Orchestrator(config)

    # Step 6: 註冊系統訊號處理器
    # 說明：當使用者按下 Ctrl+C 或收到系統終止訊號時，優雅地關閉程式
    def signal_handler(sig, frame):
        """
        訊號處理回調函式
        
        說明：
            當收到 SIGINT (Ctrl+C) 或 SIGTERM 訊號時，
            先呼叫 orchestrator.stop() 停止所有執行緒和工作，
            然後以正常結束碼離開程式。
        
        參數：
            sig: 訊號編號
            frame: 目前的堆疊框架 (未使用)
        """
        print("\nShutting down...")
        orchestrator.stop()
        sys.exit(0)

    # 註冊訊號處理器
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C 中斷訊號
    signal.signal(signal.SIGTERM, signal_handler)  # 程式終止訊號

    # Step 7: 顯示起動訊息
    print("Starting Voice Activated Assistant...")
    print(f"Rules file: {args.rules}")
    if config.mock_mode:
        print("Mode: MOCK (no audio input)")
    elif args.device is not None:
        print(f"Using audio device: {args.device}")
    print("Press Ctrl+C to stop")

    # Step 8: 啟動 Orchestrator 協調器
    success = orchestrator.start()

    # Step 9: 檢查起動是否成功
    if not success:
        print("\nFailed to start audio. Use --list-devices to see available devices")
        print("Example: python src/main.py --rules config/rules.json --device 0")
        print("Or use --mock-mode for testing without microphone")
        sys.exit(1)

    # Step 10: 若有 --test 參數，模擬一段語句進行測試
    # 說明：在模擬模式下，直接餵入指定文字繞過語音辨識，測試規則匹配和 TTS
    if args.test:
        print(f"\n[Test] Simulating: '{args.test}'")
        orchestrator.simulate_utterance(args.test)

    # Step 11: 進入無限迴圈，等待訊號
    # 說明：signal.pause() 會阻塞直到收到訊號，這樣可以保持程式運作
    while True:
        try:
            signal.pause()
        except AttributeError:
            # Windows 系統不支援 signal.pause()，fallback 到 time.sleep
            import time
            time.sleep(1)


# 標準 Python 慣例：當此檔案被直接執行時才起動 main()
if __name__ == "__main__":
    main()
