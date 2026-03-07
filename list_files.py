import os

base_path = r"d:\github\chiisen\voice-activated-assistant.py\.venv\Lib\site-packages\qwen_tts"
for root, dirs, files in os.walk(base_path):
    for file in files:
        print(os.path.join(root, file))
