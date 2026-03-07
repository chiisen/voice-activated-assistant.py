import torch
from qwen_tts import Qwen3TTSModel
import inspect

model_path = "models/Qwen3-TTS-12Hz-0.6B-CustomVoice" # This is from orchestrator.py
try:
    print(f"Inspecting Qwen3TTSModel methods...")
    # List methods
    methods = [m for m in dir(Qwen3TTSModel) if not m.startswith('__')]
    print(f"Methods: {methods}")
    
    # Check generate_custom_voice signature
    if hasattr(Qwen3TTSModel, 'generate_custom_voice'):
        sig = inspect.signature(Qwen3TTSModel.generate_custom_voice)
        print(f"generate_custom_voice signature: {sig}")
    
    # Check if there's any 'stream' related methods
    stream_methods = [m for m in methods if 'stream' in m.lower()]
    print(f"Stream-related methods: {stream_methods}")

except Exception as e:
    print(f"Error: {e}")
