import torch
from qwen_tts import Qwen3TTSModel
import inspect

try:
    print(f"Inspecting Qwen3TTSModel...")
    # List methods
    for m in dir(Qwen3TTSModel):
        if not m.startswith('__'):
            print(f"Method: {m}")
    
    # Check if from_pretrained returns an object with more methods
    model_path = "models/Qwen3-TTS-12Hz-0.6B-CustomVoice"
    # Wait, let's just inspect the class methods for now without loading the whole model (which is slow)
    # But some methods might be attached at runtime.
    
    # Let's see if there's any 'stream' or 'generator' related stuff in the module
    import qwen_tts
    print(f"Contents of qwen_tts: {dir(qwen_tts)}")

except Exception as e:
    print(f"Error: {e}")
