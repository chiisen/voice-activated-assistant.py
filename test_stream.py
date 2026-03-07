import torch
from qwen_tts import Qwen3TTSModel
import time

model_path = "models/Qwen3-TTS-12Hz-0.6B-CustomVoice"
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.bfloat16 if "cuda" in device else torch.float32

print(f"Loading model from {model_path}...")
engine = Qwen3TTSModel.from_pretrained(
    model_path, 
    device_map=device,
    dtype=torch_dtype,
    trust_remote_code=True
)

from transformers import BaseStreamer

class TokenStreamer(BaseStreamer):
    def __init__(self):
        self.tokens = []
    def put(self, value):
        print(f"Streamer received: {value.shape}")
        self.tokens.append(value)
    def end(self):
        print("Streamer end")

streamer = TokenStreamer()

text = "你好，我是語音助理。"
speaker = "vivian"

print("Testing generate_custom_voice with streamer (hypothetical)...")
# Since generate_custom_voice is a wrapper, we need to see if it passes kwargs to generate
try:
    # We'll try to call the internal generate if needed
    # But let's see if we can just pass it to generate_custom_voice
    wavs, sr = engine.generate_custom_voice(
        text, 
        speaker=speaker,
        streamer=streamer
    )
    print("Done")
except Exception as e:
    print(f"Error: {e}")
