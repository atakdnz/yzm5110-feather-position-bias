# Pre-fetch model and datasets into the HF cache on the Colab VM.
import os, time
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
t = time.time()
from huggingface_hub import snapshot_download
p = snapshot_download("llava-hf/llava-1.5-7b-hf", revision="b234b804b114d9e37bb655e11cbbb5f5e971b7a9", allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt"])
print("model at", p, "rev", os.path.basename(p), f"{time.time()-t:.0f}s", flush=True)
from datasets import load_dataset
a = load_dataset("HuggingFaceM4/A-OKVQA", split="validation")
print("aokvqa", len(a), a.column_names, f"{time.time()-t:.0f}s", flush=True)
m = load_dataset("lmms-lab/MME", split="test")
print("mme", len(m), m.column_names, f"{time.time()-t:.0f}s", flush=True)
print("DONE", flush=True)
