import os, sys, traceback
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY","1")
from huggingface_hub import snapshot_download
REPOS = [
    ("depth-anything/Depth-Anything-V2-Small-hf", ["*.json","*.safetensors","*.txt"]),
    ("nvidia/segformer-b0-finetuned-ade-512-512", ["*.json","*.safetensors","*.bin","*.txt"]),
]
for rid, allow in REPOS:
    for attempt in range(4):
        try:
            p = snapshot_download(rid, allow_patterns=allow)
            print("OK", rid, p, flush=True); break
        except Exception as e:
            print("retry", attempt, rid, type(e).__name__, str(e)[:150], flush=True)
    else:
        print("FAILED", rid, flush=True)
# torchvision backbone for VPR / traversability shared trunk
try:
    import torchvision
    m = torchvision.models.mobilenet_v3_small(weights=torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    print("OK mobilenet_v3_small", flush=True)
except Exception as e:
    print("FAILED mobilenet", type(e).__name__, str(e)[:200], flush=True)
print("DONE")
