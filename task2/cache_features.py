"""C11: download the selected COCO images and cache SigLIP's 196 final patch features in FP16.

python -m task2.cache_features --mode {smoke,full}
last_hidden_state of SiglipVisionModel already has post_layernorm applied (transformers modeling_siglip,
SiglipVisionTransformer.forward), so it is stored as is: the final layer norm is applied exactly once.
Preprocessing is the processor's deterministic resize to 224 x 224 (no crop or flip); 14 x 14 row-major order.
"""
import argparse
import io
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from PIL import Image

from common.results import ROOT, read_json, sha256_file, write_json

VISION = "google/siglip-base-patch16-224"
OUT = ROOT / "data/cache/task2_features"


def fetch(item):
    iid, url = item
    path = ROOT / "data/cache/task2_images" / f"{iid}.jpg"
    if not path.exists():
        for attempt in range(4):
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                data = urllib.request.urlopen(url, timeout=60).read()
                Image.open(io.BytesIO(data)).verify()
                path.write_bytes(data)
                break
            except Exception:
                time.sleep(2 ** attempt)
    return iid, path


def encode(model, proc, paths, device, batch=64) -> dict[str, torch.Tensor]:
    feats = {}
    for i in range(0, len(paths), batch):
        chunk = paths[i:i + batch]
        ims = [Image.open(p).convert("RGB") for _, p in chunk]
        px = proc(images=ims, return_tensors="pt")["pixel_values"].to(device, torch.float16)
        with torch.no_grad():
            h = model(pixel_values=px).last_hidden_state            # (b, 196, 768), post_layernorm applied
        assert h.shape[1:] == (196, 768)
        for (iid, _), f in zip(chunk, h):
            feats[iid] = f.half().cpu()
    return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "full"], required=True)
    args = ap.parse_args()
    from transformers import AutoImageProcessor, SiglipVisionModel
    images = read_json(ROOT / "data/task2_images.json")
    if args.mode == "smoke":
        images = dict(list(images.items())[:8])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    t0 = time.time()
    with ThreadPoolExecutor(32) as ex:
        paths = list(ex.map(fetch, sorted(images.items())))
    missing = [i for i, p in paths if not p.exists()]
    assert not missing, f"{len(missing)} images failed to download, e.g. {missing[:5]}"
    t_dl = time.time() - t0

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    model = SiglipVisionModel.from_pretrained(VISION, dtype=torch.float16).to(device).eval()
    proc = AutoImageProcessor.from_pretrained(VISION)
    feats = encode(model, proc, paths, device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_enc = time.time() - t0
    peak = torch.cuda.max_memory_allocated() if device.type == "cuda" else None

    # C11 checks: shapes and NaN, determinism on 4 images, post_layernorm not re-applied
    assert all(f.shape == (196, 768) and not torch.isnan(f).any() for f in feats.values())
    again = encode(model, proc, paths[:4], device)
    det = max(float((again[i].float() - feats[i].float()).abs().max()) for i, _ in paths[:4])
    one = feats[paths[0][0]].to(device)
    post_ln = next(m for n, m in model.named_modules() if n.endswith("post_layernorm"))
    relayer = float((post_ln(one) - one).abs().max())
    del model
    torch.cuda.empty_cache() if device.type == "cuda" else None

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"features{'_smoke' if args.mode == 'smoke' else ''}.pt"
    torch.save(feats, path)
    report = {"images": len(feats), "download_seconds": t_dl, "encode_seconds": t_enc, "peak_memory_bytes": peak,
              "storage_bytes": path.stat().st_size, "sha256": sha256_file(path), "determinism_max_abs": det,
              "reapply_post_layernorm_changes_by": relayer, "dtype": "float16", "grid": [14, 14]}
    write_json(ROOT / f"results/task2/feature_cache{'_smoke' if args.mode == 'smoke' else ''}.json", report)
    print(report)


if __name__ == "__main__":
    main()
