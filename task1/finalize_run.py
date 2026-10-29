"""Recover a finished run's final artifacts from its last resume snapshot (used once, after the Hub quota stopped
Full FT's final upload at update 1,250).

python -m task1.finalize_run --method full_ft --snapshot <resume-1250.pt> --store hub:<repo> --peak-from <resources.json>
Writes and uploads final/ weights, resources.json, train_log.jsonl and config.json. Peak memory is not inside the
snapshot, so it is taken from the run's own 20-update dry run on the same GPU and flagged as such.
"""
import argparse
from pathlib import Path

import torch

from common.hub_sync import make_store
from common.results import ROOT, read_json, write_json
from task1.train import build_model, file_bytes, load_cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--store", required=True)
    ap.add_argument("--peak-from", required=True)
    args = ap.parse_args()
    cfg = load_cfg(args.method, "full")
    run_dir = ROOT / "results/task1" / args.method
    s = torch.load(args.snapshot, map_location="cpu", weights_only=False)
    assert s["update"] == cfg["updates"], s["update"]
    model = build_model(cfg, torch.device("cpu"))
    model.load_state_dict(s["model"])
    final = run_dir / "final"
    model.save_pretrained(final, safe_serialization=True)
    store = make_store(args.store)
    for f in sorted(final.iterdir()):
        store.put(f, f"task1/{args.method}/final/{f.name}")
    dry = read_json(args.peak_from)
    sizes = {"resume_bytes": file_bytes(args.snapshot), "final_bytes": sum(file_bytes(f) for f in final.iterdir())}
    if (run_dir / "step-0625").exists():
        sizes["step-0625_bytes"] = sum(file_bytes(f) for f in (run_dir / "step-0625").iterdir())
    write_json(run_dir / "resources.json", {
        "method": args.method, "updates_done": s["update"], "trainable_params": sum(p.numel() for p in model.parameters()),
        "device": dry["device"], "peak_memory_bytes": dry["peak_memory_bytes"],
        "peak_memory_source": "20-update dry run of the same configuration on the same GPU (snapshot lacks it)",
        **s["timers"], "overhead": {}, "checkpoint_sizes": sizes})
    write_json(run_dir / "config.json", cfg)
    for name in ("train_log.jsonl", "resources.json", "config.json"):
        store.put(run_dir / name, f"task1/{args.method}/results/{name}")
    print(read_json(run_dir / "resources.json"))


if __name__ == "__main__":
    main()
