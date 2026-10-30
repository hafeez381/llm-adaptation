"""C2: one trainer for all five Task 1 runs (working doc D1, D3, D5, D6).

python -m task1.train --method {full_ft,l2sp,lora4,lora16,oplora16} --mode {smoke,full}
                      --store {hub:<user>/llm-adaptation | <local folder>} [--resume] [--stop-after N]

Per update: 16 microbatches of 2; each microbatch loss is its summed token CE divided by the valid
response tokens of all 32 examples, so the accumulated gradient is the effective batch's token mean.
Then unscale, add the L2-SP gradient (if lam > 0), clip to global norm 1.0, step, update the scaler.
"""
import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM

from common.guards import assert_backbone_frozen, assert_source_order
from common.hub_sync import make_store
from common.results import ROOT, append_jsonl, read_json, read_jsonl, write_json
from common.seeding import SEED, rng_state, seed_everything, set_rng_state

CFG_PATH = ROOT / "task1/configs/train.json"


def load_cfg(method: str, mode: str) -> dict:
    raw = read_json(CFG_PATH)
    cfg = {k: v for k, v in raw.items() if k not in ("methods", "smoke")}
    cfg |= raw["methods"][method] | {"method": method, "mode": mode}
    if mode == "smoke":
        cfg |= {k: v for k, v in raw["smoke"].items() if k != "k"}
        if "k" in cfg:
            cfg["k"] = raw["smoke"]["k"]
    return cfg


def build_model(cfg: dict, device):
    full = cfg["kind"] == "full"
    if cfg["mode"] == "smoke":
        # tiny random Qwen3 with the real tokenizer's vocabulary, so the code path matches the real run
        mc = AutoConfig.from_pretrained(cfg["model"])
        for k, v in cfg["tiny_model"].items():
            setattr(mc, k, v)
        mc.layer_types = mc.layer_types[:mc.num_hidden_layers]
        torch.manual_seed(SEED)
        model = AutoModelForCausalLM.from_config(mc, dtype=torch.float32)
    else:
        # adapters too load in FP32 first: the released bf16 weights upcast exactly, so OPLoRA's SVD
        # sees the same W0 as the FP32 diagnostics (S6); the backbone is cast to FP16 afterwards
        rev = read_json(ROOT / "data/manifest.json")["revisions"][f"hf_model:{cfg['model']}"]
        model = AutoModelForCausalLM.from_pretrained(cfg["model"], revision=rev, dtype=torch.float32)
    model.config.use_cache = False
    if not full:
        from task1.adapters import attach_adapters, cast_backbone_fp16
        model.requires_grad_(False)
        model.to(device)  # SVD of 196 matrices runs on the GPU in a full run
        attach_adapters(model, cfg)
        cast_backbone_fp16(model)
    model.to(device)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    return model


def sft_loss_sum(model, input_ids, attention_mask, labels):
    """Summed next-token CE over supervised positions (D3: lm_head only where labels are kept)."""
    h = model.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    tgt = labels[:, 1:]
    keep = tgt != -100
    logits = model.lm_head(h[:, :-1][keep]).float()
    return F.cross_entropy(logits, tgt[keep], reduction="sum"), int(keep.sum())


def collate(examples: list[tuple[list[int], int]], pad_id: int, device):
    """Right-pads; labels are -100 on the prompt and padding."""
    T = max(len(s) for s, _ in examples)
    ids = torch.full((len(examples), T), pad_id)
    attn = torch.zeros((len(examples), T), dtype=torch.long)
    labels = torch.full((len(examples), T), -100)
    for i, (s, p) in enumerate(examples):
        ids[i, :len(s)] = torch.tensor(s)
        attn[i, :len(s)] = 1
        labels[i, p:len(s)] = ids[i, p:len(s)]
    return ids.to(device), attn.to(device), labels.to(device)


def lr_lambda(cfg: dict):
    """Linear warmup over updates 1..W, then cosine to lr_floor at the last update (S3)."""
    W, U, floor = cfg["warmup_updates"], cfg["updates"], cfg["lr_floor"]

    def f(step):  # LambdaLR passes the number of scheduler steps taken so far, so update = step + 1
        u = step + 1
        if u <= W:
            return u / W
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * (u - W - 1) / (U - W)))
    return f


def param_groups(model, weight_decay: float):
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    decay = [p for n, p in trainable if p.ndim >= 2]
    no_decay = [p for n, p in trainable if p.ndim < 2]
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def trainable_state(model, full: bool) -> dict:
    return model.state_dict() if full else {n: p.detach() for n, p in model.named_parameters() if p.requires_grad}


def file_bytes(p) -> int:
    return Path(p).stat().st_size


class Trainer:
    def __init__(self, cfg: dict, data: dict, store_spec: str, run_dir, device=None, amp: bool = True,
                 snapshot_store: str | None = None):
        self.cfg, self.data = cfg, data
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.amp = amp
        self.full = cfg["kind"] == "full"
        self.run = f"task1/{cfg['method']}" + ("_smoke" if cfg["mode"] == "smoke" else "")
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.run_dir / "train_log.jsonl"
        self.store = make_store(store_spec)
        # resume snapshots may live elsewhere (a VM-local folder) when the Hub quota cannot hold 7 GB Full FT states
        self.snap = make_store(snapshot_store) if snapshot_store else self.store

        seed_everything(cfg["seed"])
        self.model = build_model(cfg, self.device)
        self.params = [p for p in self.model.parameters() if p.requires_grad]
        if not self.full:
            assert_backbone_frozen(self.model, {n for n, _ in self.model.named_parameters() if ".lora_" in n})
        self.opt = torch.optim.AdamW(param_groups(self.model, cfg["weight_decay"]), lr=cfg["lr"],
                                     betas=tuple(cfg["betas"]), eps=cfg["eps"])
        self.sched = torch.optim.lr_scheduler.LambdaLR(self.opt, lr_lambda(cfg))
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=amp)
        self.update = 0
        self.timers = {"train_seconds": 0.0, "snapshot_seconds": 0.0}
        self.overhead = {}
        self.theta0 = None
        if cfg.get("lam", 0) > 0:
            t = time.time()
            # FP32 copy of theta0. GPUs under 20 GB (a T4) cannot also hold it next to Full FT's weights, gradients
            # and AdamW state, so it stays in pinned CPU memory and is streamed in each update (timed below).
            small = self.device.type == "cuda" and torch.cuda.get_device_properties(0).total_memory < 20e9
            self.theta0_on_cpu = small
            self.theta0 = [p.detach().to("cpu").pin_memory() if small else p.detach().clone() for p in self.params]
            self.overhead |= {"theta0_bytes": sum(p.numel() * p.element_size() for p in self.theta0),
                              "theta0_copy_seconds": time.time() - t,
                              "theta0_device": "cpu (pinned)" if small else str(self.device),
                              "theta0_transfer_seconds": 0.0}
        if hasattr(self.model, "adapter_overhead"):
            self.overhead |= self.model.adapter_overhead
        self.E = cfg["microbatch"] * cfg["grad_accum"]
        self.pad_id = data.get("pad_id", 0)

    # ---------- one optimizer update ----------
    def accumulate(self, batch: list[tuple[list[int], int]], microbatch: int) -> tuple[float, int]:
        """Backward over the effective batch with one shared denominator; returns (token-mean loss, n_valid)."""
        n_valid = sum(len(s) - p for s, p in batch)
        sft = 0.0
        for m in range(0, len(batch), microbatch):
            ids, attn, labels = collate(batch[m:m + microbatch], self.pad_id, self.device)
            with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.amp):
                loss_sum, _ = sft_loss_sum(self.model, ids, attn, labels)
            loss = loss_sum / n_valid
            self.scaler.scale(loss).backward()
            sft += loss.item()
        return sft, n_valid

    def step(self) -> dict:
        cfg, u = self.cfg, self.update + 1
        sync = torch.cuda.synchronize if self.device.type == "cuda" else (lambda: None)
        sync()
        t0 = time.time()
        lo = (u - 1) * self.E
        batch = [(self.data["input_ids"][i], self.data["prompt_len"][i]) for i in range(lo, lo + self.E)]
        sft, n_valid = self.accumulate(batch, cfg["microbatch"])
        self.scaler.unscale_(self.opt)
        row = {"update": u, "sft_loss": sft, "lr": self.sched.get_last_lr()[0], "tokens": n_valid}
        if self.theta0 is not None:
            with torch.no_grad():
                if self.theta0_on_cpu:
                    omega, tt = 0.0, time.time()
                    for p, p0 in zip(self.params, self.theta0):
                        d = p - p0.to(self.device, non_blocking=True)
                        omega += 0.5 * float(d.double().pow(2).sum())
                        p.grad.add_(d, alpha=cfg["lam"])
                    self.overhead["theta0_transfer_seconds"] += time.time() - tt
                else:
                    diff = torch._foreach_sub(self.params, self.theta0)
                    omega = 0.5 * sum(float(d.double().pow(2).sum()) for d in diff)
                    torch._foreach_add_([p.grad for p in self.params], diff, alpha=cfg["lam"])
            row |= {"omega": omega, "lam_omega": cfg["lam"] * omega, "total_loss": sft + cfg["lam"] * omega}
        gnorm = torch.nn.utils.clip_grad_norm_(self.params, cfg["clip_grad_norm"])
        scale = self.scaler.get_scale()
        self.scaler.step(self.opt)
        self.scaler.update()
        self.sched.step()
        self.opt.zero_grad(set_to_none=True)
        sync()
        dt = time.time() - t0
        self.timers["train_seconds"] += dt
        self.update = u
        row |= {"grad_norm": float(gnorm), "loss_scale": scale,
                "skipped": bool(self.amp and self.scaler.get_scale() < scale), "seconds": dt}
        return row

    # ---------- snapshots (D6) ----------
    def snapshot(self, keep_weights: bool) -> None:
        t0 = time.time()
        u = self.update
        name = f"resume-{u:04d}.pt"
        local = self.run_dir / name
        torch.save({"update": u, "examples_seen": u * self.E, "model": trainable_state(self.model, self.full),
                    "optimizer": self.opt.state_dict(), "scheduler": self.sched.state_dict(),
                    "scaler": self.scaler.state_dict(), "rng": rng_state(), "timers": self.timers}, local)
        self.snap.put(local, f"{self.run}/resume/{name}")
        self.snap.put(self.log_path, f"{self.run}/resume/train_log.jsonl")
        # a second copy of the log in the main store, so losing the snapshot store never loses the curve
        self.store.put(self.log_path, f"{self.run}/results/train_log.jsonl")
        prev = self.snap.read_text(f"{self.run}/resume/LATEST")
        self.snap.write_text(f"{self.run}/resume/LATEST", name)
        if prev and prev.strip() != name:  # only after the new snapshot's sha is verified
            self.snap.delete(f"{self.run}/resume/{prev.strip()}")
            (self.run_dir / prev.strip()).unlink(missing_ok=True)
        sizes = {"resume_bytes": file_bytes(local)}
        if keep_weights:
            tag = "final" if u == self.cfg["updates"] else f"step-{u:04d}"
            wdir = self.run_dir / tag
            if self.full:
                self.model.save_pretrained(wdir, safe_serialization=True)
            else:
                wdir.mkdir(exist_ok=True)
                torch.save(self.model.export_adapters(), wdir / "adapters.pt")
            for f in sorted(wdir.iterdir()):
                self.store.put(f, f"{self.run}/{tag}/{f.name}")
            sizes[f"{tag}_bytes"] = sum(file_bytes(f) for f in wdir.iterdir())
        self.timers["snapshot_seconds"] += time.time() - t0
        self.sizes = getattr(self, "sizes", {}) | sizes

    def resume(self) -> None:
        name = (self.snap.read_text(f"{self.run}/resume/LATEST") or "").strip()
        if not name:
            raise FileNotFoundError(f"no snapshot for {self.run}")
        local = self.snap.get(f"{self.run}/resume/{name}", self.run_dir / name)
        self.snap.get(f"{self.run}/resume/train_log.jsonl", self.log_path)
        s = torch.load(local, map_location=self.device, weights_only=False)
        if self.full:
            self.model.load_state_dict(s["model"])
        else:
            with torch.no_grad():
                for n, p in self.model.named_parameters():
                    if p.requires_grad:
                        p.copy_(s["model"][n])
        self.opt.load_state_dict(s["optimizer"])
        self.sched.load_state_dict(s["scheduler"])
        self.scaler.load_state_dict(s["scaler"])
        set_rng_state(s["rng"])
        self.update, self.timers = s["update"], s["timers"]
        rows = [r for r in read_jsonl(self.log_path) if r["update"] <= self.update]
        self.log_path.write_text("".join(__import__("json").dumps(r) + "\n" for r in rows))

    # ---------- main loop ----------
    def fit(self, stop_after: int | None = None) -> list[dict]:
        cfg = self.cfg
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        rows = []
        end = cfg["updates"] if stop_after is None else min(cfg["updates"], self.update + stop_after)
        while self.update < end:
            row = self.step()
            append_jsonl(self.log_path, row)
            rows.append(row)
            # extra resume points between the PDF's recovery checkpoints (small adapter states only)
            every = cfg.get("resume_every", {}).get(cfg["kind"], 0)
            if self.update in cfg["ckpt_updates"] or (every and self.update % every == 0):
                self.snapshot(keep_weights=self.update in cfg["keep_weights_at"])
        self.write_resources()
        for name in ("train_log.jsonl", "resources.json", "config.json"):  # results never live only on the VM
            if (self.run_dir / name).exists():
                self.store.put(self.run_dir / name, f"{self.run}/results/{name}")
        return rows

    def write_resources(self) -> None:
        peak = torch.cuda.max_memory_allocated() if self.device.type == "cuda" else None
        write_json(self.run_dir / "resources.json", {
            "method": self.cfg["method"], "updates_done": self.update,
            "trainable_params": sum(p.numel() for p in self.params),
            "device": torch.cuda.get_device_name(0) if self.device.type == "cuda" else "cpu",
            "peak_memory_bytes": peak, **self.timers,
            "overhead": self.overhead | getattr(self.model, "adapter_overhead", {}),
            "checkpoint_sizes": getattr(self, "sizes", {})})


def load_data(cfg: dict) -> dict:
    from transformers import AutoTokenizer
    smoke = cfg["mode"] == "smoke"
    d = torch.load(ROOT / f"data/cache/task1_train{'_smoke' if smoke else ''}.pt")
    if not smoke:
        man = read_json(ROOT / "data/manifest.json")["task1_data"]
        assert_source_order(d["source_ids"], man, list(map(len, d["input_ids"])), cfg["max_len"])
    n = cfg["examples"]
    assert len(d["input_ids"]) >= n
    d["pad_id"] = AutoTokenizer.from_pretrained(cfg["model"]).pad_token_id
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=list(read_json(CFG_PATH)["methods"]))
    ap.add_argument("--mode", choices=["smoke", "full"], required=True)
    ap.add_argument("--store", required=True, help="hub:<user>/<repo> or a local folder")
    ap.add_argument("--snapshot-store", help="separate store for resume snapshots (default: --store)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--stop-after", type=int, help="dry run: stop after this many updates, no final save")
    args = ap.parse_args()

    cfg = load_cfg(args.method, args.mode)
    run_dir = ROOT / "results/task1" / (args.method + ("_smoke" if args.mode == "smoke" else ""))
    tr = Trainer(cfg, load_data(cfg), args.store, run_dir, snapshot_store=args.snapshot_store)
    if args.resume:
        tr.resume()
    write_json(run_dir / "config.json", cfg)
    rows = tr.fit(args.stop_after)
    print(f"updates {tr.update}, last loss {rows[-1]['sft_loss']:.4f}, "
          f"skipped {sum(r['skipped'] for r in read_jsonl(tr.log_path))}")


if __name__ == "__main__":
    main()
