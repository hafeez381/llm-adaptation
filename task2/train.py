"""C13-C16: the three-stage VLM run with development evaluation (working doc D10, D14, S8).

python -m task2.train --mode {smoke,full} --store {hub:<repo>|<folder>} [--resume]
One AdamW for the whole run. The connector group exists from update 1; the LoRA group joins at Stage 2.
Each group follows its stage's schedule: ceil(5%) warmup, then cosine to 10% of that stage's peak.
Sampler: update u draws from default_rng([6304, u]), so the sampler position is the update number.
Stage 1 walks two seeded permutations of the 18,000 captions; Stages 2-3 draw components with the PDF's
probabilities, each draw giving two examples (a complete pair, or two replayed captions).
"""
import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from common.guards import assert_backbone_frozen
from common.hub_sync import make_store
from common.results import ROOT, append_jsonl, read_json, read_jsonl, write_json
from common.seeding import SEED, rng_state, seed_everything, set_rng_state
from task2.vlm import VLM, Connector, Encoder, answer_nll, collate, example_loss, load_lm, tokenizer, user_text

CFG = ROOT / "task2/configs/train.json"


def load_cfg(mode: str) -> dict:
    cfg = read_json(CFG)
    if mode == "smoke":
        s = cfg["smoke"]
        for st, n in zip(cfg["stages"], s["stage_updates"]):
            st["updates"] = n
        cfg |= {k: s[k] for k in ("ckpt_every", "dev_eval_at", "gen_check_at", "calibration_images")}
        cfg["tiny_lm"] = s["tiny_lm"]
    cfg["mode"] = mode
    return cfg


def stage_of(cfg, u: int) -> tuple[int, int]:
    """(stage index, 1-based step within the stage) for global update u."""
    start = 0
    for i, st in enumerate(cfg["stages"]):
        if u <= start + st["updates"]:
            return i, u - start
        start += st["updates"]
    raise ValueError(u)


def stage_lr(cfg, peak: float, j: int, n: int) -> float:
    w = math.ceil(cfg["warmup_frac"] * n)
    if j <= w:
        return peak * j / w
    f = cfg["lr_floor_frac"]
    return peak * (f + (1 - f) * 0.5 * (1 + math.cos(math.pi * (j - w) / max(n - w, 1))))


class Data:
    def __init__(self, items: dict, feats):
        self.items, self.feats = items, feats
        self.caps = items["caption_train"]
        self.pools = {"presence": items["presence_pool"], "counting": items["counting_pool"], "vqa": items["vqa_train"]}

    def batch(self, cfg, u: int) -> tuple[list[dict], Counter]:
        """The 16 examples of update u and the component counts (D14)."""
        s, j = stage_of(cfg, u)
        E = cfg["microbatch"] * cfg["grad_accum"]
        if s == 0:
            n = len(self.caps)
            pos = (j - 1) * E
            perm = np.random.default_rng([SEED, 1000 + pos // n]).permutation(n)
            idx = [perm[(pos + k) % n] for k in range(E)]
            return [self.caps[i] for i in idx], Counter(caption=E // 2)
        rng = np.random.default_rng([SEED, u])
        mix = cfg["stages"][s]["mix"]
        names, probs = list(mix), np.array(list(mix.values()))
        out, counts = [], Counter()
        for _ in range(E // 2):
            c = names[rng.choice(len(names), p=probs)]
            counts[c] += 1
            if c == "caption":
                out += [self.caps[int(rng.integers(len(self.caps)))] for _ in range(2)]
            else:
                pool = self.pools[c]
                out += pool[int(rng.integers(len(pool)))]["items"]
        return out, counts


def encode_batch(enc, exs, image=True, with_target=True, feats=None, image_of=None):
    seqs, fs = [], []
    for x in exs:
        text = user_text(x["kind"], x.get("question"))
        seqs.append(enc.example(text, x["target"] if with_target else None, image))
        iid = image_of(x) if image_of else x["image_id"]
        fs.append(feats[str(iid)] if image else None)
    return seqs, fs


class Trainer:
    def __init__(self, cfg, items, feats, store_spec, run_dir, device=None):
        self.cfg, self.feats = cfg, feats
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.amp = self.device.type == "cuda"
        self.run = "task2/main" + ("_smoke" if cfg["mode"] == "smoke" else "")
        self.dir = Path(run_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.store = make_store(store_spec)
        self.data = Data(items, feats)
        self.items = items
        seed_everything(SEED)
        rev = read_json(ROOT / "data/manifest.json")["revisions"]["hf_model:Qwen/Qwen3-0.6B"]
        self.tok = tokenizer(rev)
        lm = load_lm(rev, self.device, cfg.get("tiny_lm"))
        lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        torch.manual_seed(SEED)
        conn = Connector(768, lm.config.hidden_size).to(self.device)
        self.model = VLM(lm, conn, Encoder(self.tok))
        self.opt = torch.optim.AdamW([{"params": list(conn.parameters()), "weight_decay": cfg["weight_decay"],
                                       "name": "connector"}], lr=cfg["stages"][0]["conn_lr"])
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=self.amp)
        self.update, self.lora_on = 0, False
        self.timers = {"train_seconds": 0.0, "snapshot_seconds": 0.0, "dev_eval_seconds": 0.0}
        self.stage_res = {}

    # ---------- D10 calibration ----------
    def calibrate(self):
        caps = [x["target"] for x in self.items["caption_train"]]
        emb = self.model.lm.get_input_embeddings().weight
        ids = [i for c in caps for i in self.tok(c, add_special_tokens=False).input_ids]
        num = float(emb[torch.tensor(ids, device=emb.device)].float().norm(dim=-1).mean())
        imgs = sorted({x["image_id"] for x in self.items["caption_train"]})
        pick = np.random.default_rng(SEED).choice(imgs, min(self.cfg["calibration_images"], len(imgs)), replace=False)
        with torch.no_grad():
            outs = [self.model.connector.net(self.feats[str(i)].to(self.device).float()).norm(dim=-1).mean() for i in pick]
        den = float(torch.stack(outs).mean())
        s = num / den
        self.model.connector.scale.fill_(s)
        cal = {"rule": "s = mean L2 norm of Qwen input embeddings over training-caption tokens / mean L2 norm of "
                       "initial connector outputs over 512 training images (all 196 tokens)",
               "caption_tokens": len(ids), "images": [int(i) for i in pick], "text_norm": num, "visual_norm": den,
               "scale": s}
        write_json(self.dir / "calibration.json", cal)
        if self.cfg["mode"] == "full":
            write_json(ROOT / "task2/configs/calibration.json", cal)
        return cal

    def trainable(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def attach_lora(self):
        from task1.adapters import attach_adapters
        lcfg = self.cfg["lora"] | {"project": False}
        attach_adapters(self.model.lm, lcfg)
        self.model.lm.to(self.device)
        lora = [p for n, p in self.model.lm.named_parameters() if ".lora_" in n]
        self.opt.add_param_group({"params": lora, "weight_decay": self.cfg["weight_decay"], "name": "lora",
                                  "lr": self.cfg["stages"][1]["lora_lr"]})
        self.lora_on = True
        names = {f"connector.{n}" for n, _ in self.model.connector.named_parameters()} | \
                {f"lm.{n}" for n, _ in self.model.lm.named_parameters() if ".lora_" in n}
        assert_backbone_frozen(self.model, names, bias_ok=("connector.",))

    def set_lr(self, u):
        s, j = stage_of(self.cfg, u)
        st = self.cfg["stages"][s]
        for g in self.opt.param_groups:
            peak = st["conn_lr"] if g["name"] == "connector" else st["lora_lr"]
            g["lr"] = stage_lr(self.cfg, peak, j, st["updates"])

    def step(self) -> dict:
        u = self.update + 1
        s, j = stage_of(self.cfg, u)
        if s >= 1 and not self.lora_on:
            self.attach_lora()
        sync = torch.cuda.synchronize if self.device.type == "cuda" else (lambda: None)
        sync()
        t0 = time.time()
        self.set_lr(u)
        exs, counts = self.data.batch(self.cfg, u)
        E = len(exs)
        loss_total = 0.0
        self.model.lm.train()
        for m in range(0, E, self.cfg["microbatch"]):
            seqs, fs = encode_batch(self.model.enc, exs[m:m + self.cfg["microbatch"]], feats=self.feats)
            ids, attn, labels = collate(seqs, self.tok.pad_token_id, self.device)
            with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.amp):
                nll, keep = self.model.token_nll(ids, attn, labels, fs)
            loss = example_loss(nll, keep).sum() / E             # equation (6): example mean, then batch mean
            self.scaler.scale(loss).backward()
            loss_total += loss.item()
        self.scaler.unscale_(self.opt)
        gnorm = torch.nn.utils.clip_grad_norm_(self.trainable(), self.cfg["clip_grad_norm"])
        scale = self.scaler.get_scale()
        self.scaler.step(self.opt)
        self.scaler.update()
        self.opt.zero_grad(set_to_none=True)
        sync()
        dt = time.time() - t0
        self.timers["train_seconds"] += dt
        self.stage_res.setdefault(s, {"seconds": 0.0})["seconds"] += dt
        self.update = u
        lrs = {g["name"]: g["lr"] for g in self.opt.param_groups}
        return {"update": u, "stage": s + 1, "loss": loss_total, "lr_connector": lrs["connector"],
                "lr_lora": lrs.get("lora"), "grad_norm": float(gnorm), "loss_scale": scale,
                "skipped": bool(self.amp and self.scaler.get_scale() < scale), "mixture": dict(counts),
                "examples": E, "seconds": dt}

    # ---------- development evaluation (C13) ----------
    @torch.no_grad()
    def nll_of(self, exs, image_of=None, image=True, answer=True):
        out = []
        self.model.lm.eval()
        for m in range(0, len(exs), 8):
            chunk = exs[m:m + 8]
            seqs, fs = encode_batch(self.model.enc, chunk, image=image, feats=self.feats, image_of=image_of)
            ids, attn, labels = collate(seqs, self.tok.pad_token_id, self.device)
            with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.amp):
                nll, keep = self.model.token_nll(ids, attn, labels, fs)
            out += (answer_nll(nll, keep) if answer else example_loss(nll, keep)).tolist()
        return out

    def dev_eval(self) -> dict:
        t0 = time.time()
        row = {"update": self.update,
               "caption_loss": float(np.mean(self.nll_of(self.items["caption_dev"], answer=False)))}
        for suite in ("grounding_dev", "vqa_dev"):
            pairs = self.items[suite]
            exs = [x for p in pairs for x in p["items"]]
            partner = {}
            for p in pairs:
                a, b = p["items"]
                partner[id(a)], partner[id(b)] = b["image_id"], a["image_id"]
            c = self.nll_of(exs)
            sw = self.nll_of(exs, image_of=lambda x: partner[id(x)])
            ni = self.nll_of(exs, image=False)
            row[suite] = {"nll_correct": float(np.mean(c)), "nll_complementary": float(np.mean(sw)),
                          "nll_no_image": float(np.mean(ni)), "delta_nll": float(np.mean(np.array(sw) - np.array(c)))}
        if self.update in self.cfg["gen_check_at"]:
            from task2.final_eval import generate
            gens = []
            for suite in ("grounding_dev", "vqa_dev"):
                exs = [x for p in self.items[suite] for x in p["items"]]
                gens += generate(self, exs, lambda x: x["image_id"], True)
            row["gen_check"] = {"n": len(gens), "empty_or_capped": sum(g["empty"] or g["capped"] for g in gens) / len(gens)}
        self.timers["dev_eval_seconds"] += time.time() - t0
        return row

    # ---------- checkpoints (D6, every 250 updates) ----------
    def trainable_state(self):
        st = {"connector": self.model.connector.state_dict()}
        if self.lora_on:
            st["lora"] = {n: p.detach() for n, p in self.model.lm.named_parameters() if ".lora_" in n}
        return st

    def snapshot(self, stage_end: int | None = None):
        t0 = time.time()
        name = f"resume-{self.update:04d}.pt"
        local = self.dir / name
        torch.save({"update": self.update, "lora_on": self.lora_on, "model": self.trainable_state(),
                    "optimizer": self.opt.state_dict(), "scaler": self.scaler.state_dict(), "rng": rng_state(),
                    "timers": self.timers, "stage_res": self.stage_res}, local)
        self.store.put(local, f"{self.run}/resume/{name}")
        for f in ("train_log.jsonl", "dev_metrics.jsonl"):
            if (self.dir / f).exists():
                self.store.put(self.dir / f, f"{self.run}/resume/{f}")
        prev = self.store.read_text(f"{self.run}/resume/LATEST")
        self.store.write_text(f"{self.run}/resume/LATEST", name)
        if prev and prev.strip() != name:
            self.store.delete(f"{self.run}/resume/{prev.strip()}")
            (self.dir / prev.strip()).unlink(missing_ok=True)
        if stage_end is not None:
            w = self.dir / f"stage{stage_end}_weights.pt"
            torch.save(self.trainable_state(), w)
            self.store.put(w, f"{self.run}/stage{stage_end}/weights.pt")
        self.timers["snapshot_seconds"] += time.time() - t0

    def resume(self):
        name = (self.store.read_text(f"{self.run}/resume/LATEST") or "").strip()
        local = self.store.get(f"{self.run}/resume/{name}", self.dir / name)
        for f in ("train_log.jsonl", "dev_metrics.jsonl"):
            self.store.get(f"{self.run}/resume/{f}", self.dir / f)
        s = torch.load(local, map_location=self.device, weights_only=False)
        if s["lora_on"]:
            self.attach_lora()
        self.model.connector.load_state_dict(s["model"]["connector"])
        if s["lora_on"]:
            with torch.no_grad():
                for n, p in self.model.lm.named_parameters():
                    if ".lora_" in n:
                        p.copy_(s["model"]["lora"][n])
        self.opt.load_state_dict(s["optimizer"])
        self.scaler.load_state_dict(s["scaler"])
        set_rng_state(s["rng"])
        self.update, self.timers, self.stage_res = s["update"], s["timers"], s["stage_res"]
        for f in ("train_log.jsonl", "dev_metrics.jsonl"):
            rows = [r for r in read_jsonl(self.dir / f) if r["update"] <= self.update]
            (self.dir / f).write_text("".join(json.dumps(r) + "\n" for r in rows))

    def fit(self, stop_after: int | None = None):
        total = sum(st["updates"] for st in self.cfg["stages"])
        if stop_after is not None:
            total = min(total, self.update + stop_after)
        ends = np.cumsum([st["updates"] for st in self.cfg["stages"]]).tolist()
        if self.update == 0:
            append_jsonl(self.dir / "dev_metrics.jsonl", self.dev_eval())
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        while self.update < total:
            row = self.step()
            append_jsonl(self.dir / "train_log.jsonl", row)
            s = row["stage"] - 1
            if self.device.type == "cuda":
                self.stage_res[s]["peak_memory_bytes"] = max(self.stage_res[s].get("peak_memory_bytes", 0),
                                                             torch.cuda.max_memory_allocated())
            self.stage_res[s]["trainable_params"] = sum(p.numel() for p in self.trainable())
            if self.update in self.cfg["dev_eval_at"]:
                append_jsonl(self.dir / "dev_metrics.jsonl", self.dev_eval())
            end = ends.index(self.update) + 1 if self.update in ends else None
            if self.update % self.cfg["ckpt_every"] == 0 or end:
                self.snapshot(stage_end=end)
        write_json(self.dir / "resources.json", {"timers": self.timers, "stages": self.stage_res,
                                                 "device": torch.cuda.get_device_name(0) if self.device.type == "cuda" else "cpu"})
        for f in ("train_log.jsonl", "dev_metrics.jsonl", "resources.json", "calibration.json"):
            if (self.dir / f).exists():
                self.store.put(self.dir / f, f"{self.run}/results/{f}")


def load_inputs(mode: str):
    suffix = "_smoke" if mode == "smoke" else ""
    items = read_json(ROOT / f"data/cache/task2_items{suffix}.json")
    if mode == "smoke":  # random stand-in features: smoke checks plumbing, not learning
        ids = {str(x["image_id"]) for k, v in items.items()
               for x in (v if k.startswith("caption") else [y for p in v for y in p["items"]])}
        g = torch.Generator().manual_seed(SEED)
        feats = {i: torch.randn(196, 768, generator=g).half() for i in sorted(ids)}
    else:
        feats = torch.load(ROOT / "data/cache/task2_features/features.pt")
    return items, feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "full"], required=True)
    ap.add_argument("--store", required=True)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    cfg = load_cfg(args.mode)
    items, feats = load_inputs(args.mode)
    run_dir = ROOT / ("results/task2/train" + ("_smoke" if args.mode == "smoke" else ""))
    tr = Trainer(cfg, items, feats, args.store, run_dir)
    if args.resume:
        tr.resume()
        cal = read_json(ROOT / "task2/configs/calibration.json") if args.mode == "full" else read_json(run_dir / "calibration.json")
        tr.model.connector.scale.fill_(cal["scale"])
    else:
        print("calibration", tr.calibrate())
    write_json(run_dir / "config.json", cfg)
    tr.fit()
    print("done", tr.update, read_jsonl(run_dir / "train_log.jsonl")[-1]["loss"])


if __name__ == "__main__":
    main()
