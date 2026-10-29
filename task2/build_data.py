"""C10: caption, annotation-derived and natural-VQAv2 sets at the PDF's sizes (working doc D11-D13, S7, S9).

python -m task2.build_data --mode {smoke,full}
Reads COCO 2014 captions/instances and VQAv2 v2 files from data/cache/task2_raw. Writes the committed ID files
data/task2_ids.json and data/task2_images.json, and data/cache/task2_items.json (the example text the VM needs).
Label inspection (D13) is programmatic: crowd annotations excluded, counted instances >= 0.5% of the image,
presence positives with one instance >= 1% of the image (deviation from visual inspection, see working doc).
"""
import argparse
import json
from collections import Counter, defaultdict

import numpy as np

from common.results import ROOT, sha256_json, write_json
from common.seeding import SEED
from task2.vqa_norm import normalize
from task2.vlm import LM

RAW = ROOT / "data/cache/task2_raw"
CAPS = {"question": 80, "caption": 64, "short": 16}
VQA_TRAIN = {"yes/no": 640, "number": 256, "other": 640}
VQA_DEV = {"yes/no": 12, "number": 8, "other": 12}
VQA_TEST = {"yes/no": 36, "number": 24, "other": 36}
GROUND = {"dev": (12, 12), "test": (32, 32)}      # (presence pairs, counting pairs), S7
POOL = {"presence": 4000, "counting": 4000}         # reusable training pools, sampled with replacement
MIN_COUNT_AREA, MIN_PRESENCE_AREA = 0.005, 0.01
PLURAL = {"person": "people", "bus": "buses", "knife": "knives", "mouse": "mice", "sheep": "sheep", "skis": "skis",
          "scissors": "scissors", "wine glass": "wine glasses", "couch": "couches", "bench": "benches",
          "sandwich": "sandwiches", "toothbrush": "toothbrushes", "glass": "glasses", "potted plant": "potted plants",
          "dining table": "dining tables", "hair drier": "hair driers", "teddy bear": "teddy bears", "skateboard": "skateboards",
          "surfboard": "surfboards", "tennis racket": "tennis rackets", "baseball bat": "baseball bats",
          "baseball glove": "baseball gloves", "sports ball": "sports balls", "fire hydrant": "fire hydrants",
          "stop sign": "stop signs", "parking meter": "parking meters", "traffic light": "traffic lights",
          "cell phone": "cell phones", "hot dog": "hot dogs"}


def plural(name: str) -> str:
    return PLURAL.get(name, name + "s")


def article(name: str) -> str:
    return ("an " if name[0] in "aeiou" else "a ") + name


def load(name):
    with open(RAW / name) as f:
        return json.load(f)


def ntok(tok, texts):
    return [len(x) for x in tok(texts, add_special_tokens=False).input_ids]


# ---------- captions (D12) ----------
def select_captions(tok, rng, smoke):
    tr = load("annotations/captions_train2014.json")
    va = load("annotations/captions_val2014.json")
    imgs = {i["id"]: i for i in tr["images"]} | {i["id"]: i for i in va["images"]}
    by_img = defaultdict(list)
    anns = sorted(tr["annotations"], key=lambda a: a["id"])
    for a, n in zip(anns, ntok(tok, [a["caption"].strip() for a in anns])):
        if n <= CAPS["caption"]:
            by_img[a["image_id"]].append(a)
    eligible = sorted(i for i, v in by_img.items() if len(v) >= 3)
    n_train = 60 if smoke else 6000
    assert len(eligible) >= n_train
    train_imgs = sorted(rng.choice(eligible, n_train, replace=False).tolist())
    train = [{"kind": "caption", "image_id": i, "caption_id": a["id"], "target": a["caption"].strip()}
             for i in train_imgs for a in by_img[i][:3]]
    val_by = defaultdict(list)
    vanns = sorted(va["annotations"], key=lambda a: a["id"])
    for a, n in zip(vanns, ntok(tok, [a["caption"].strip() for a in vanns])):
        if n <= CAPS["caption"]:
            val_by[a["image_id"]].append(a)
    return train, val_by, imgs


# ---------- natural VQAv2 (S9) ----------
def vqa_pairs(split: str):
    qs = {q["question_id"]: q for q in load(f"v2_OpenEnded_mscoco_{split}2014_questions.json")["questions"]}
    an = {a["question_id"]: a for a in load(f"v2_mscoco_{split}2014_annotations.json")["annotations"]}
    pairs = load(f"v2_mscoco_{split}2014_complementary_pairs.json")
    keep = defaultdict(list)
    for q1, q2 in pairs:
        items = []
        for q in (q1, q2):
            a = an[q]
            norm = [normalize(x["answer"]) for x in a["answers"]]
            cons, votes = Counter(norm).most_common(1)[0]
            raw = Counter(x["answer"] for x in a["answers"] if normalize(x["answer"]) == cons)
            target = sorted(raw.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
            items.append({"kind": "vqa", "question_id": q, "image_id": a["image_id"], "question": qs[q]["question"],
                          "answers": [x["answer"] for x in a["answers"]], "consensus": cons, "votes": votes,
                          "target": target, "answer_type": a["answer_type"]})
        a1, a2 = items
        if a1["votes"] >= 8 and a2["votes"] >= 8 and a1["consensus"] != a2["consensus"] \
                and a1["answer_type"] == a2["answer_type"] and a1["image_id"] != a2["image_id"]:
            keep[a1["answer_type"]].append({"pair_id": f"vqa:{q1}-{q2}", "type": a1["answer_type"], "items": items})
    return keep


def sample_pairs(pool: dict, sizes: dict, rng, used: set) -> list[dict]:
    out = []
    for t, n in sizes.items():
        cand = sorted(pool[t], key=lambda p: p["pair_id"])
        for j in rng.permutation(len(cand)):
            p = cand[j]
            ims = {x["image_id"] for x in p["items"]}
            if ims & used:
                continue
            out.append(p)
            used |= ims
            if sum(q["type"] == t for q in out) == n:
                break
        assert sum(q["type"] == t for q in out) == n, f"not enough {t} pairs"
    return out


# ---------- annotation-derived grounding (D13) ----------
def instance_index(split: str, universe: set | None):
    ins = load(f"annotations/instances_{split}2014.json")
    cats = {c["id"]: c["name"] for c in ins["categories"]}
    area = {i["id"]: i["width"] * i["height"] for i in ins["images"]}
    per = defaultdict(lambda: defaultdict(list))   # image -> category -> [(area fraction, iscrowd)]
    for a in ins["annotations"]:
        if universe is None or a["image_id"] in universe:
            per[a["image_id"]][a["category_id"]].append((a["area"] / area[a["image_id"]], a["iscrowd"]))
    imgs = sorted(area) if universe is None else sorted(universe & set(area))
    return cats, per, imgs


def presence_pos(per, img, c):
    xs = per[img].get(c, [])
    return any(not cr for _, cr in xs) and max((f for f, cr in xs if not cr), default=0) >= MIN_PRESENCE_AREA


def count_of(per, img, c):
    xs = per[img].get(c, [])
    if not xs or any(cr for _, cr in xs) or any(f < MIN_COUNT_AREA for f, _ in xs):
        return None
    n = len(xs)
    return n if 1 <= n <= 4 else None


def presence_pair(cats, per, imgs, rng, c, used=frozenset()):
    pos = [i for i in imgs if i not in used and presence_pos(per, i, c)]
    neg = [i for i in imgs if i not in used and c not in per[i]]
    if not pos or not neg:
        return None
    p, n = int(rng.choice(pos)), int(rng.choice(neg))
    q = f"Does this image contain {article(cats[c])}?"
    return {"pair_id": f"presence:{c}:{p}-{n}", "type": "presence", "category": cats[c],
            "items": [{"kind": "presence", "image_id": p, "question": q, "target": "yes", "answers": ["yes"]},
                      {"kind": "presence", "image_id": n, "question": q, "target": "no", "answers": ["no"]}]}


def counting_pair(cats, per, imgs, rng, c, used=frozenset()):
    by_n = defaultdict(list)
    for i in imgs:
        if i not in used:
            n = count_of(per, i, c)
            if n:
                by_n[n].append(i)
    ns = sorted(by_n)
    if len(ns) < 2:
        return None
    n1, n2 = rng.choice(ns, 2, replace=False).tolist()
    a, b = int(rng.choice(by_n[n1])), int(rng.choice(by_n[n2]))
    q = f"How many {plural(cats[c])} can you see?"
    return {"pair_id": f"counting:{c}:{a}-{b}", "type": "counting", "category": cats[c],
            "items": [{"kind": "counting", "image_id": a, "question": q, "target": str(n1), "answers": [str(n1)]},
                      {"kind": "counting", "image_id": b, "question": q, "target": str(n2), "answers": [str(n2)]}]}


def grounding_pool(cats, per, imgs, rng, n, make):
    out, ids = [], sorted(cats)
    while len(out) < n:
        p = make(cats, per, imgs, rng, int(rng.choice(ids)))
        if p:
            out.append(p)
    return out


def grounding_suite(cats, per, imgs, rng, n_pres, n_cnt, used: set):
    out = []
    for n, make in ((n_pres, presence_pair), (n_cnt, counting_pair)):
        got = 0
        while got < n:
            p = make(cats, per, imgs, rng, int(rng.choice(sorted(cats))), used)
            if p:
                out.append(p)
                used |= {x["image_id"] for x in p["items"]}
                got += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "full"], required=True)
    args = ap.parse_args()
    smoke = args.mode == "smoke"
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(LM)
    rng = np.random.default_rng(SEED)

    cap_train, val_caps, coco_imgs = select_captions(tok, rng, smoke)
    vqa_tr = vqa_pairs("train")
    scale = (lambda d: {k: max(1, v // 40) for k, v in d.items()}) if smoke else (lambda d: d)
    vqa_train = sample_pairs(vqa_tr, scale(VQA_TRAIN), rng, set())
    del vqa_tr
    vqa_va = vqa_pairs("val")
    used_eval: set = set()
    vqa_dev = sample_pairs(vqa_va, scale(VQA_DEV), rng, used_eval)
    vqa_test = sample_pairs(vqa_va, scale(VQA_TEST), rng, used_eval)
    del vqa_va

    universe = {x["image_id"] for x in cap_train} | {x["image_id"] for p in vqa_train for x in p["items"]}
    cats, per, imgs = instance_index("train", universe)
    pool_n = {k: (40 if smoke else v) for k, v in POOL.items()}
    presence = grounding_pool(cats, per, imgs, rng, pool_n["presence"], presence_pair)
    counting = grounding_pool(cats, per, imgs, rng, pool_n["counting"], counting_pair)
    del per
    vcats, vper, vimgs = instance_index("val", None)
    g_dev = grounding_suite(vcats, vper, vimgs, rng, *(GROUND["dev"] if not smoke else (2, 2)), used_eval)
    g_test = grounding_suite(vcats, vper, vimgs, rng, *(GROUND["test"] if not smoke else (2, 2)), used_eval)
    del vper
    # caption dev images: val2014, unused by any other evaluation set, first caption <= 64 tokens
    cand = sorted(i for i in val_caps if i not in used_eval)
    cap_dev = []
    for j in rng.permutation(len(cand)):
        i = cand[j]
        cap_dev.append({"kind": "caption", "image_id": i, "caption_id": val_caps[i][0]["id"],
                        "target": val_caps[i][0]["caption"].strip()})
        used_eval.add(i)
        if len(cap_dev) == (8 if smoke else 96):
            break

    sets = {"caption_train": cap_train, "caption_dev": cap_dev, "vqa_train": vqa_train, "vqa_dev": vqa_dev,
            "vqa_test": vqa_test, "presence_pool": presence, "counting_pool": counting,
            "grounding_dev": g_dev, "grounding_test": g_test}
    # caps (asserted, never truncated)
    items = [x for k, v in sets.items() for x in (v if k.startswith("caption") else [y for p in v for y in p["items"]])]
    assert max(ntok(tok, [x["question"] for x in items if "question" in x])) <= CAPS["question"]
    assert max(ntok(tok, [x["target"] for x in items if x["kind"] == "caption"])) <= CAPS["caption"]
    assert max(ntok(tok, [x["target"] for x in items if x["kind"] != "caption"])) <= CAPS["short"]
    for p in counting + g_dev + g_test:
        if p["type"] == "counting":
            a, b = (x["target"] for x in p["items"])
            assert a != b and {a, b} <= {"1", "2", "3", "4"}

    img_sets = {k: sorted({x["image_id"] for x in (v if k.startswith("caption") else [y for p in v for y in p["items"]])})
                for k, v in sets.items()}
    ids = {k: ([x["caption_id"] for x in v] if k.startswith("caption") else [p["pair_id"] for p in v]) for k, v in sets.items()}
    report = {"sizes": {k: len(v) for k, v in sets.items()},
              "vqa_types": {k: dict(Counter(p["type"] for p in sets[k])) for k in ("vqa_train", "vqa_dev", "vqa_test")},
              "grounding_types": {k: dict(Counter(p["type"] for p in sets[k])) for k in ("grounding_dev", "grounding_test")},
              "images": {k: len(v) for k, v in img_sets.items()}, "label_rules": {
                  "crowd": "excluded", "min_count_instance_area": MIN_COUNT_AREA,
                  "min_presence_instance_area": MIN_PRESENCE_AREA}}
    all_imgs = sorted(set().union(*img_sets.values()))
    images = {str(i): coco_imgs[i]["coco_url"] for i in all_imgs}
    suffix = "_smoke" if smoke else ""
    write_json(ROOT / f"data/cache/task2_items{suffix}.json", sets)
    if not smoke:
        write_json(ROOT / "data/task2_ids.json", {"seed": SEED, "ids": ids, "image_sets": img_sets, "report": report,
                                                  "items_sha256": sha256_json(sets)})
        write_json(ROOT / "data/task2_images.json", images)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
