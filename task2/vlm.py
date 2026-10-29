"""C12: SigLIP features -> MLP connector -> frozen Qwen3-0.6B (working doc D9, D10).

User turn: <|vision_start|> + 196 x <|image_pad|> + <|vision_end|> + "\n" + text, inside Qwen's chat template
with thinking off. The <|image_pad|> embeddings are overwritten with s * g(F(I)). A no-image input drops the
whole vision span, so the prefix is shorter and positions stay contiguous. Labels cover the target and
<|im_end|> only.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

LM = "Qwen/Qwen3-0.6B"
N_VIS = 196
CAPTION_PROMPT = "Describe the image in one sentence."
SHORT_SUFFIX = "Answer the question using a single word or phrase."


class Connector(nn.Module):
    def __init__(self, d_vis=768, d_lm=1024):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_vis, d_lm), nn.GELU(), nn.Linear(d_lm, d_lm))
        self.register_buffer("scale", torch.ones(()))  # fixed calibration s (D10), not trained

    def forward(self, feats):
        return self.scale * self.net(feats.float())


def user_text(kind: str, question: str | None) -> str:
    return CAPTION_PROMPT if kind == "caption" else f"{question}\n{SHORT_SUFFIX}"


class Encoder:
    """Token ids for one example: prompt (with or without the vision span) + target + <|im_end|>."""

    def __init__(self, tok, n_vis=N_VIS):
        self.tok, self.n_vis = tok, n_vis
        self.pad = tok.convert_tokens_to_ids("<|image_pad|>")
        self.vs, self.ve = tok.convert_tokens_to_ids("<|vision_start|>"), tok.convert_tokens_to_ids("<|vision_end|>")
        self.eot = tok.convert_tokens_to_ids("<|im_end|>")

    def prompt(self, text: str, image: bool) -> list[int]:
        vis = "<|vision_start|>" + "<|image_pad|>" * self.n_vis + "<|vision_end|>\n" if image else ""
        return self.tok.apply_chat_template([{"role": "user", "content": vis + text}], add_generation_prompt=True,
                                            enable_thinking=False, tokenize=True, return_dict=False)

    def example(self, text: str, target: str | None, image: bool) -> tuple[list[int], int]:
        p = self.prompt(text, image)
        if target is None:
            return p, len(p)
        return p + self.tok(target, add_special_tokens=False).input_ids + [self.eot], len(p)


class VLM(nn.Module):
    def __init__(self, lm, connector, enc: Encoder):
        super().__init__()
        self.lm, self.connector, self.enc = lm, connector, enc

    def embed(self, ids: torch.Tensor, feats: list[torch.Tensor | None]) -> torch.Tensor:
        """Text embeddings with each example's <|image_pad|> positions replaced by its visual tokens."""
        emb = self.lm.get_input_embeddings()(ids)
        pads = ids == self.enc.pad
        rows = [b for b, f in enumerate(feats) if f is not None]
        if rows:
            vis = self.connector(torch.stack([feats[b] for b in rows]).to(ids.device))   # (n, 196, 1024), FP32
            emb = emb.clone()
            for j, b in enumerate(rows):
                assert int(pads[b].sum()) == self.enc.n_vis, "an image example needs exactly n_vis visual positions"
                emb[b, pads[b]] = vis[j].to(emb.dtype)
        assert all(not pads[b].any() for b, f in enumerate(feats) if f is None), "no-image input kept visual positions"
        return emb

    def token_nll(self, ids, attn, labels, feats) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-position NLL (B, T-1) and the supervised mask, lm_head applied at supervised positions only."""
        h = self.lm.model(inputs_embeds=self.embed(ids, feats), attention_mask=attn).last_hidden_state
        tgt = labels[:, 1:]
        keep = tgt != -100
        nll = torch.zeros(tgt.shape, device=ids.device, dtype=torch.float32)
        logits = self.lm.lm_head(h[:, :-1][keep]).float()
        nll[keep] = F.cross_entropy(logits, tgt[keep], reduction="none")
        return nll, keep


def collate(examples, pad_id: int, device, left: bool = False):
    """examples: [(ids, prompt_len)]. Right padding for training/NLL, left padding for generation."""
    T = max(len(s) for s, _ in examples)
    ids = torch.full((len(examples), T), pad_id)
    attn = torch.zeros((len(examples), T), dtype=torch.long)
    labels = torch.full((len(examples), T), -100)
    for i, (s, p) in enumerate(examples):
        o = T - len(s) if left else 0
        ids[i, o:o + len(s)] = torch.tensor(s)
        attn[i, o:o + len(s)] = 1
        labels[i, o + p:o + len(s)] = ids[i, o + p:o + len(s)]
    return ids.to(device), attn.to(device), labels.to(device)


def example_loss(nll, keep) -> torch.Tensor:
    """Equation (6) for one microbatch: mean over each example's supervised tokens; caller averages examples."""
    return (nll.sum(1) / keep.sum(1).clamp(min=1))


def answer_nll(nll, keep) -> torch.Tensor:
    """Reference-answer NLL with termination excluded: drop each example's last supervised token (<|im_end|>)."""
    keep = keep.clone()
    last = keep.shape[1] - 1 - keep.flip(1).float().argmax(1)
    keep[torch.arange(keep.shape[0]), last] = False
    return (nll * keep).sum(1) / keep.sum(1).clamp(min=1)


def load_lm(rev: str, device, tiny: dict | None = None):
    if tiny:
        from transformers import AutoConfig
        mc = AutoConfig.from_pretrained(LM)
        for k, v in tiny.items():
            setattr(mc, k, v)
        mc.layer_types = mc.layer_types[:mc.num_hidden_layers]
        torch.manual_seed(6304)
        lm = AutoModelForCausalLM.from_config(mc, dtype=torch.float16)
    else:
        lm = AutoModelForCausalLM.from_pretrained(LM, revision=rev, dtype=torch.float16)
    lm.config.use_cache = False
    lm.requires_grad_(False)
    return lm.to(device)


def tokenizer(rev: str):
    return AutoTokenizer.from_pretrained(LM, revision=rev)
