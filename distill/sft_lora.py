"""LoRA SFT of a Gemma 4 QAT checkpoint on build_sft.py samples; writes a PEFT adapter the harness can serve.

Plain torch, no PEFT: every language-model attention and MLP projection gets y = W x + (alpha / r) * B A x, with W
frozen as loaded (compressed-tensors; --decompress loads it as bfloat16 instead). Data parallel over all visible GPUs
(torchrun), one sample per micro-step, LoRA gradients averaged across ranks. Logits are computed only at the output
positions (`logits_to_keep`), so memory does not scale with the 262k vocabulary times the prompt length.

The adapter keys follow the organizer sample (`base_model.model.<module path>.lora_A.weight`).

  torchrun --nproc_per_node 4 sft_lora.py --model DIR --data DIR --out ADAPTER_DIR [--rank 16 --alpha 32 ...]
"""
import argparse, gzip, json, math, os, random, time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file
from torch.utils.checkpoint import checkpoint

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--rank", type=int, default=16); ap.add_argument("--alpha", type=float, default=32)
ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--epochs", type=float, default=2)
ap.add_argument("--grad-accum", type=int, default=8); ap.add_argument("--max-len", type=int, default=24576)
ap.add_argument("--max-completion", type=int, default=4096); ap.add_argument("--warmup", type=int, default=10)
ap.add_argument("--max-steps", type=int, default=0, help="stop after this many optimizer steps (smoke); 0 = all")
ap.add_argument("--decompress", action="store_true"); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

GPU = torch.cuda.is_available()  # CPU (gloo) only for the local logic test on a tiny model
dist.init_process_group("nccl" if GPU else "gloo"); rank, world = dist.get_rank(), dist.get_world_size()
dev = torch.device("cuda", int(os.environ.get("LOCAL_RANK", rank))) if GPU else torch.device("cpu")
if GPU: torch.cuda.set_device(dev)
mem = lambda f: round(f(dev) / 2**30, 2) if GPU else 0
log = lambda **kw: rank == 0 and print(json.dumps(kw), flush=True)

from transformers import AttentionInterface, AutoModelForCausalLM
from transformers.integrations.sdpa_attention import sdpa_attention_forward


def chunked_sdpa(module, query, key, value, attention_mask, **kw):
    # query [b, h, T, d]. Head dims above 256 (E4B's global layers: 512) have no fused SDPA kernel, so the math fallback
    # materialises [h, T, S] scores and, in backward, their gradients: about 7 GB at 12k tokens. Those layers run in
    # query chunks instead, each recomputed in backward.
    T, S = query.shape[2], key.shape[2]
    if query.shape[-1] <= 256 or T <= 1024:
        return sdpa_attention_forward(module, query, key, value, attention_mask, **kw)
    causal = kw.pop("is_causal", None); causal = getattr(module, "is_causal", True) if causal is None else causal
    outs = []
    for i in range(0, T, 1024):
        j = min(T, i + 1024)
        if attention_mask is not None:
            m = attention_mask[..., i:j, :] if attention_mask.shape[-2] == T else attention_mask
        elif causal:  # as SDPA's is_causal: query t sees keys up to t
            m = (torch.arange(S, device=query.device)[None, :] <= torch.arange(i, j, device=query.device)[:, None])[None, None]
        else:
            m = None
        f = lambda q, m=m: sdpa_attention_forward(module, q, key, value, m, is_causal=False, **kw)[0]
        q = query[:, :, i:j]
        outs.append(checkpoint(f, q, use_reentrant=False) if torch.is_grad_enabled() else f(q))
    return torch.cat(outs, dim=1), None


AttentionInterface.register("sdpa", chunked_sdpa)
t0 = time.time()
if a.decompress: from transformers import CompressedTensorsConfig
model = AutoModelForCausalLM.from_pretrained(
    a.model, dtype=torch.bfloat16, device_map={"": dev}, attn_implementation="sdpa",
    **({"quantization_config": CompressedTensorsConfig(run_compressed=False)} if a.decompress else {}))
model.config.use_cache = False
for p in model.parameters(): p.requires_grad_(False)


class OnHost(nn.Module):  # a frozen lookup table kept in host memory; only the rows looked up go to the GPU
    def __init__(self, m):
        super().__init__(); self.m = m.cpu()

    def forward(self, ids):
        return self.m(ids.cpu()).to(dev)


if GPU:  # E4B's per-layer embedding table (262k x 42 x 256, about 5.6 GB in bf16) is only ever indexed
    for name, m in list(model.named_modules()):
        if name.endswith("embed_tokens_per_layer"):
            setattr(model.get_submodule(name.rsplit(".", 1)[0]), name.rsplit(".", 1)[1], OnHost(m)); torch.cuda.empty_cache()


class LoRA(nn.Module):
    def __init__(self, base, d_in, d_out):
        super().__init__()
        self.base, self.scale = base, a.alpha / a.rank
        self.A = nn.Parameter(torch.empty(a.rank, d_in, device=dev, dtype=torch.float32))
        self.B = nn.Parameter(torch.zeros(d_out, a.rank, device=dev, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))  # PEFT's default init: B = 0, so training starts at W

    def forward(self, x):
        # In the activation dtype: an fp32 copy of a long sequence's MLP input does not fit next to the model on an L4.
        # The parameters stay fp32 (AdamW state); their gradients come back through the casts.
        return self.base(x) + F.linear(F.linear(x, self.A.to(x.dtype)), self.B.to(x.dtype)) * self.scale


def dims(m):  # compressed-tensors linears keep the logical shape in weight_shape; plain ones in weight
    ws = getattr(m, "weight_shape", None)
    if ws is not None: return int(ws[1]), int(ws[0])
    return m.in_features, m.out_features


wrapped = {}
for name, m in list(model.named_modules()):
    if ("layers." in name and name.split(".")[-1] in TARGETS and not any(x in name for x in ("vision", "audio"))
            and not isinstance(m, LoRA)):
        parent = model.get_submodule(name.rsplit(".", 1)[0])
        d_in, d_out = dims(m)
        setattr(parent, name.rsplit(".", 1)[1], lora := LoRA(m, d_in, d_out)); wrapped[name] = lora
assert wrapped, "no target modules found"
torch.manual_seed(a.seed)
for l in wrapped.values(): nn.init.kaiming_uniform_(l.A, a=math.sqrt(5))  # same init on every rank
params = [p for l in wrapped.values() for p in (l.A, l.B)]
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); model.train()
log(event="loaded", seconds=round(time.time() - t0, 1), attn=model.config._attn_implementation, wrapped=len(wrapped), lora_params=sum(p.numel() for p in params),
    mem_gib=mem(torch.cuda.memory_allocated), world=world)


def load(path):
    return [s for s in map(json.loads, gzip.open(path, "rt"))
            if len(s["prompt_ids"]) + len(s["completion_ids"]) <= a.max_len and 0 < len(s["completion_ids"]) <= a.max_completion]


train, val = load(Path(a.data) / "train.jsonl.gz"), load(Path(a.data) / "val.jsonl.gz")
log(event="data", train=len(train), val=len(val), train_tokens=sum(len(s["completion_ids"]) for s in train))


# The model's forward returns final hidden states (lm_head and soft-cap moved into chunk_loss). The top-level forward is
# kept: calling the inner decoder directly skips compressed-tensors' setup ("'Linear' object has no attribute 'weight'").
LM_HEAD, CAP = model.lm_head, model.config.get_text_config().final_logit_softcapping
model.lm_head = nn.Identity(); model.config.get_text_config().final_logit_softcapping = None


def chunk_loss(h, tgt):  # summed cross-entropy of one chunk of output positions, with the model's logit soft-cap
    logits = LM_HEAD(h).float()
    if CAP: logits = torch.tanh(logits / CAP) * CAP
    return F.cross_entropy(logits, tgt, reduction="sum")


def loss_of(s):
    # The 262k-vocabulary logits of a whole output (and their soft-cap and fp32 copies) do not fit next to the model on
    # an L4, so they are made 512 positions at a time, each chunk recomputed in backward (only its hidden states kept).
    ids = torch.tensor(s["prompt_ids"] + s["completion_ids"][:-1], device=dev)[None]
    tgt = torch.tensor(s["completion_ids"], device=dev)
    h = model(input_ids=ids, logits_to_keep=len(tgt)).logits[0]
    loss = 0
    for i in range(0, len(tgt), 512):
        args = (h[i:i + 512], tgt[i:i + 512])
        loss = loss + (checkpoint(chunk_loss, *args, use_reentrant=False) if torch.is_grad_enabled() else chunk_loss(*args))
    return loss, len(tgt)


@torch.no_grad()
def val_nll():  # mean nats per output token over the held-out tasks, shared across ranks
    tot = torch.zeros(2, device=dev)
    for s in val[rank::world]:
        l, n = loss_of(s); tot += torch.tensor([l.item(), n], device=dev)
    dist.all_reduce(tot)
    return (tot[0] / tot[1].clamp(min=1)).item()


opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
per_rank = len(train) // world
total = int(a.epochs * per_rank) // a.grad_accum
if a.max_steps: total = min(total, a.max_steps)
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: min(1, (s + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(1, s / max(1, total)))))
log(event="schedule", optimizer_steps=total, micro_per_rank=per_rank, val_nll=val_nll() if val else None)

step, micro, t1, acc = 0, 0, time.time(), torch.zeros(2, device=dev)
rng = random.Random(a.seed)
while step < total:
    order = list(range(len(train))); rng.shuffle(order)  # same order on every rank; each takes its slice
    for j in order[rank::world][:per_rank]:
        l, n = loss_of(train[j]); (l / n / a.grad_accum).backward(); acc += torch.tensor([l.item(), n], device=dev)
        micro += 1
        if micro % a.grad_accum: continue
        for p in params:  # a module no sample reached still takes part in the reduce
            if p.grad is None: p.grad = torch.zeros_like(p)
            dist.all_reduce(p.grad); p.grad /= world
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
        dist.all_reduce(acc)
        log(event="step", step=step, loss=round((acc[0] / acc[1]).item(), 4), grad_norm=round(gn, 3),
            lr=sched.get_last_lr()[0], tok_s=round(acc[1].item() / (time.time() - t1), 1),
            mem_gib=mem(torch.cuda.max_memory_allocated))
        acc.zero_(); t1 = time.time()
        if step >= total: break
log(event="trained", steps=step, val_nll=val_nll() if val else None, minutes=round((time.time() - t0) / 60, 1))

if rank == 0:
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    from safetensors import safe_open
    ckpt = {k.rsplit(".", 1)[0] for f in Path(a.model).glob("*.safetensors") for k in safe_open(f, "pt").keys()}

    def ckpt_name(name):  # vLLM maps adapter keys onto the checkpoint's module paths, which may differ from HF's
        for c in (name, name.replace("model.", "model.language_model.", 1)):
            if c in ckpt: return c
        raise AssertionError(f"{name} has no weight in the checkpoint")
    tensors = {}
    for name, l in wrapped.items():
        tensors[f"base_model.model.{ckpt_name(name)}.lora_A.weight"] = l.A.detach().to(torch.bfloat16).cpu().contiguous()
        tensors[f"base_model.model.{ckpt_name(name)}.lora_B.weight"] = l.B.detach().to(torch.bfloat16).cpu().contiguous()
    save_file(tensors, out / "adapter_model.safetensors")
    layers = sorted({int(n.split("layers.")[1].split(".")[0]) for n in wrapped})
    (out / "adapter_config.json").write_text(json.dumps({
        "peft_type": "LORA", "task_type": "CAUSAL_LM", "base_model_name_or_path": Path(a.model).name, "r": a.rank,
        "lora_alpha": a.alpha, "lora_dropout": 0.0, "bias": "none",
        "target_modules": sorted({n.split(".")[-1] for n in wrapped}), "layers_to_transform": layers,
        "init_lora_weights": True, "inference_mode": True, "fan_in_fan_out": False, "use_rslora": False,
        "use_dora": False, "modules_to_save": None}, indent=1))
    (out / "train_args.json").write_text(json.dumps(vars(a), indent=1))
    log(event="saved", out=str(out), tensors=len(tensors))
dist.destroy_process_group()
