# chunked_sdpa (sft_lora.py) against transformers' SDPA: outputs and gradients, head dim 512, GQA, causal and masked. CPU.
import torch, types
from torch.utils.checkpoint import checkpoint
from transformers.integrations.sdpa_attention import sdpa_attention_forward
src = open(__file__.replace("test_chunked_attn.py", "sft_lora.py")).read()
src = src[src.index("def chunked_sdpa"):src.index('AttentionInterface.register("sdpa"')]
g = {"torch": torch, "checkpoint": checkpoint, "sdpa_attention_forward": sdpa_attention_forward}; exec(src, g)
torch.manual_seed(0)
mod = types.SimpleNamespace(num_key_value_groups=4, is_causal=True)
for T, S, masked in [(2500, 2500, False), (2500, 2500, True), (1500, 1700, False)]:
    q = torch.randn(1, 8, T, 512, requires_grad=True); k = torch.randn(1, 2, S, 512, requires_grad=True); v = torch.randn(1, 2, S, 512, requires_grad=True)
    mask = (torch.rand(1, 1, T, S) > 0.3) | torch.eye(T, S, dtype=torch.bool)[None, None] if masked else None
    ref, _ = sdpa_attention_forward(mod, q, k, v, mask, scaling=0.05)
    got, _ = g["chunked_sdpa"](mod, q, k, v, mask, scaling=0.05)
    w = torch.randn_like(ref)
    gr = torch.autograd.grad((ref * w).sum(), (q, k, v)); gg = torch.autograd.grad((got * w).sum(), (q, k, v))
    print(T, S, masked, "out", (ref - got).abs().max().item(), "grads", max((a - b).abs().max().item() for a, b in zip(gr, gg)))
    assert torch.allclose(ref, got, atol=1e-5) and all(torch.allclose(a, b, atol=1e-4) for a, b in zip(gr, gg))
print("ok")
