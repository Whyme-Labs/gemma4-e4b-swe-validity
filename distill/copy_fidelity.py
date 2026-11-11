"""Copy fidelity at the logits layer: can the served model write an edit_file old_string that exactly matches a file it
has just read?

From logged agent requests (proxy log) whose last message is a read_file response, pick a random span of 1-6 lines of
the content just read (non-blank, unique in it), and teacher-force the native tool call
  <|tool_call>call:edit_file{filepath:..,new_string:<|"|># TODO<|"|>,old_string:<|"|>SPAN<|"|>}<tool_call|>
after a short thought. vLLM returns prompt logprobs (the serving kernels and adapter), and for every old_string token we
record whether it was the argmax and its logprob. A span is "copied" when every one of its tokens is the argmax: what
greedy decoding would write. Conditions:
  escaped: the tool response as the harness sends it (the file inside a JSON string: \\n, \\", one line);
  raw:     the same response with the file content as plain text (counterfactual; the harness cannot do this).

  python copy_fidelity.py REQUESTS_GZ MODEL_DIR URL OUT_JSON --models e4b=MODEL_NAME sft1=sft1 [--n 150]
"""
import argparse, collections, gzip, json, random, re, urllib.request

from transformers import AutoTokenizer

from chat_render import conversation, content_format

ap = argparse.ArgumentParser()
ap.add_argument("requests"); ap.add_argument("model_dir"); ap.add_argument("url"); ap.add_argument("out")
ap.add_argument("--models", nargs="+", required=True, help="label=served model name")
ap.add_argument("--n", type=int, default=150); ap.add_argument("--max-prompt", type=int, default=16000)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained(a.model_dir); fmt = content_format(tok); rng = random.Random(a.seed)


def cases():
    seen = set()
    for line in gzip.open(a.requests, "rt"):
        req = json.loads(line)["request"]; last = req["messages"][-1]
        if last.get("role") != "tool": continue
        c = last.get("content"); c = c if isinstance(c, str) else "".join(p.get("text", "") for p in c or [])
        try: d = json.loads(c)
        except ValueError: continue
        if not isinstance(d, dict) or "content" not in d or "start_line" not in d: continue
        lines = d["content"].split("\n"); k = rng.randint(1, 6)
        starts = [i for i in range(len(lines) - k + 1) if all(l.strip() for l in lines[i:i + k])]
        rng.shuffle(starts)
        span = next((s for s in ("\n".join(lines[i:i + k]) for i in starts) if d["content"].count(s) == 1 and len(s) > 10), None)
        if span is None or (d.get("filepath"), span) in seen: continue
        seen.add((d.get("filepath"), span))
        yield req, d, span


def render(req, d, span, raw):
    msgs = [dict(m) for m in req["messages"]]
    if raw:
        meta = {k: v for k, v in d.items() if k != "content"}
        msgs[-1]["content"] = json.dumps(meta) + "\ncontent:\n" + d["content"]
    prompt = tok.apply_chat_template(conversation({**req, "messages": msgs}, fmt), tools=req.get("tools"),
                                     add_generation_prompt=True, tokenize=False, **(req.get("chat_template_kwargs") or {}))
    q = '<|"|>'
    head = (f"<|channel>thought\nI'll make the edit.\n<channel|><|tool_call>call:edit_file{{filepath:{q}{d.get('filepath')}{q},"
            f"new_string:{q}# TODO{q},old_string:{q}")
    p_ids = tok(prompt + head, add_special_tokens=False)["input_ids"]
    enc = tok(span + q + "}<tool_call|>", add_special_tokens=False, return_offsets_mapping=True)
    n_span = sum(1 for s, e in enc["offset_mapping"] if s < len(span))
    return p_ids, enc["input_ids"], n_span


def score(model, ids, start, n):
    body = json.dumps({"model": model, "prompt": ids, "max_tokens": 1, "temperature": 0, "prompt_logprobs": 1}).encode()
    r = json.load(urllib.request.urlopen(urllib.request.Request(a.url + "/v1/completions", body, {"Content-Type": "application/json"}), timeout=600))
    pl = r["choices"][0]["prompt_logprobs"]
    out = []
    for i in range(start, start + n):
        d = pl[i]; t = str(ids[i]); me = d[t]
        top = min(d.items(), key=lambda kv: kv[1]["rank"])
        out.append({"ok": me["rank"] == 1, "lp": me["logprob"], "tok": me.get("decoded_token"), "argmax": top[1].get("decoded_token")})
    return out


def kind(t):
    if t is None: return "?"
    if t.strip() == "": return "whitespace" + ("+newline" if "\n" in t else "")
    if "\\" in t or '"' in t or "'" in t: return "quote/backslash"
    return "other"


rows = []; done = 0
for req, d, span in cases():
    for raw in (False, True):
        p, c, n = render(req, d, span, raw)
        if len(p) + len(c) > a.max_prompt: break
        for label_model in a.models:
            label, model = label_model.split("=")
            toks = score(model, p + c, len(p), n)
            rows.append({"model": label, "cond": "raw" if raw else "escaped", "lines": span.count("\n") + 1,
                         "copied": all(t["ok"] for t in toks), "acc": sum(t["ok"] for t in toks) / n, "nll": -sum(t["lp"] for t in toks) / n,
                         "errors": [[kind(t["tok"]), t["tok"], t["argmax"]] for t in toks if not t["ok"]][:5]})
    else:
        done += 1
        if done >= a.n: break

summary = {}
groups = collections.defaultdict(list)
for r in rows: groups[(r["model"], r["cond"])].append(r)
for (m, cond), g in sorted(groups.items()):
    errs = collections.Counter(e[0] for r in g for e in r["errors"])
    by_lines = {k: round(sum(r["copied"] for r in g if r["lines"] == k) / max(1, sum(r["lines"] == k for r in g)), 3) for k in range(1, 7)}
    summary[f"{m}/{cond}"] = {"n": len(g), "copied": round(sum(r["copied"] for r in g) / len(g), 3),
                              "token_acc": round(sum(r["acc"] for r in g) / len(g), 4), "nll": round(sum(r["nll"] for r in g) / len(g), 4),
                              "copied_by_lines": by_lines, "first_errors": dict(errs)}
print(json.dumps(summary, indent=1))
json.dump({"summary": summary, "rows": rows}, open(a.out, "w"))
