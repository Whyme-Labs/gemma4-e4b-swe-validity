"""Build SFT samples from a data-generation arm (run.py with E1_SAMPLES > 1 and E1_PROXY_LOG).

One sample per model call of a resolved attempt: the exact prompt token ids vLLM saw and the exact output token ids
it generated (proxy log), so training sees what serving sees, compaction included. Calls are mapped to their
attempt by proxy port (sample k = port - 9000) and to their task by the problem statement in the first user
message; sub-agent calls whose prompt does not name the task are skipped. Per task, at most MAX_PER_TASK resolved
attempts are kept, those with the fewest output tokens first (the budget rewards short solutions). Tasks in
--holdout go to the validation file only.

The teacher's chat template is not the student's (31B and E4B ship different chat_template.jinja), so with --student
the prompt is re-rendered from the logged request with the student's template, the way vLLM renders a chat request;
the output ids are kept (same tokenizer, checked). --parity first re-renders logged requests with the template of
the model that served them and reports how many match vLLM's prompt ids exactly: the renderer's own check.

  python build_sft.py ARM_DIR TASKS_JSONL OUT_DIR [--holdout id,id] [--max-per-task 4] [--empty-pass id,id]
                      [--student MODEL_DIR] [--parity ARM_DIR=MODEL_DIR ...]
"""
import argparse, collections, gzip, json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("arm"); ap.add_argument("tasks"); ap.add_argument("out")
ap.add_argument("--holdout", default=""); ap.add_argument("--max-per-task", type=int, default=4)
ap.add_argument("--empty-pass", default="", help="tasks any non-breaking patch resolves: never training data")
ap.add_argument("--student", default=""); ap.add_argument("--parity", nargs="*", default=[])
ap.add_argument("--min-parity", type=float, default=0.95)
ap.add_argument("--drop-failed-edits", action="store_true", help="skip calls whose edit_file the next request reports as failed")
a = ap.parse_args()


from chat_render import conversation, render


tokenizers = {}
def tokenizer(d):
    if d not in tokenizers:
        from transformers import AutoTokenizer
        tokenizers[d] = AutoTokenizer.from_pretrained(d)
    return tokenizers[d]


parity = {}
for spec in a.parity:  # ARM_DIR=MODEL_DIR: that arm's requests were served by that model
    arm_dir, model_dir = spec.split("=")
    n = same = 0; first_bad = None
    for line in gzip.open(Path(arm_dir) / "requests.jsonl.gz", "rt"):
        rec = json.loads(line); ids = rec["response"].get("prompt_token_ids")
        if not ids: continue
        got = render(tokenizer(model_dir), rec["request"]); n += 1; same += got == ids
        if got != ids and first_bad is None:
            i = next((j for j, (x, y) in enumerate(zip(got, ids)) if x != y), min(len(got), len(ids)))
            tk = tokenizer(model_dir)
            first_bad = {"at": i, "len": [len(got), len(ids)], "ours": tk.decode(got[max(0, i - 20): i + 20]),
                         "vllm": tk.decode(ids[max(0, i - 20): i + 20])}
        if n >= 300: break
    parity[spec] = {"checked": n, "exact": same, "first_mismatch": first_bad}
    print(json.dumps({"parity": spec, **parity[spec]}), flush=True)
    assert n and same / n >= a.min_parity, f"renderer parity {same}/{n} below {a.min_parity} for {spec}"
if a.student and a.parity:
    t_files = {Path(s.split("=")[1]) / "tokenizer.json" for s in a.parity} | {Path(a.student) / "tokenizer.json"}
    assert len({f.read_bytes() for f in t_files}) == 1, f"tokenizers differ: {sorted(map(str, t_files))}"
arm, out = Path(a.arm), Path(a.out); out.mkdir(parents=True, exist_ok=True)
holdout = set(filter(None, a.holdout.split(","))); empty_pass = set(filter(None, a.empty_pass.split(",")))

problems = {}
for l in open(a.tasks):
    t = json.loads(l); problems[t["instance_id"]] = t["problem_statement"].strip()[:300]
resolved = {(r["id"], r.get("sample", 0)) for r in map(json.loads, open(arm / "tasks.jsonl")) if r["resolved"]}


def task_of(req):
    users = [m for m in req["messages"] if m.get("role") == "user"]
    text = users[0]["content"] if users else ""
    if isinstance(text, list): text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
    hits = [i for i, p in problems.items() if p and p in text]
    return hits[0] if len(hits) == 1 else None


calls = collections.defaultdict(list)  # (task, sample) -> [(t, prompt_ids, completion_ids, finish)]
skipped = collections.Counter()
with gzip.open(arm / "requests.jsonl.gz", "rt") as f:
    for line in f:
        try: rec = json.loads(line)
        except ValueError: skipped["bad_line"] += 1; continue
        resp = rec["response"]
        if rec["status"] != 200 or not resp.get("prompt_token_ids"): skipped["no_ids"] += 1; continue
        task = task_of(rec["request"])
        if task is None: skipped["no_task"] += 1; continue
        k = rec["port"] - 9000 if rec["port"] >= 9000 else 0
        ch = resp["choices"][0]
        req = rec["request"] if a.student and (task, k) in resolved else None  # rendered only if kept
        msgs = rec["request"]["messages"]; j = max((i for i, m in enumerate(msgs) if m.get("role") == "assistant"), default=len(msgs))
        results = " ".join(str(m.get("content")) for m in msgs[j + 1:] if m.get("role") == "tool")  # the previous call's tool results
        edits = any(tc["function"]["name"] == "edit_file" for tc in (ch.get("message") or {}).get("tool_calls") or [])
        calls[(task, k)].append((rec["t"], resp["prompt_token_ids"], ch["token_ids"], ch.get("finish_reason"), req, edits, results))

attempts = collections.defaultdict(list)  # task -> [(output tokens, sample)]
for (task, k), cs in calls.items():
    if (task, k) in resolved and task not in empty_pass:
        attempts[task].append((sum(len(c[2]) for c in cs), k))
stats = {"student": a.student or None, "parity": parity, "skipped": dict(skipped), "attempts_logged": len(calls), "resolved_attempts": sum(map(len, attempts.values())),
         "tasks": {}}
with gzip.open(out / "train.jsonl.gz", "wt") as tr, gzip.open(out / "val.jsonl.gz", "wt") as va:
    for task, att in sorted(attempts.items()):
        keep = sorted(att)[: a.max_per_task]
        stats["tasks"][task] = {"resolved": len(att), "kept": [k for _, k in keep], "holdout": task in holdout}
        for _, k in keep:
            cs = sorted(calls[(task, k)], key=lambda c: c[0])
            for i, (_, p, c, fin, req, edits, _) in enumerate(cs):
                if a.drop_failed_edits and edits and i + 1 < len(cs) and "FileEditError" in cs[i + 1][6]:
                    stats["dropped_failed_edits"] = stats.get("dropped_failed_edits", 0) + 1; continue
                if a.student: p = render(tokenizer(a.student), req)
                (va if task in holdout else tr).write(json.dumps(
                    {"task": task, "sample": k, "i": i, "finish": fin, "prompt_ids": p, "completion_ids": c}) + "\n")
stats["train_tasks"] = sum(not v["holdout"] for v in stats["tasks"].values())
(out / "stats.json").write_text(json.dumps(stats, indent=1))
print(json.dumps({k: v for k, v in stats.items() if k != "tasks"}))
