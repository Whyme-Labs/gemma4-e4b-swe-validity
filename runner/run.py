"""E1 on AWS: the organizer sample submission on the 129 public tasks (docs/experiments/e1-sample-baseline.md).

Mirrors the 2026-10-13 starter notebook (docs/sources/starter-notebook-2026-10-21.ipynb), cells 2, 4 and 5, with
these changes only: all tasks instead of tasks[:2], paths from the environment instead of /kaggle, the docker sandbox
by default (see E1 v2's amendment), results written incrementally, a wall-clock guard, and identity logging. The wheelhouse install (cell 1) is done by bootstrap.sh.

  E1_MODEL       model to run; empty = the one sample_submission declares (31B, untouched). Any other allowed model
                 gets the notebook's rewrite: every `model:` replaced, `adapter:` lines and adapters/ removed.
  E1_TASK_LIMIT  first N tasks only (smoke); 0 = all
  E1_SANDBOX     docker (default; the harness default, image swebench-sandbox:latest) or subprocess (the notebook's)
  E1_EVAL_OVERRIDE  JSON merged into eval_config.yaml's `evaluation:` after the untouched check (a declared budget arm);
                 empty = the sample's own budget
  E1_SAMPLING_OVERRIDE  JSON merged into configs/sampling.yaml (one level deep, so thinking_config keys merge)
  E1_TASK_IDS    comma-separated task ids to run (file order kept); empty = all
  E1_STRIP_ADAPTERS  1 = remove `adapter:` lines and adapters/ with the model unchanged (the sample's are no-ops)
  E1_ADAPTER_DIR a PEFT adapter dir: after the model rewrite, copied to adapters/<dir name> and set as `adapter:` on
                 every agent (a trained or smoke adapter); empty = none added
  E1_PROMPT_FILE file copied over prompts/system.md after the untouched check (a declared prompt arm); empty = the sample's
  E1_SAMPLES     attempts per task (default 1); E1_CONCURRENCY  attempts in flight at once against one vLLM (default 1).
                 Either above 1 runs attempts concurrently, one Evaluator (results dir) per sample; 1 and 1 is the
                 sequential starter loop.
  E1_PROXY_LOG   path: put aws_lab/e1/proxy.py between the harness and vLLM and log every chat completion with its
                 prompt and output token ids (training-data provenance); "auto" = <E1_OUT>/requests.jsonl.gz; empty = no
                 proxy. Sample k talks to the proxy on port 9000+k (8000 with one sample), logged as "port".
  E1_DATA        competition data dir;  E1_MODELS  dir holding <model-name>/ weight dirs;  E1_OUT  output dir
  E1_PIN         1 = research-only teacher-data environment: /opt/pin_deps.py (local_lab/pin_deps.py) patched into the
                 harness and the host wheel cache E1_WHEELS_DIR (bootstrap.sh's /data/comp/wheels-pin). Not scorer
                 parity: tasks valid only this way are training data, never evaluation (docs/experiments/e0-gold-verify.md)
"""
import asyncio, concurrent.futures, hashlib, json, os, re, shutil, subprocess, sys, time
from pathlib import Path

T0 = time.time()
os.environ.update({
    "TRANSFORMERS_NO_TF": "1", "TRANSFORMERS_NO_FLAX": "1", "TRANSFORMERS_OFFLINE": "1", "VLLM_NO_USAGE_STATS": "1",
    "VLLM_WORKER_MULTIPROC_METHOD": "spawn", "VLLM_ENGINE_READY_TIMEOUT_S": "1200",
    "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS": "1", "NCCL_DEBUG": "WARN", "LITELLM_LOCAL_MODEL_COST_MAP": "True",
    "LITELLM_TELEMETRY": "False", "LITELLM_LOG": "WARNING", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "OTEL_SDK_DISABLED": "true", "OTEL_PYTHON_DISABLED": "true"})
os.environ.setdefault("OPENAI_API_KEY", "EMPTY"); os.environ.setdefault("LOCAL_API_KEY", "EMPTY")

DATA_DIR = Path(os.environ.get("E1_DATA", "/data/comp"))
MODELS_DIR = Path(os.environ.get("E1_MODELS", "/data/models"))
OUT = Path(os.environ.get("E1_OUT", "/data/out/e1")); OUT.mkdir(parents=True, exist_ok=True)
TASK_LIMIT = int(os.environ.get("E1_TASK_LIMIT", "0")) or None
WALL_GUARD_S = float(os.environ.get("E1_WALL_GUARD_H", "11")) * 3600
REQUESTED_MODEL = os.environ.get("E1_MODEL", "")
SANDBOX = os.environ.get("E1_SANDBOX", "docker")
EVAL_OVERRIDE = json.loads(os.environ.get("E1_EVAL_OVERRIDE") or "{}")
SAMPLING_OVERRIDE = json.loads(os.environ.get("E1_SAMPLING_OVERRIDE") or "{}")
TASK_IDS = [t for t in os.environ.get("E1_TASK_IDS", "").split(",") if t]
PROMPT_FILE = os.environ.get("E1_PROMPT_FILE", "")
STRIP_ADAPTERS = os.environ.get("E1_STRIP_ADAPTERS", "") == "1"
SAMPLES = int(os.environ.get("E1_SAMPLES") or 1); CONCURRENCY = int(os.environ.get("E1_CONCURRENCY") or 1)
PROXY_LOG = os.environ.get("E1_PROXY_LOG", "")
if PROXY_LOG == "auto": PROXY_LOG = str(OUT / "requests.jsonl.gz")
ADAPTER_DIR = os.environ.get("E1_ADAPTER_DIR", "")

# 2026-10-13 sample_submission, from docs/organizer-package-2026-10-13.md. A mismatch means the run is not "untouched".
EXPECTED_SHA256 = {
    "agent.yaml": "c7fbbbbdc44778419be8e9f53a85800eeb17d0cf151e0846c46e107401b72034",
    "eval_config.yaml": "adb486b67535fa59b427b79d44dafc131176f1f1964d184e16046ed520ede02d",
    "configs/sampling.yaml": "3dab0b2506dae34ce92fef7b380d6073729104b23fe2cf66d44ac1d2f213aa6a",
    "prompts/system.md": "f1cbf7943ee9687382321b1dfd9cd88581d608c1c8b8e6ea61a763bd1a3234af",
    "sub_agents/code_analyzer.yaml": "7a764798e36cc68aa38900256245e4aed969b5439b992c7da4105b6ebcb7e62b",
}
ALLOWED_MODEL_NAMES = frozenset({"gemma-4-31b-it-qat-w4a16-ct", "gemma-4-12b-it-qat-w4a16-ct",
                                 "gemma-4-e4b-it-qat-w4a16-ct", "gemma-4-e2b-it-qat-w4a16-ct"})
MODEL_VLLM_EXTRA_ARGS = {"gemma-4-12b-it-qat-w4a16-ct": (
    "--hf-overrides", '{"architectures": ["Gemma4ForCausalLM"], "model_type": "gemma4"}')}
_SAFE_PATH_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def log(event, **kw):
    kw.update(event=event, t=round(time.time() - T0, 1))
    with open(OUT / "events.jsonl", "a") as f:
        f.write(json.dumps(kw, default=str) + "\n")
    print(json.dumps(kw, default=str), flush=True)


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# Identity: what actually ran
import importlib.metadata as md
import torch
log("identity", python=sys.version.split()[0],
    versions={p: md.version(p) for p in ("vllm", "torch", "transformers", "swegemma", "adk-submission", "google-adk",
                                         "litellm", "compressed-tensors", "flashinfer-python")},
    cuda=torch.version.cuda, gpus=torch.cuda.device_count(),
    names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    mem_gib=[round(torch.cuda.get_device_properties(i).total_memory / 2**30, 1) for i in range(torch.cuda.device_count())],
    cpus=os.cpu_count(),
    driver=subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.split())

# Cell 2: data and sample submission
from swegemma.models import load_tasks
src = DATA_DIR / "sample_submission"
got = {k: sha256(src / k) for k in EXPECTED_SHA256}
log("sample_submission", untouched=got == EXPECTED_SHA256, mismatched=[k for k in got if got[k] != EXPECTED_SHA256[k]],
    all_files={str(p.relative_to(src)): sha256(p) for p in sorted(src.rglob("*")) if p.is_file()})
assert got == EXPECTED_SHA256, "sample_submission differs from the 2026-10-13 package; re-declare E1 before running"

AGENT_DIR = OUT / "sample_submission"
if AGENT_DIR.exists(): shutil.rmtree(AGENT_DIR)
shutil.copytree(src, AGENT_DIR)
NEW_MODEL = REQUESTED_MODEL and REQUESTED_MODEL != "gemma-4-31b-it-qat-w4a16-ct"
if NEW_MODEL or STRIP_ADAPTERS:  # the notebook's rewrite, verbatim logic; STRIP_ADAPTERS keeps the model
    for yaml_path in [AGENT_DIR / "agent.yaml", *sorted((AGENT_DIR / "sub_agents").glob("*.yaml"))]:
        if yaml_path.is_file():
            text = yaml_path.read_text(encoding="utf-8")
            if NEW_MODEL: text = re.sub(r"^(\s*model:\s*).*$", rf"\g<1>{REQUESTED_MODEL}", text, flags=re.MULTILINE)
            text = re.sub(r"^\s*adapter:\s*.*\n?", "", text, flags=re.MULTILINE)
            yaml_path.write_text(text, encoding="utf-8")
    shutil.rmtree(AGENT_DIR / "adapters", ignore_errors=True)
if ADAPTER_DIR:
    name = Path(ADAPTER_DIR).name
    assert _SAFE_PATH_COMPONENT.fullmatch(name), name
    shutil.copytree(ADAPTER_DIR, AGENT_DIR / "adapters" / name)
    for yaml_path in [AGENT_DIR / "agent.yaml", *sorted((AGENT_DIR / "sub_agents").glob("*.yaml"))]:
        text = yaml_path.read_text(encoding="utf-8")
        assert not re.search(r"^\s*adapter:", text, flags=re.MULTILINE), f"{yaml_path} already has an adapter"
        yaml_path.write_text(re.sub(r"^(\s*)(model:.*)$", rf"\g<1>\g<2>\n\g<1>adapter: {name}", text, count=1,
                                    flags=re.MULTILINE), encoding="utf-8")
    log("adapter_added", name=name, files={p.name: sha256(p) for p in sorted((AGENT_DIR / "adapters" / name).iterdir())})
if EVAL_OVERRIDE:  # written into the submission copy, so the harness and a shipped submission see the same budget
    import yaml
    ec = AGENT_DIR / "eval_config.yaml"; d = yaml.safe_load(ec.read_text(encoding="utf-8"))
    d["evaluation"].update(EVAL_OVERRIDE); ec.write_text(yaml.safe_dump(d, sort_keys=False), encoding="utf-8")
    log("eval_override", override=EVAL_OVERRIDE, sha256=sha256(ec))
if SAMPLING_OVERRIDE:  # configs/sampling.yaml is !include'd by agent.yaml and sub_agents/code_analyzer.yaml
    import yaml
    sc = AGENT_DIR / "configs" / "sampling.yaml"; d = yaml.safe_load(sc.read_text(encoding="utf-8"))
    for k, v in SAMPLING_OVERRIDE.items():
        d[k] = {**d[k], **v} if isinstance(v, dict) and isinstance(d.get(k), dict) else v
    sc.write_text(yaml.safe_dump(d, sort_keys=False), encoding="utf-8")
    log("sampling_override", override=SAMPLING_OVERRIDE, result=d, sha256=sha256(sc))

if PROMPT_FILE:  # agent.yaml !include's prompts/system.md; the sub-agent's analyzer.md is left as shipped
    shutil.copyfile(PROMPT_FILE, AGENT_DIR / "prompts" / "system.md")
    log("prompt_override", source=PROMPT_FILE, sha256=sha256(AGENT_DIR / "prompts" / "system.md"))

TASKS_PATH = DATA_DIR / "tasks.jsonl"
log("tasks_file", sha256=sha256(TASKS_PATH))
tasks = load_tasks(TASKS_PATH)
if TASK_IDS:
    missing = set(TASK_IDS) - {t.instance_id for t in tasks}
    assert not missing, f"unknown task ids: {sorted(missing)}"
    tasks = [t for t in tasks if t.instance_id in set(TASK_IDS)]
if TASK_LIMIT: tasks = tasks[:TASK_LIMIT]
log("tasks", n=len(tasks))

# Cell 4: vLLM server
import litellm
from adk_submission import VllmConfig, VllmServer, discover_adapters
from adk_submission.discovery import discover_declared_models
from swegemma.config import ALLOWED_ADAPTER_EXTENSIONS
litellm.drop_params = True

declared_models = discover_declared_models(AGENT_DIR)
assert len(declared_models) == 1, f"Expected exactly 1 declared model, got: {sorted(declared_models)}"
declared_model = next(iter(declared_models))
assert declared_model in ALLOWED_MODEL_NAMES, f"Unsupported model {declared_model!r}"
MODEL_PATH = MODELS_DIR / declared_model
assert MODEL_PATH.exists(), f"Model weights not found at {MODEL_PATH}"
log("model", name=declared_model, path=str(MODEL_PATH),
    files={p.name: p.stat().st_size for p in sorted(MODEL_PATH.iterdir())},
    config_sha256=sha256(MODEL_PATH / "config.json"))

adapters = discover_adapters(str(AGENT_DIR), adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS)
adapter_map = getattr(adapters, "adapters", None) or {}
for adapter_name in adapter_map:
    assert _SAFE_PATH_COMPONENT.fullmatch(adapter_name), f"Unsafe adapter name {adapter_name!r}"
detected_ranks = [r for info in adapter_map.values() if (r := getattr(info, "rank", None)) is not None and r > 0]
gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 1
tp_size = 4 if gpu_count >= 4 else (2 if gpu_count >= 2 else 1)
vllm_cfg = VllmConfig(
    model=str(MODEL_PATH), port=8001 if PROXY_LOG else 8000, host="127.0.0.1", tool_call_parser="gemma4", reasoning_parser="gemma4",
    max_model_len=32768, dtype="bfloat16" if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else "auto",
    gpu_memory_utilization=0.90, enable_auto_tool_choice=True, enable_lora=bool(adapter_map),
    max_loras=max(1, len(adapter_map)), max_lora_rank=max(detected_ranks) if detected_ranks else 128,
    scheduler_reserve_full_isl=False, tensor_parallel_size=tp_size, startup_timeout=60 * 20,
    extra_args=MODEL_VLLM_EXTRA_ARGS.get(declared_model, ()))
t = time.time(); server_instance = VllmServer(vllm_cfg, adapter_manifest=adapters); server_instance.start()
log("vllm_started", tp=tp_size, adapters=sorted(adapter_map), seconds=round(time.time() - t, 1))
if PROXY_LOG:  # vLLM moved to 8001; the harness talks to the proxy (8000, or 9000+k per sample)
    ports = [8000] if SAMPLES == 1 else [9000 + k for k in range(SAMPLES)]
    proxy = subprocess.Popen([sys.executable, str(Path(__file__).with_name("proxy.py")),
                              f"{ports[0]}-{ports[-1]}", "8001", PROXY_LOG])
    import atexit  # also on a crash: the log gets its gzip trailer and the ports are free for the next arm
    atexit.register(lambda: (proxy.terminate(), proxy.wait(30)))
    import urllib.request
    for _ in range(60):
        try: urllib.request.urlopen(f"http://127.0.0.1:{ports[-1]}/health", timeout=2); break
        except Exception: time.sleep(1)
    else: raise RuntimeError("proxy did not come up")
    log("proxy_started", log=PROXY_LOG, ports=ports)
def registry(port):  # the registry's api_base follows config.port
    server_instance.config.port = port
    return server_instance.create_model_registry(aliases=[declared_model], model_prefix="openai/",
                                                 api_key=os.environ.get("LOCAL_API_KEY", "EMPTY"))
models = registry(8000)

# Cell 5: phase 1 + phase 2, sequential as in the starter
import yaml
from google.adk.agents.context_cache_config import ContextCacheConfig
from google.adk.apps._configs import EventsCompactionConfig
from swegemma.config import EvalConfig, build_submission_limits
from swegemma.evaluate import Evaluator
if os.environ.get("E1_PIN") == "1":
    sys.path.insert(0, "/opt"); import pin_deps; pin_deps.install()


def run_sync(fn, *args, **kwargs):
    try: loop = asyncio.get_running_loop()
    except RuntimeError: loop = None
    call = lambda: asyncio.run(fn(*args, **kwargs))
    if loop is not None and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(call).result()
    return call()


eval_config_file = next((c for n in ("eval_config.yaml", "eval_config.yml") if (c := AGENT_DIR / n).is_file()), None)
raw = (yaml.safe_load(eval_config_file.read_text(encoding="utf-8")) or {}) if eval_config_file else {}
ev = raw.get("evaluation", raw)
turns_raw = ev.get("max_turns", ev.get("max_llm_calls"))
limits, gen_constraints = build_submission_limits()
config = EvalConfig(
    tasks_path=TASKS_PATH, snapshots_dir=DATA_DIR / "snapshots", results_dir=OUT / "results", submission_dir=AGENT_DIR,
    models=models, sandbox=SANDBOX, timeout_seconds=int(ev.get("timeout_seconds", 300)),
    verification_timeout_seconds=int(ev.get("verification_timeout_seconds", 300)),
    max_time_minutes=float(ev.get("max_time_minutes", 60.0)), max_tool_calls=int(ev.get("max_tool_calls", 100)),
    max_turns=int(turns_raw) if turns_raw is not None else None, limits=limits, generation_constraints=gen_constraints,
    adapter_manifest=adapters,
    context_cache_config=ContextCacheConfig(min_tokens=2048, ttl_seconds=1800, cache_intervals=10),
    events_compaction_config=EventsCompactionConfig(compaction_interval=15, overlap_size=2, token_threshold=14336,
                                                    event_retention_size=5),
    graph_dir=str(DATA_DIR / "graphs"), embeddings_dir=str(DATA_DIR / "embeddings"), wheels_dir=Path(os.environ.get("E1_WHEELS_DIR") or DATA_DIR / "wheels"),
    verbose=False)
# The harness caches the unpacked wheels as $TMPDIR/swegemma_sp_cache_v8/sp_base.tar, keyed by name only, so a later
# run on the same box reuses an earlier run's wheels (E4-G2: sft4's C3 eval got the pinned arm's site-packages and
# its verifier passed tasks whatever the patch). Key the temp dir on the wheel set.
import tempfile
wheel_key = hashlib.sha256("\n".join(f"{w.name} {w.stat().st_size}" for w in sorted(config.wheels_dir.glob("*.whl"))).encode()).hexdigest()[:12]
tempfile.tempdir = f"/tmp/e1-wheels-{wheel_key}"; os.makedirs(tempfile.tempdir, exist_ok=True)
log("eval_config", sandbox=SANDBOX, pin=os.environ.get("E1_PIN") == "1", wheelsDir=str(config.wheels_dir), wheelKey=wheel_key, **{k: ev.get(k) for k in ("timeout_seconds", "max_tool_calls", "max_time_minutes", "max_turns")})
import dataclasses
evaluators = [Evaluator(config if SAMPLES == 1 else dataclasses.replace(
    config, results_dir=OUT / "results" / f"s{k}", **({"models": registry(9000 + k)} if PROXY_LOG else {})))
    for k in range(SAMPLES)]
log("schedule", samples=SAMPLES, concurrency=CONCURRENCY)

rows = []
(OUT / "patches").mkdir(exist_ok=True)


def record(task, k, r, t, err=None):
    if err is None:
        patch = (r.agent_patch or "").strip()
        (OUT / "patches" / (f"{task.instance_id}.diff" if SAMPLES == 1 else f"{task.instance_id}.s{k}.diff")).write_text(patch)
        g = lambda *names: next((getattr(r, n) for n in names if getattr(r, n, None) is not None), None)
        row = {"id": task.instance_id, "repo": task.repo, "resolved": bool(r.resolved), "exit": g("test_exit_code"),
               "status": g("status"), "tool_calls": g("tool_calls"), "llm_calls": g("total_llm_calls"),
               "tokens": g("total_tokens"), "duration_s": g("duration_seconds"), "patch_chars": len(patch),
               "error": g("error_message", "error"), "wall_s": round(time.time() - t, 1)}
    else:  # recorded, never counted as a pass or a fail
        row = {"id": task.instance_id, "repo": task.repo, "resolved": None, "error": repr(err)[:2000],
               "wall_s": round(time.time() - t, 1)}
    if SAMPLES > 1: row["sample"] = k
    rows.append(row)
    with open(OUT / "tasks.jsonl", "a") as f:
        f.write(json.dumps(row, default=str) + "\n")
    print(f"[{len(rows)}/{len(tasks) * SAMPLES}] {task.instance_id} s{k} resolved={row['resolved']} wall={row['wall_s']}s", flush=True)


if SAMPLES == 1 and CONCURRENCY == 1:
    for i, task in enumerate(tasks, 1):
        if time.time() - T0 > WALL_GUARD_S:
            log("wall_guard_stop", done=len(rows)); break
        t = time.time()
        try: record(task, 0, run_sync(evaluators[0].evaluate_task, task=task, task_index=i, total_tasks=len(tasks)), t)
        except Exception as e: record(task, 0, None, t, e)
else:
    async def attempts():
        sem = asyncio.Semaphore(CONCURRENCY)
        async def one(k, i, task):  # sample-major order, so early samples finish first
            async with sem:
                if time.time() - T0 > WALL_GUARD_S: return
                t = time.time()
                try: record(task, k, await evaluators[k].evaluate_task(task=task, task_index=i, total_tasks=len(tasks)), t)
                except Exception as e: record(task, k, None, t, e)
        await asyncio.gather(*(one(k, i, task) for k in range(SAMPLES) for i, task in enumerate(tasks, 1)))
    asyncio.run(attempts())
    if time.time() - T0 > WALL_GUARD_S: log("wall_guard_stop", done=len(rows))

ok = [r for r in rows if r["resolved"] is not None]
by_repo = {}
for r in ok:
    b = by_repo.setdefault(r["repo"], [0, 0]); b[0] += r["resolved"]; b[1] += 1
summary = {"schema": "e1-summary/2", "model": declared_model, "evalOverride": EVAL_OVERRIDE,
           "samplingOverride": SAMPLING_OVERRIDE,
           "stripAdapters": STRIP_ADAPTERS, "promptSha256": sha256(AGENT_DIR / "prompts" / "system.md"), "taskIds": TASK_IDS or None, "complete": len(rows) == len(tasks) * SAMPLES and len(ok) == len(rows),
           "samples": SAMPLES, "concurrency": CONCURRENCY, "tasks": len(tasks), "ran": len(rows), "errors": len(rows) - len(ok),
           "resolved": sum(r["resolved"] for r in ok), "resolvedRate": round(sum(r["resolved"] for r in ok) / max(1, len(ok)), 4),
           "byRepo": by_repo, "wallHours": round((time.time() - T0) / 3600, 3),
           "taskWallMedianS": sorted(r["wall_s"] for r in rows)[len(rows) // 2] if rows else None}
(OUT / "summary.json").write_text(json.dumps(summary, indent=1))
log("done", **summary)
server_instance.stop()
