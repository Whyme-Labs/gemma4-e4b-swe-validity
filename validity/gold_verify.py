"""E0: per-task sandbox setup time and verifier ground truth on the public tasks, CPU only, organizer harness.

For each task (subprocess sandbox as in the organizer notebook, or --sandbox docker as in the harness default):
  setup_s   the agent-phase sandbox setup (the same steps as swegemma.harness.agent_runner, timed; it counts against
            the 12 h scoring budget)
  empty     verify_task with no patch: must NOT resolve (the task is a real failure)
  gold      verify_task with the reference patch: must resolve (the verifier works here)
Rows go to <out>/rows.jsonl as each task finishes (resumable); a summary to <out>/summary.json.

  python local_lab/gold_verify.py --data <competition data dir> --out runs/e0 [--limit N] [--ids a b] [--sandbox docker]
"""
import argparse, asyncio, json, os, time
from pathlib import Path

from swegemma.config import EvalConfig
from swegemma.evaluate import Evaluator
from swegemma.harness import container_setup as cs
from swegemma.harness.verification import verify_task
from swegemma.models import load_tasks
from swegemma.sandbox.base import sandbox_start, sandbox_stop


async def setup_seconds(ev, task, snap):
    """The agent sandbox's setup steps, in agent_runner's order."""
    sb, cfg = ev.docker, ev.config
    t = time.perf_counter(); sid = await sandbox_start(sb)
    try:
        await asyncio.to_thread(cs.setup_container_wheels, sb, sid, cfg)
        await asyncio.to_thread(cs.extract_snapshot, sb, sid, snap)
        await asyncio.to_thread(cs.setup_git_exclude, sb, sid)
        await asyncio.to_thread(cs.install_editable_package, sb, sid)
        await asyncio.to_thread(cs.install_test_dependencies, sb, sid, task.repo, config=cfg)
        await asyncio.to_thread(cs.setup_workspace_test_config, sb, sid, repo=task.repo)
        await asyncio.to_thread(cs.setup_baseline_commit, sb, sid, "baseline")
        return round(time.perf_counter() - t, 2)
    finally:
        await sandbox_stop(sb, sid)


async def verify(ev, task, snap, patch):
    t = time.perf_counter()
    r = await verify_task(ev.docker, ev.config, task, snap, agent_patch=patch, start_time=t)
    return {"resolved": bool(r.resolved), "exit": r.test_exit_code, "error": (r.error or "")[:500],
            "seconds": round(time.perf_counter() - t, 2)}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int); ap.add_argument("--ids", nargs="*")
    ap.add_argument("--sandbox", choices=("subprocess", "docker"), default="subprocess")
    ap.add_argument("--wheels-dir", type=Path, help="host wheel cache the harness injects (default <data>/wheels)")
    ap.add_argument("--pin", action="store_true", help="local_lab/pin_deps.py: honour each task's own dependency pins")
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    if a.pin:
        import pin_deps; pin_deps.install()
    cfg = EvalConfig(tasks_path=a.data / "tasks.jsonl", snapshots_dir=a.data / "snapshots", results_dir=a.out / "results",
                     submission_dir=a.data / "sample_submission", models=None, sandbox=a.sandbox,
                     wheels_dir=a.wheels_dir or a.data / "wheels", graph_dir=str(a.data / "graphs"),
                     embeddings_dir=str(a.data / "embeddings"), display_mode="quiet")
    ev = Evaluator(cfg)
    tasks = load_tasks(cfg.tasks_path)
    if a.ids: tasks = [t for t in tasks if t.instance_id in set(a.ids)]
    if a.limit: tasks = tasks[:a.limit]
    rows_p = a.out / "rows.jsonl"
    done = {json.loads(l)["id"] for l in rows_p.read_text().splitlines()} if rows_p.exists() else set()
    for i, task in enumerate(tasks, 1):
        if task.instance_id in done: continue
        snap = next((p for p in (cfg.snapshots_dir / f"{task.instance_id}.tgz", cfg.snapshots_dir / f"{task.instance_id}.tar.gz") if p.exists()), None)
        row = {"id": task.instance_id, "repo": task.repo}
        try:
            if a.pin: pin_deps.TASK.set(task)
            row["setup_s"] = await setup_seconds(ev, task, snap)
            row["empty"] = await verify(ev, task, snap, "")
            row["gold"] = await verify(ev, task, snap, task.patch)
        except Exception as e:  # recorded as infrastructure, never as pass/fail
            row["infra_error"] = repr(e)[:1000]
        with open(rows_p, "a") as f: f.write(json.dumps(row) + "\n")
        print(i, len(tasks), json.dumps(row)[:300], flush=True)
    rows = [json.loads(l) for l in rows_p.read_text().splitlines()]
    good = [r for r in rows if "infra_error" not in r]
    s = sorted(r["setup_s"] for r in good)
    summary = {"schema": "e0-gold-verify/1", "tasks": len(rows), "infraErrors": len(rows) - len(good),
               "goldResolved": sum(r["gold"]["resolved"] for r in good),
               "emptyResolved": sum(r["empty"]["resolved"] for r in good),
               "valid": sum(r["gold"]["resolved"] and not r["empty"]["resolved"] for r in good),
               "setupSeconds": {"median": s[len(s) // 2] if s else None, "p90": s[int(len(s) * .9)] if s else None,
                                "total": round(sum(s), 1)},
               "goldVerifySecondsMedian": sorted(r["gold"]["seconds"] for r in good)[len(good) // 2] if good else None}
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1)); print(json.dumps(summary))


if __name__ == "__main__":
    asyncio.run(main())
