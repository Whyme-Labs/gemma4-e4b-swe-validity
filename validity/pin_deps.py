"""Research-only sandbox fix for teacher-data tasks: after the harness installs test dependencies (newest cached wheel
of each package), reinstall every project dependency whose installed version violates the workspace's own
pyproject.toml constraint, using the highest satisfying wheel in the image's /wheels. Not scorer parity: the scorer
keeps the newest wheel (docs/experiments/e0-gold-verify.md, C3), so tasks valid only with this fix are training data,
never evaluation tasks.

  import pin_deps; pin_deps.install()   # before running swegemma
"""
import contextvars, os
from pathlib import Path

from swegemma import evaluate
from swegemma.evaluate import Evaluator
from swegemma.harness import agent_runner, container_setup, sample_verification, verification

TASK = contextvars.ContextVar("pin_task", default=None)  # the task being set up; install_test_dependencies is not told
# Tests that depend on a library's exact behaviour (Rich snapshots Pygments' highlighting): install the newest release
# by the time the task was created, as the project's CI would have. Wheels come from $PIN_EXTRA_WHEELS.
DATED = {"Textualize/rich": {"pygments": {
    "2.14.0": "2023-01-01", "2.15.0": "2023-04-10", "2.15.1": "2023-04-18", "2.16.0": "2023-08-06", "2.16.1": "2023-08-06",
    "2.17.0": "2023-11-18", "2.17.1": "2023-11-19", "2.17.2": "2023-11-21", "2.18.0": "2024-05-04", "2.19.0": "2025-01-05",
    "2.19.1": "2025-01-06", "2.19.2": "2025-06-21", "2.20.0": "2026-03-29", "2.21.0": "2026-08-17"}}}

NONET = '''
def pytest_collection_modifyitems(config, items):
    gone = [i for i in items if i.originalname in ("test_connect_timeout", "test_total_timeout_connect")
            or i.name == "test_redirecting_to_bad_url[http://localhost:-1-InvalidURL]"]
    if gone: config.hook.pytest_deselected(items=gone); items[:] = [i for i in items if i not in gone]
'''
FIX = r'''
import importlib.metadata as md, json, os, shutil, subprocess, sys, sysconfig, tomllib
try: from packaging.requirements import Requirement
except ImportError: from pip._vendor.packaging.requirements import Requirement
try: deps = tomllib.load(open("/workspace/pyproject.toml", "rb")).get("project", {}).get("dependencies", [])
except Exception: deps = []
bad = []
for d in deps:
    r = Requirement(d)
    if r.marker and not r.marker.evaluate(): continue
    try: v = md.version(r.name)
    except md.PackageNotFoundError: continue
    if r.specifier and not r.specifier.contains(v, prereleases=True): bad.append(f"{r.name}{r.specifier}")
def remove(name):  # wheels the harness unpacked have no usable RECORD, so pip cannot uninstall them: remove by hand
    try: dist = md.distribution(name)
    except md.PackageNotFoundError: return
    root = dist.locate_file("")
    for f in dist.files or []:
        p = root / f
        if p.is_file(): p.unlink()
    shutil.rmtree(root / name.replace("-", "_").lower(), ignore_errors=True)
    shutil.rmtree(dist._path, ignore_errors=True)
for spec in bad: remove(Requirement(spec).name)
pkg, shadow = "@PKG@", False  # a cached wheel of the repo's own package shadows /workspace, so tests never import the patch
try: shadow = not str(md.distribution(pkg).locate_file(pkg)).startswith("/workspace")
except md.PackageNotFoundError: pass
if shadow:
    remove(pkg); src = "/workspace/src" if os.path.isdir(f"/workspace/src/{pkg}") else "/workspace"
    open(f"{sysconfig.get_paths()['purelib']}/_workspace.pth", "w").write(src + "\n")
if pkg == "requests":  # tests that fail whatever the patch: connect timeouts (no network), port -1 (Python 3.13 urllib.parse)
    sp = sysconfig.get_paths()["purelib"]; os.makedirs(f"{sp}/pin_nonet-0.dist-info", exist_ok=True)
    open(f"{sp}/pin_nonet.py", "w").write(@NONET@)
    open(f"{sp}/pin_nonet-0.dist-info/METADATA", "w").write("Metadata-Version: 2.1\nName: pin-nonet\nVersion: 0\n")
    open(f"{sp}/pin_nonet-0.dist-info/entry_points.txt", "w").write("[pytest11]\npin_nonet = pin_nonet\n")
dated = "@DATED@".split()
for spec in dated: remove(spec.split("==")[0])
if dated: subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--no-index", "--find-links=/tmp/pin_wheels", *dated], check=True)
if bad: subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--no-index", "--find-links=/wheels", *bad], check=True)
print(json.dumps({"pinned": bad, "unshadowed": pkg if shadow else None, "dated": dated}))
'''


def _wrap(orig):
    def run(docker, container_id, repo="", **kw):
        orig(docker, container_id, repo, **kw)
        dated, created = "", (getattr(TASK.get(), "created_at", None) or "")[:10]
        for name, dates in DATED.get(repo, {}).items() if created and os.environ.get("PIN_EXTRA_WHEELS") else ():
            v = max((v for v, d in dates.items() if d <= created), key=lambda v: dates[v], default=None)
            whl = next(Path(os.environ["PIN_EXTRA_WHEELS"]).glob(f"[{name[0]}{name[0].upper()}]{name[1:]}-{v}-*.whl"), None) if v else None
            if whl:
                docker.exec(container_id, "mkdir -p /tmp/pin_wheels"); docker.copy_to(container_id, whl, f"/tmp/pin_wheels/{whl.name}")
                dated += f" {name}=={v}"
        fix = FIX.replace("@PKG@", repo.rsplit("/", 1)[-1].lower()).replace("@DATED@", dated).replace("@NONET@", repr(NONET))
        res = docker.exec(container_id, "python3 - <<'PINEOF'\n" + fix + "PINEOF")
        print("pin_deps", container_id[:12], "exit",  res.exit_code, res.stdout.strip()[-300:], res.stderr.strip()[-300:])
    return run


def install():
    orig_eval = Evaluator.evaluate_task
    async def evaluate_task(self, task, *a, **kw):
        TASK.set(task); return await orig_eval(self, task, *a, **kw)
    Evaluator.evaluate_task = evaluate_task
    f = _wrap(container_setup.install_test_dependencies)
    for m in (container_setup, verification, agent_runner, evaluate, sample_verification):
        if hasattr(m, "install_test_dependencies"): m.install_test_dependencies = f
