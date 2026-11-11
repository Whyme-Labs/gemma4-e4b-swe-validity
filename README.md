# Gemma 4 E4B SWE agent: verifier validity and distillation code

Code for the Kaggle Gemma 4 Developer Agent paper-track writeup *Measuring Before Training: Verifier Validity, Replay
and Breadth in Distilling a 31B Gemma 4 Agent into E4B*. It runs on top of the organizers' harness (`swegemma`
0.2.11) and competition data, neither of which is included. No competition task data is included.

Docstrings refer to paths in our private working repository (`docs/experiments/...`, `local_lab/...`,
`aws_lab/...`); the files here are those, unchanged apart from one example path.

| Path | What it is | Paper section |
|---|---|---|
| `validity/gold_verify.py` | Per task: sandbox setup time, verifier with an empty patch and with the reference patch. A task is valid if gold resolves and empty does not. `--sandbox docker`, `--wheels-dir`, `--pin`. | 3.1 |
| `validity/pin_deps.py` | Research-only pinned environment: reinstall dependencies that violate the workspace's own `pyproject.toml`, unshadow the package under test, date Pygments for Rich, deselect three environment-broken Requests tests. Patches the harness's `install_test_dependencies`. Not scorer parity. | 3.3 |
| `wheels/sandbox-extra-wheels.txt` | Hash-pinned wheels missing from the organizers' cache (our scorer-parity setup "C3"). | 3.1 |
| `wheels/sandbox-pin-wheels.txt`, `wheels/pin-extra-wheels.txt` | Extra hash-pinned wheels for the pinned environment; Pygments releases for dating. | 3.3 |
| `runner/run.py` | Runs the organizers' sample submission on chosen tasks with declared overrides (model, budget, sampling, adapter, prompt, samples, pinned environment) and writes per-task rows. | 2, 4, 5 |
| `runner/proxy.py` | Logging proxy between the harness and vLLM: every chat completion with prompt and output token ids. | 4.1 |
| `distill/build_sft.py`, `distill/chat_render.py` | SFT samples from the teacher's resolved attempts: exact token ids, prompts re-rendered with the student's chat template, gated by a renderer parity check against vLLM's logged prompt ids. | 4.1 |
| `distill/sft_lora.py` | LoRA SFT for Gemma 4 E4B on 4x L4, plain torch: LoRA on every attention and MLP projection, logits only at output positions (262k vocabulary), query-chunked attention for the 512-dimension global heads; writes a harness-servable adapter. `torchrun --nproc_per_node 4`. | 4.1 |
| `distill/test_chunked_attn.py` | Equivalence check of the chunked attention against SDPA. | 4.1 |
| `distill/copy_fidelity.py` | Logits-level copy test: teacher-forced `edit_file` calls scored through vLLM prompt logprobs, JSON-escaped against raw tool output. | 4.3 |

Typical use, with the competition data in `$DATA` and the harness installed:

```
python validity/gold_verify.py --data $DATA --out out/c3 --sandbox docker --wheels-dir $WHEELS
PYTHONPATH=validity PIN_EXTRA_WHEELS=$PYGMENTS_WHEELS python validity/gold_verify.py --data $DATA --out out/pin --sandbox docker --wheels-dir $WHEELS_PIN --pin
python distill/build_sft.py TEACHER_RUN_DIR $DATA/tasks.jsonl out/sft --holdout ID,ID --max-per-task 4 --student E4B_DIR --parity TEACHER_RUN_DIR=TEACHER_MODEL_DIR
torchrun --nproc_per_node 4 distill/sft_lora.py --model E4B_DIR --data out/sft --out out/adapter --lr 1e-4 --max-len 16384 --max-completion 4096
```

Licence: Apache 2.0.
