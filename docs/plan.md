# indic-rl-transfer Implementation Plan

> Execute task by task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Reward functions, MGSM eval harness, GRPO training script and a Saturn Cloud smoke-test plan for the EN-vs-Indic RL transfer study.

**Architecture:** Small Python package `indic_rl` (pure functions: answer parsing, rewards, script-based language ID, data loading) tested locally on CPU. Two thin GPU scripts (`eval_mgsm.py`, `train_grpo.py`) import it and run on Saturn. Configs in YAML so both arms differ only by dataset.

**Tech Stack:** Python 3.11, TRL `GRPOTrainer`, Unsloth, vLLM (eval + generation), `datasets`, pytest, PyYAML.

**Spec:** `docs/design.md`

## Global Constraints

- Free Saturn Cloud hours only; no paid instance types. Nothing launches without the user's OK.
- HF account: `dhairyapandya` (currently logged in). Tokens only from env / `hf auth login`; never printed, logged or committed.
- Public/synthetic data only: GSM8K (`openai/gsm8k`), MGSM (`juletxara/mgsm`), `ai4bharat/indic-gsm8k` (bn/te/en columns) minus the 250 MGSM items, for both arms.
- Eval languages: en, bn, te, sw. Train languages Arm B: bn, te.
- Arms A and B: identical model, steps, group size, prompt count, reward.
- Public repo: plain commit messages.

## Review Focus

1. Answers written in Bengali (০-৯) or Telugu (౦-౯) digits must parse to the same number as ASCII. Test in Task 1.
2. Answers with commas/decimals ("1,234", "12.0", "$5") match gold "1234"/"12"/"5". Test in Task 1.
3. Completion with no parseable number gets reward 0, never crashes. Test in Task 1.
4. Train set must not contain MGSM test questions (MGSM is translated from GSM8K **test**; train uses GSM8K **train**). Assert in Task 3.
5. Empty or pure-number reasoning returns language ratio 0.0 and doesn't divide by zero. Test in Task 2.

---

### Task 1: Repo skeleton, answer parsing and rewards

**Files:**
- Create: `pyproject.toml`, `.gitignore` (`.env`, `outputs/`, `wandb/`, `*.safetensors`), `README.md` (one paragraph + status), `src/indic_rl/__init__.py`, `src/indic_rl/rewards.py`
- Test: `tests/test_rewards.py`

**Interfaces:**
- Produces:
  - `normalize_digits(s: str) -> str` (Bengali/Telugu/Devanagari digits to ASCII)
  - `extract_answer(text: str) -> float | None` (number after `####` or inside `\boxed{}`; else last number)
  - `correctness_reward(completions: list[str], answer: list[str], **kw) -> list[float]` (1.0 match within 1e-6, else 0.0; TRL reward signature)
  - `format_reward(completions: list[str], **kw) -> list[float]` (0.2 if completion ends with `#### <number>`, else 0.0)

- [ ] **Step 1: `git init`, write skeleton files.**
- [ ] **Step 2: Write failing tests**

```python
def test_bengali_digits(): assert extract_answer("উত্তর #### ১২৩") == 123
def test_telugu_digits(): assert extract_answer("#### ౪౫") == 45
def test_commas_dollars(): assert extract_answer("so $1,234.") == 1234
def test_boxed(): assert extract_answer(r"\boxed{12.0}") == 12
def test_no_number(): assert correctness_reward(["no idea"], ["5"]) == [0.0]
def test_correct(): assert correctness_reward(["... #### 5"], ["5"]) == [1.0]
def test_format(): assert format_reward(["x #### 7", "7"]) == [0.2, 0.0]
```

- [ ] **Step 3: Run `pytest tests/test_rewards.py -v`.** Expected: FAIL (import error).
- [ ] **Step 4: Implement the four functions in `src/indic_rl/rewards.py`.**
- [ ] **Step 5: Run `pytest tests/test_rewards.py -v`.** Expected: 7 passed.
- [ ] **Step 6: Commit** `feat: answer parsing and verifiable rewards`

### Task 2: Script-based language ID and consistency reward

**Files:**
- Create: `src/indic_rl/langid.py`
- Test: `tests/test_langid.py`

**Interfaces:**
- Produces:
  - `SCRIPT_RANGES = {"bn": (0x0980, 0x09FF), "te": (0x0C00, 0x0C7F), "hi": (0x0900, 0x097F)}`
  - `script_ratio(text: str, lang: str) -> float` (letters in that script / all letters; 0.0 if no letters; `"en"` counts ASCII letters)
  - `english_ratio(text: str) -> float` (ASCII-letter share)
  - `consistency_reward(completions: list[str], lang: list[str], **kw) -> list[float]` (`0.3 * script_ratio`; for `en`/`sw` returns 0.0, so this signal applies to Indic prompts only)
- Builder call: script ratio, not fastText. Bengali and Telugu have their own scripts, so this is exact and has no dependencies. Swahili drift is not measured (it shares the Latin script with English); this is noted as a limitation in the report.

- [ ] **Step 1: Write failing tests**

```python
def test_pure_bengali(): assert script_ratio("আমি ভাত খাই", "bn") == 1.0
def test_mixed(): assert 0.4 < script_ratio("আমি eat", "bn") < 0.7
def test_empty(): assert script_ratio("", "te") == 0.0 and script_ratio("123", "te") == 0.0
def test_english_ratio(): assert english_ratio("so the answer") == 1.0
def test_consistency_en_zero(): assert consistency_reward(["hello"], ["en"]) == [0.0]
```

- [ ] **Step 2: Run `pytest tests/test_langid.py -v`.** Expected: FAIL.
- [ ] **Step 3: Implement in `src/indic_rl/langid.py`.**
- [ ] **Step 4: Run tests.** Expected: 5 passed.
- [ ] **Step 5: Commit** `feat: script-based language id and consistency reward`

### Task 3: Data loading and prompts

**Files:**
- Create: `src/indic_rl/data.py`
- Test: `tests/test_data.py` (network tests marked `@pytest.mark.net`)

**Interfaces:**
- Consumes: none.
- Produces:
  - `SYSTEM_PROMPT = "Solve the problem step by step in the same language as the question. End with '#### <number>'."`
  - `build_prompt(question: str) -> list[dict]` (chat messages: system + user)
  - `load_train(arm: str, n: int, seed: int = 0) -> datasets.Dataset` (`arm` in `{"en", "indic"}`; columns `prompt`, `answer`, `lang`; `en` = English questions; `indic` = each problem in bn or te, seeded 50/50; `n<=1069`)
  - `load_mgsm(lang: str, limit: int | None = None) -> datasets.Dataset` (columns `prompt`, `answer`, `lang`, `question`)
  - `assert_no_overlap(train: Dataset, test_questions: list[str]) -> None` (raises `ValueError` on any exact-match normalised question)

- [ ] **Step 1: Write tests:** `build_prompt` returns 2 messages with the system role first; `assert_no_overlap` raises on a planted duplicate (offline); `@net` test: `load_train("indic", 20)` has 10 bn + 10 te rows and `load_mgsm("bn", 5)` has 5 rows with numeric answers.
- [ ] **Step 2: Run `pytest tests/test_data.py -v -m "not net"`.** Expected: FAIL.
- [ ] **Step 3: Implement.** Confirm the MGSM8KInstruct column names by loading it once; if bn/te are missing there, stop and report before substituting another source.
- [ ] **Step 4: Run `pytest tests/test_data.py -v`** (including net). Expected: all pass.
- [ ] **Step 5: Commit** `feat: dataset loaders and contamination check`

### Task 4: MGSM eval harness

**Files:**
- Create: `scripts/eval_mgsm.py`, `src/indic_rl/evaluate.py`
- Test: `tests/test_evaluate.py`

**Interfaces:**
- Consumes: `load_mgsm`, `extract_answer`, `script_ratio`, `english_ratio`.
- Produces:
  - `score_rows(rows: list[dict]) -> dict` (input rows have `lang`, `answer`, `completion`; returns per-lang `{"acc", "n", "en_reasoning_share", "native_script_share"}`; `en_reasoning_share` = fraction of completions with `english_ratio > 0.5`)
  - CLI: `python scripts/eval_mgsm.py --model <hf_id> [--adapter <path>] --langs en bn te sw --limit 250 --seeds 0 1 2 --temperature 0.0|0.6 --out outputs/eval/<name>`. Writes `completions.jsonl` and `summary.json`. Uses vLLM with LoRA when `--adapter` is given; `max_tokens=1024`.

- [ ] **Step 1: Test `score_rows` on 4 hand-made rows** (2 bn with 1 correct and 1 English-reasoning; 2 en both correct). Expect bn acc 0.5 with en_reasoning_share 0.5, and en acc 1.0.
- [ ] **Step 2: Run the test.** Expected: FAIL.
- [ ] **Step 3: Implement `score_rows`, then the CLI.**
- [ ] **Step 4: Run `pytest -v`.** Expected: all pass. Run `python scripts/eval_mgsm.py --help`; it should exit 0 (vLLM import is lazy).
- [ ] **Step 5: Commit** `feat: MGSM eval harness`

### Task 5: GRPO training script

**Files:**
- Create: `scripts/train_grpo.py`, `configs/base.yaml`, `configs/arm_en.yaml`, `configs/arm_indic.yaml`, `configs/smoke.yaml`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: `load_train`, `correctness_reward`, `format_reward`, `consistency_reward`.
- Produces:
  - `load_config(*paths: str) -> dict` (later files override earlier ones, shallow merge)
  - `base.yaml`: `model: unsloth/Qwen2.5-1.5B-Instruct`, `lora_r: 32`, `lr: 5e-6`, `num_generations: 8`, `max_prompt_len: 384`, `max_completion_len: 768`, `steps: 400`, `n_train: 3200`, `per_device_batch: 8`, `grad_accum: 4`, `use_consistency: false`, `save_every: 100`, `seed: 0`
  - `arm_en.yaml`: `arm: en`. `arm_indic.yaml`: `arm: indic`. `smoke.yaml`: `steps: 20`, `n_train: 160`, `save_every: 20`
  - CLI: `python scripts/train_grpo.py configs/base.yaml configs/arm_en.yaml [configs/smoke.yaml] --out outputs/runs/<name> [--push-to-hub <repo>]`. Writes `step_times.json` (seconds/step) and the final adapter.
- Builder calls: Unsloth `FastLanguageModel` + `fast_inference=True` (vLLM) when available; the model is swapped to Gemma 3 1B only if the step-0 baseline says so.

- [ ] **Step 1: Test `load_config`:** merging base + smoke gives `steps == 20` and keeps `lr == 5e-6`; arm_en + arm_indic configs differ only in `arm`.
- [ ] **Step 2: Run the test.** Expected: FAIL.
- [ ] **Step 3: Implement `load_config` and the training script.**
- [ ] **Step 4: Run `pytest -v`.** Expected: all pass. Run `python scripts/train_grpo.py --help`; it should exit 0.
- [ ] **Step 5: Commit** `feat: GRPO training script and configs`

### Task 6: Saturn Cloud setup and smoke-test plan

**Files:**
- Create: `saturn/setup.sh` (pip install pinned requirements, `pip install -e .`), `requirements.txt` (pinned trl, unsloth, vllm, datasets, peft), `docs/smoke_test.md`
- Modify: `README.md` (how to run)

- [ ] **Step 1: Check which free GPU types the Saturn account offers** (read-only, through its API or UI with `SATURN_TOKEN` from env). Record them in `docs/smoke_test.md`.
- [ ] **Step 2: Write `docs/smoke_test.md`** with three gated stages, each needing the user's OK before launch:
  1. Baseline: `eval_mgsm.py` on Qwen2.5-1.5B and Gemma-3-1B, all 4 langs, limit 250, greedy (est. ≤1 GPU h). Decision rule: pick the model with higher mean bn/te accuracy; if both <8%, go up to a 3B model.
  2. Smoke train: `smoke.yaml` for each arm (20 steps) plus a 32-item eval. Pass when the reward rises above 0, there's no OOM and `step_times.json` exists.
  3. Sizing: full run hours = 400 × sec/step / 3600 per arm. Proceed if both arms + evals ≤ 60 GPU h (keeps >50% headroom).
- [ ] **Step 3: Run `bash -n saturn/setup.sh` and `pytest -v`.** Expected: both clean.
- [ ] **Step 4: Commit** `docs: Saturn smoke-test plan`
