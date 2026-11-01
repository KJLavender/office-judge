# office-judge

**English** | [繁體中文](README.zh-TW.md)

Triage the email, chat messages and meeting notes employees write every day: route tickets, stop data leaks, extract action items. Built on **judge-core**, a small library that puts Jev (TypeSafe's System One model), an LLM, deterministic rules and a human review queue behind a single interface and runs them as a confidence-based cascade.

Plans: [PLANS.md](PLANS.md) · [00 shared core](00-shared-core.md) · [02 office-judge](02-office-judge-plan.md)

![Demo: an indirect leak is caught; the uncertain sentence walks rule → Kev-4B → Claude → human queue](docs/demo-indirect.png)

## Results (Traditional Chinese DLP, fictional company data)

| | Result |
|---|---|
| End to end (rules → Kev-4B → Claude → human), 100 held-out messages | **0 of 56 leaks missed**, accuracy 0.93, 42% of sentences needed the LLM |
| Kev-4B threshold tuned on one set, frozen, applied to a new set | accuracy 0.88 (vs 0.89 if re-tuned on the new set): the threshold transfers |
| Kev-4B latency on an RTX 3060 Ti | p50 62 ms per sentence, fast enough to check before the user hits send |
| Small encoder decision models (Julia-1, Laya EN / multilingual) | not usable on Traditional Chinese: AUC 0.66–0.80, and swings with prompt wording |

![AUC by model](docs/model-auc.png)

Findings that changed the design:
- **Thresholds are per model and per prompt wording.** Kev-4B's default 0.5 cut-off missed 42% of leaks while its ranking was near perfect (AUC 0.98). Changing a few words in the prompt shifted the probabilities enough to send every sentence to the LLM. Policies therefore carry `model_thresholds`, and the prompt is pinned to the wording that was validated.
- **Mask known formats before any model sees the text.** Taiwan ID / ARC / UBN with checksums, cards, phones, keys and codenames are caught by rules and replaced with reversible placeholders, so the judge models never receive them.
- Caveat: data and labels are LLM-synthesized. A human-labelled sample is being collected to check them.

## What it does

| Pipeline | Role | How |
|---|---|---|
| DLP | gating | Rules (Taiwan ID/ARC/UBN with checksums, cards, phones, API keys, project codenames) mask known formats first, then one Jev request judges every sentence. Outcomes: pass / mask (reversible) / block with a reason |
| Ticket routing | routing + ranking | One Jev request asks for the team (choice, team scopes in option descriptions) and urgency (score) |
| Action items | extraction | One Jev request asks, for every sentence, "is this an action item?" plus the owner (attendees + "unassigned"), with neighbouring sentences as context. The LLM only resolves deadlines for the sentences that were picked |

```
Rule ──hit──▶ decide
  ↓ miss
Jev (all questions in one request) ──confident──▶ decide
  ↓ gray zone
LLM (gray questions in one request) ──confident──▶ decide
  ↓
Human queue (gating stays blocked until reviewed)
```

Each layer is written to a SQLite audit log: input hash, every layer's answer and probability, model version, latency, cost.

## Quick start

Live demo in one command (Kev-4B + Claude + web page; `--offline` runs a keyword heuristic with no models):

```bash
./scripts/demo.sh
```

Or step by step:

```bash
uv sync --extra eval
uv run pytest
uv run uvicorn office_judge.app:app --reload   # http://127.0.0.1:8000
```

With no API key the demo runs on a keyword heuristic so you can try the whole flow. Backends are chosen from environment variables:

| Variable | Meaning |
|---|---|
| `TYPESAFE_API_KEY` | Use Jev as the primary judge |
| `TYPESAFE_BASE_URL` | Point at a [jev-rs](https://github.com/yijunyu/jev-rs) server for a fully local backend (same wire format) |
| `TYPESAFE_DEFAULT_MODEL` | Default `jev-1.14.0`; pin a version, since aliases move |
| `OFFICE_JUDGE_PRIMARY` | `jev` / `llm` / `demo` |
| `OFFICE_JUDGE_FALLBACK`, `OFFICE_JUDGE_WRITER` | `on` / `off`. LLM fallback layer, and LLM-written block reasons and deadlines. On by default when the `claude` CLI is installed |
| `JUDGE_LLM_MODEL` | Model for the local Claude Code CLI, default `sonnet` |

The LLM backend calls the local `claude -p` CLI (subscription login, no API key).

### Local decision models (llama.cpp)

`llama-server` serves `/v1/systemone` natively, so open decision models plug in through `TYPESAFE_BASE_URL`. On Traditional Chinese DLP data only **Kev-4B** was usable (validated: rules → Kev-4B → Claude missed 0 of 56 leaks end to end; see `00-shared-core.md` §8). Its thresholds live in `policies/dlp.yaml` under `model_thresholds`.

```bash
KEV_GPU=1 ONLY=kev-4b ./scripts/serve_local.sh &
TYPESAFE_BASE_URL=http://127.0.0.1:8195 TYPESAFE_DEFAULT_MODEL=kev-4b uv run uvicorn office_judge.app:app
```

## Evaluation

```bash
uv run python scripts/toy_eval.py                        # threshold sweep + calibration on 20 toy rows
uv run python scripts/toy_eval.py --backends jev llm --cascade   # compare backends, LLM rescue rate
uv run python scripts/jev_checks.py                      # day-one Jev checks: zh-TW, batching, option order
uv run python scripts/gen_data.py dlp --n 300            # synthetic labelled data (fictional company)
```

**LLM rescue rate** is the share of gray-zone items where the LLM overturned Jev and was right. Published results suggest Jev and LLM judges often fail on the same items, so measure this before relying on the LLM layer.

## Layout

```
src/judge_core/      Judge interface, Jev / LLM / Rule / Human / Fake backends, cascade, policies, audit, eval
src/office_judge/    detectors, text (split, reversible masking), pipelines, policies/*.yaml, company.yaml, web app
scripts/             toy_eval, jev_checks, gen_data
data/toy_dlp.jsonl   20 hand-labelled sentences
```

All company names, people and personal data are fictional.
