# FOURFOLD

Reference implementation for **"Beyond Mutational Distance: Structure-Constrained Navigation of
Protein Fitness Landscapes."**

> Method code only — no trained weights, no anchor pools, no assay data. Nothing needed to
> reproduce the paper is missing. The two ensembles retrain **bitwise-identically** from the public
> assay files at the seeds given below, and the search reproduces the reported behaviour from any
> unfiltered sample of 100 measured starting sequences — which is what the paper's pools are. More
> anchors, or a longer schedule, should only improve on it. Weights will be released separately as
> a convenience, not as a dependency.

---

## Method

FOURFOLD edits a protein one accepted state at a time. Each round asks four questions of the
**current** sequence, not of the sequence it started from:

| | | |
|---|---|---|
| **WHERE** | which positions to reopen | mask each position in turn and rank sites by `R_i = p_0 − p_g`, the drop in the language model's support for the residue actually there once fitness guidance is applied. Keep the top `m_t`. |
| **WHAT** | what to put there | nine ESM3 completions over those sites with the incumbent residues forbidden; then, after WHICH, resample **every** position the unmasked model supports below 0.70 — anywhere in the sequence, inside the mask or outside it |
| **WHICH** | which candidate to carry forward | `argmin σ_s(c)` — the one the five-member scorer ensemble disagrees on least |
| **WHETHER** | may it become the next state | only if its mean ESMFold Cα-pLDDT reaches 0.80; otherwise retry from the unchanged sequence, up to 20 batches |

Because `R_i` is computed from the current sequence, mutation priorities move as the sequence
moves. Because refinement can touch positions outside the mask, the mask width does not bound how
much of the sequence changes: WHERE decides where an edit *starts*, not how far it reaches.

The search runs at most 15 rounds — 5 at mask width 10, then 10 at width 5. Only accepted
proposals update the state. **The evaluation oracle is never read during search.**

> pLDDT is ESMFold's confidence in its own prediction, not experimental evidence of folding.
> Every fitness number this code produces is an oracle prediction.

### Two predictor ensembles

The search needs fitness signal on two different kinds of input, which is why there are two
ensembles. Both are five-member MLPs over the flattened one-hot sequence, hidden widths (800, 400),
GELU, with per-member output normalisation before pooling.

| | **guide** | **scorer** |
|---|---|---|
| Input | one-hot **+ 4 reveal-time features** | one-hot |
| Scores | partially masked sequences | complete sequences |
| Supplies | `μ_g`, `v_g` — the guidance field and its gradient | `σ_s` for WHICH, `μ_s` for the refinement target |
| Member | `MaskedTailPredictor` | `CompleteSequenceScorer` |
| Ensemble | `GuideEnsemble` | `ScorerEnsemble` |

The four reveal-time features are `(r, r², sin πr, cos πr)` where `r` is the unmasked fraction.
They are what let the guide score a half-masked state — which WHERE and generation both need, and
the scorer never does.

### Two guidance fields

`GuideEnsemble` exposes `guide_form ∈ {"linear", "probit"}`. They are different methods; results
from them must not be pooled.

- **`linear`** (the default; WHERE and generation) returns `(μ_g − τ)/0.5`. Guidance consumes only
  its *gradient* and `τ` enters additively, so **`τ` cancels exactly**: the field is pure
  fitness-directed mean guidance with no threshold. `--rejection-threshold` therefore has no effect
  on site selection and is kept only for interface parity.
- **`probit`** (refinement) returns `log Φ((μ_g − τ)/σ)` with `σ = sqrt(sg[v_g] + 2.31²)`. Its
  gradient *does* depend on `τ`, so guidance weakens once predicted fitness passes the target:
  refinement is given a level to reach, generation a direction to climb. The ensemble spread is
  stop-gradiented so guidance climbs the mean, not the ensemble's own disagreement.

### Where each decision lives

```
train_guide.py     train the guide ensemble   (builds the split, then 5 members)
train_scorer.py    train the scorer ensemble  (reuses that split)
run_fourfold.py    the search: CLI, anchor loop, structural gate
search_stage.py    one mask stage = the round loop that calls the four decisions
```

| `src/fourfold/` | Role |
|---|---|
| `rejection_navigator.py` | **WHERE** — grouped screen, exact rescore, top-\|P_t\| |
| `direct_exo.py` | **WHAT** — guided generation over P_t, low-support resampling |
| `ensembles.py` | guide and scorer ensembles, both guidance fields, **WHICH** |
| `guidance.py` | the first-order Taylor (TAG) estimate guidance rides on |
| `tail_curriculum.py` | the split builder and multitask loss both trainers use |
| `oracle.py` | the 21-letter alphabet and the frozen evaluation oracle |
| `esm3_lora.py`, `esm3_backend.py`, `noisy_predictor.py`, `esmfold.py` | ESM3 editor, token tables, encoders, ESMFold loader |

---

## Install

```bash
python3.10 -m venv .venv
.venv/bin/pip install -e .
.venv/bin/pip install -r requirements-esm3.txt --no-deps   # only needed for the search
```

Run everything **from this directory** with `PYTHONPATH=src:.`; paths below are relative to it.

Training runs on CPU. The search needs one NVIDIA GPU with ≥ 40 GB.

Versions every reported number was produced under: Python 3.10.19, PyTorch 2.9.0+cu128, CUDA 12.8,
`esm` 3.2.1, `transformers` 4.48.1, `peft` 0.14.0, `numpy` 2.2.6, `scipy` 1.15.3, `pandas` 2.3.3,
`biotite` 1.2.0.

ESM3 is the open `esm3_sm_open_v1` checkpoint loaded through the `esm` package; ESMFold is
`facebook/esmfold_v1` loaded through `transformers`. ESMFold must run with its language tower in
half precision and its folding trunk in **fp32** — casting the trunk returns an all-NaN distogram,
so the mixed precision is required rather than chosen.

---

## Data

Assay data is not redistributed. Fetch it from ProteinGuide at the pinned commit:

```bash
git clone https://github.com/junhaobearxiong/proteinguide vendor/proteinguide
cd vendor/proteinguide && git checkout 93a962c03aff0c73f31e63cf82b4574e24750023 && cd -
```

The three processed assay files are then at
`vendor/proteinguide/pdg/fitness/data/{ube4b,pabp,creilov}_processed.csv`.

To use another landscape, supply one CSV with a `sequence` column (all rows the same length —
FOURFOLD substitutes only, never inserts or deletes) and a real-valued `fitness` column, higher is
better. Other columns are ignored, and no normalisation is needed. There is no separate wild-type
input: the split builder derives a consensus sequence from the library itself.

---

## Running it

### 1. Train the two ensembles

Train the guide first — it writes the split the scorer reuses, which is what makes the two
comparable.

```bash
export PYTHONPATH=src:.
DS=ube4b                       # or pabp, creilov, or your own

.venv/bin/python train_guide.py \
  --source-csv vendor/proteinguide/pdg/fitness/data/${DS}_processed.csv \
  --split-dir  data/${DS}_split \
  --output-dir models/${DS}_guide

.venv/bin/python train_scorer.py \
  --split-dir  data/${DS}_split \
  --output-dir models/${DS}_scorer
```

Each writes `member_0.pt … member_4.pt` plus a `training_summary.json` recording the full
configuration and every member's blind-test metrics.

**The defaults are the paper's values. Pass nothing else unless you mean to deviate** — guide
`--seed 42`, scorer `--seed 5531`, both `--members 5 --core-epochs 6 --reward-epochs 14
--learning-rate 2e-4 --weight-decay 1e-3 --batch-size 256 --tail-sampling-weight 4.0
--rank-weight 0.3 --threshold-weight 0.2`. Both scripts are landscape-agnostic; the dataset enters
only through those three path flags.

`train_guide.py` writes six files into `--split-dir`: `core_train.csv` (bottom 90% of fitness),
`core_validation.csv` (10% of core), `tail_reward_train.csv` and `tail_reward_validation.csv`
(90th–99.5th percentile), `extreme_test.csv` (top 0.5%, blind) and `manifest.json`. Training is two
phases: 6 epochs on the core band, then 14 epochs with the tail band added at 4× sampling weight
under a multitask loss (regression + pairwise ranking + tail threshold). The top 0.5% never touches
optimisation or model selection — `training_summary.json` records
`"extreme_test_used_for_selection": false`.

**Smoke test** — one member, one epoch per phase, CPU, a few minutes:

```bash
CUDA_VISIBLE_DEVICES="" .venv/bin/python train_guide.py \
  --source-csv vendor/proteinguide/pdg/fitness/data/ube4b_processed.csv \
  --split-dir /tmp/smoke_split --output-dir /tmp/smoke_guide \
  --members 1 --core-epochs 1 --reward-epochs 1
```

One JSON line per epoch; core-validation Spearman lands near 0.48 after the core phase and near
0.59 after the tail-reward phase.

### 2. Run the search

```bash
export PYTHONPATH=src:. TOKENIZERS_PARALLELISM=false

.venv/bin/python -u run_fourfold.py \
  --dataset ube4b \
  --guide-pool    anchors/ube4b_starts100.csv \
  --training-csv  vendor/proteinguide/pdg/fitness/data/ube4b_processed.csv \
  --predictor-dir models/ube4b_guide \
  --scorer-dir    models/ube4b_scorer \
  --pipelines nav_rejection_fast \
  --anchors 100 --seed 14071 --device cuda:0 \
  --rounds10 5 --candidates10 9 \
  --rounds5  10 --candidates5  9 \
  --screen-groups 8 --candidate-pool 20 \
  --fold-guard --fold-floor 0.80 --fold-max-batches 20 \
  --fold-max-tries 1 --fold-max-drop 1.0 \
  --refine-threshold 0.7 --refine-lambda 1.0 \
  --refine-temperature 0.7 --refine-top-p 0.9 \
  --output-dir runs/ube4b_fourfold
```

`nav_rejection_fast` is the pipeline name for FOURFOLD. `--screen-groups` is the only setting that
differs by landscape: **8** for UBE4B, **12** for PABP, **15** for CreiLOV. `--training-csv` is the
labelled library, used for the novelty distance reported in `summary.json`; point it at the
landscape's assay file.

Start with `--anchors 2` to confirm the pipeline runs. A full landscape costs roughly 5.4, 5.0 and
0.6 GPU-hours for UBE4B, PABP and CreiLOV; folding dominates.

#### Starting sequences

`--guide-pool` is a CSV of measured starting sequences in the same `sequence,fitness` format.
`run_fourfold.py` draws `--anchors` rows from it with `random_state=--seed`, so one seed always
gives the same anchors. The paper's pools are **100 rows sampled at random from the measured
library, unfiltered** — no selection on fitness or on distance from the wild type:

```bash
mkdir -p anchors
.venv/bin/python - <<'PY'
import pandas as pd
DS = "ube4b"
lib = pd.read_csv(f"vendor/proteinguide/pdg/fitness/data/{DS}_processed.csv")
lib.sample(n=100, random_state=14071).to_csv(f"anchors/{DS}_starts100.csv", index=False)
PY
```

A pool's mean fitness should sit close to the library's (for UBE4B, −1.67 against −1.60). A pool
whose mean is far above it has been selected, and its numbers are not comparable with the paper's.

#### Four defaults you must override

Leave any of these at its default and you are running a different algorithm:

| Flag | Default | Why it matters |
|---|---|---|
| `--fold-guard` | off | Without it there is no WHETHER at all — no structural gate |
| `--fold-max-drop 1.0` | `0.03` | At the default, acceptance *also* rejects any proposal dropping pLDDT by more than 0.03. The rule is a pure pLDDT ≥ 0.80 floor |
| `--fold-max-tries 1` | `3` | Same — one fold attempt per proposal |
| `--refine-lambda 1.0` | `0.0` | At `0.0` refinement guidance is off and you get an unsteered low-confidence redraw |

#### Outputs

| File | Contents |
|---|---|
| `summary.json` | full resolved config plus headline numbers |
| `candidate_audit.csv` | every candidate of every round, with `guard_plddt` and `guard_exhausted` |
| `sequences_and_oracle.csv` | endpoints and their oracle scores |
| `stage_endpoints_and_oracle.csv` | per-stage endpoints |

---

## Licensing

Assay data comes from [ProteinGuide](https://github.com/junhaobearxiong/proteinguide) (MIT, pinned
at `93a962c`), which processes UBE4B (Starita et al. 2013), PABP (Melamed et al. 2013) and CreiLOV
(Chen et al. 2023). ESM3-open weights are subject to EvolutionaryScale's licence terms and ESMFold
to Meta's. None are redistributed here. Full attributions in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## Citation

```bibtex
@inproceedings{li2027fourfold,
  title     = {Beyond Mutational Distance: Structure-Constrained Navigation
               of Protein Fitness Landscapes},
  author    = {Li, Haiqing and Zheng, Zheng and Wu, Yinhao and Yang, Hongyu
               and Ma, Hehuan and Guo, Yuzhi and Huang, Junzhou},
  booktitle = {RECOMB},
  year      = {2027}
}
```
# FOURFOLD
