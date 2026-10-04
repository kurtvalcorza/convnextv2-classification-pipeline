# ConvNeXt V2 Tiny Classification E2E Notebook — Review

**Verdict: Needs revision**  
**Review date:** 2 October 2026  
**Repository:** `kurtvalcorza/convnextv2-classification-pipeline`  
**Notebook:** `tutorials/convnextv2_classification_colab.ipynb`  
**Reviewed commit:** `223a5d23b409eac1426596e2d0c82b896091bd6b` (`main`, confirmed with `gh api repos/kurtvalcorza/convnextv2-classification-pipeline/commits/main` before and after the probes)  
**Notebook Git blob:** `f7d1d0f11ae1431a5ffbf7648e9aa1c0fb6e6f1b`. This is the blob executed in the recorded Kaggle Tesla T4 run of 2026-09-26 (commit `8356fef`, `fetched_blob_verified: true`); `git diff --stat 8356fef 223a5d2` touches only `MODEL_CARD.md`, `README.md`, `STATUS.md`, `docs/release-verification.md` and `tutorials/README.md`. Generator `tools/build_notebook.py --check` and `tools/validate_release_assets.py` both exit 0 at the reviewed commit.  
**Finding prefix:** `CN2`  
**Framework:** Notebook Review Framework v1. **Requirements baseline:** NOTEBOOK_SPEC 2.2 (2026-09-26), `ml-worker` `origin/main` (`b1cfe13`).

## Executive assessment

This is a careful notebook. It carries both package modules byte-for-byte (`data.py` 341 lines, `pipeline.py` 872 lines; cell metadata digests equal the module digests), stages and re-hashes the pinned `facebook/convnextv2-tiny-1k-224` snapshot, downloads a digest-checked CIFAR-10 subset, validates it, splits it three ways, measures a majority-class, a zero-shot ImageNet-mapping and an untrained-head baseline, fine-tunes with every hyperparameter visible as a form field, records the trainable-parameter count, scores held-out and unseen splits with accuracy, balanced accuracy and a confusion matrix, probes blank and noise images before and after adaptation, exports an adapter with base-model provenance, and checks the reloaded adapter against the in-memory model with a stated tolerance. Many problems found in sibling notebooks are already solved here.

The problems are in what the default experiment can show, and in what happens when a learner follows the suggested experiment. A direct CPU run of every code cell at the documented defaults reproduced the Kaggle record:

| Measure | This review (CPU, pinned venv, defaults) | Kaggle T4 record (blob `f7d1d0f1`) |
|---|---|---|
| Code cells completed | 14/14 (172.8 s of cell time; install skipped) | 14/14 on pass 2 (pass 1 stopped at the install guard) |
| Dataset | 400 records, 200 per class, 32×32, **100 pixel-identical groups** (finding printed) | identical |
| Split | 280 / 60 / 60 | identical |
| Held-out accuracy: majority / zero-shot / untrained head / fine-tuned | 0.500 / **1.000** / 0.817 / **1.000** | 0.500 / 1.000 / 0.817 / 1.000 |
| Unseen accuracy, fine-tuned | 1.000 | 1.000 |
| Held-out / unseen images with a pixel copy in train | **25/60 / 19/60** | not measured (record counts 46/60 / 36/60 same-numbered counterparts) |
| Fine-tune cell | 144.8 s (27,868,034 trainable) | about 30 s |
| Reload check | 60 images, tolerance 1e-4, equivalent | identical |

Five problems stand in the way of `Ready for intended use`:

1. **No one-pass `Run all` (CN2-M1).** The recorded run stopped at the install cell's stale-module guard (`cuda-bindings` 12.9.4 → 13.4.3, `numpy` 2.0.2 → 2.5.3) and passed only after a restart.
2. **The split ignores the duplicates the notebook itself reports (CN2-M2).** `validate_dataset` prints "100 group(s) of pixel-identical images" (each `darkened_images/<class>/image_N.png` equals `original_images/<class>/image_N.png`), then Section 5 says a random split "is valid here because CIFAR-10 images are independent thumbnails". 25 of 60 held-out and 19 of 60 unseen images have an exact copy in the training split.
3. **The central comparison cannot show what the prose says it shows (CN2-M3).** The zero-shot ImageNet mapping already scores 1.000, so the fine-tune (1.000) shows no gain; Section 7 tells the learner to "expect roughly chance" from the untrained head, which scores 0.817 at the default seed and anywhere from 0.217 to 0.817 across five seeds; the Interpretation section never says the fine-tune added nothing measurable.
4. **The suggested head-only experiment silently runs on the already fine-tuned model when only the edited cell is re-run (CN2-M4).** It then reports 1,538 trainable parameters and 1.000 held-out accuracy, and the export cell fails with a bare `AssertionError`. Nothing says which cells to re-run.
5. **Guided layer partly absent (CN2-M5).** Declared `GUIDED`, but there is no audience statement, how-to-use, roadmap, task contract, glossary, prediction, checkpoint, troubleshooting or conclusion template, and the 1,213 lines of carried modules are not labelled as infrastructure.

## 1. Review contract and evidence

| Item | Value |
|---|---|
| Declared profile / mode | `E2E` / `GUIDED` (metadata `dimer.notebook_profile` / `notebook_mode`, opening cell) |
| Declared spec | DIMER Notebook Specification **2.1** (metadata, opening cell, `NOTEBOOK_SOURCE`) |
| Spec baseline applied | NOTEBOOK_SPEC **2.2** |
| Intended audience | Not stated. Knowledge prerequisites: basic Python and PIL; softmax; how a ConvNet pools into a classifier; accuracy, balanced accuracy, confusion matrix; why a baseline is needed |
| Supported runtime | "Google Colab or Jupyter, Python 3.12"; CUDA GPU (T4) documented for the fine-tune, CPU also runs; float32 |
| Promised outcomes | Pinned install; carried package; digest-verified snapshot; digest-verified CIFAR-10 subset (`frog`, `truck`) validated before any model runs; stratified three-way split; majority-class and zero-shot baselines; blank/noise probes; head replacement and untrained baseline; bounded fine-tune; held-out comparison; unseen-split inference; adapter export, fresh reload and equivalence check; outputs with provenance; BYOD image and BYOD dataset branches through "the full adaptation workflow" |
| Generator | `tools/build_notebook.py` (`build_notebook.py/2`) + `tools/notebook_template.py`; recorded generating revision `82b6c18` |

### Evidence actually obtained

- **Source inspection.** All 31 cells (14 code; cells 5 and 7 carry `data.py` and `pipeline.py`). Also read: the two modules (`split_dataset`, `validate_dataset`, `from_pretrained`, `finetune`, `evaluate`, `save_artifact`, `apply_artifact`, `load_artifact`), the generator and template, `README.md`, `STATUS.md`, `tutorials/README.md`, `docs/release-verification.md`. The repository has no `AGENTS.md` and no `docs/execution-evidence/` directory.
- **Documented execution evidence.** `docs/release-verification.md` plus the archived executor output for Kaggle kernel `dimer-nb2-convnextv2-classification` v1 (`run_summary.json`, `executed-pass1.ipynb`, `executed.ipynb`, `outputs/`). Kaggle Tesla T4, 2026-09-26, **the reviewed blob**, clean model cache, image torch 2.10.0+cu128 / numpy 2.0.2. Pass 1 failed in cell 3 with the restart `RuntimeError` (192.8 s); pass 2 ran 14/14 (84.0 s). No Colab run, no BYOD run and no optional-experiment run is recorded.
- **Direct execution (this review).**
  - **Environment:** `run_probes.py`, Windows 11, CPU only (`CUDA_VISIBLE_DEVICES=-1`), 24 threads, build venv `dimer-next16` (Python 3.12.10, torch 2.14.0+cu130, torchvision 0.29.0, transformers 4.57.6, safetensors 0.8.0, numpy 2.5.3, pillow 11.3.0, huggingface-hub 0.36.2 — the notebook's `PINS`). Nothing was installed.
  - **Install skipped:** cell 3 ran with `DIMER_NOTEBOOK_CI_PREINSTALLED=1`, the notebook's executor hook.
  - **Not a clean runtime:** the four snapshot data files were hard-linked into a scratch working directory; cell 9 wrote the manifest fresh and `verify_snapshot` re-hashed every file. Cell 11 fetched the dataset itself from its pinned URL and checked its digest.
  - **Executed:** every code cell at defaults (P1); duplicate leakage across the split and the scores on the leak-free subsets (P2); untrained head at init seeds 0–4 (P3); "Try next" head-only experiment re-running only cell 19 (P4) and re-running cells 17–25 (P5); "Try next" `PER_CLASS = 10` re-running cells 11–21 (P6); BYOD image through `BYOD_IMAGE_PATH` with one valid and three invalid inputs, and with an empty path outside Colab (P7); BYOD dataset through `BYOD_DATASET_PATH` with three accepted and six refused inputs (P8). P7/P8 ran in a second invocation after a probe-script bug (path escaping) in the first; `results.json` documents the merge. `google.colab.files.upload` was replaced by a shim for one upload probe; the Colab upload dialog itself was not exercised.
- **Learner observation:** none. No claim here is about measured learning effectiveness.

## 2. Separate judgments

- **Technical correctness:** good on the default path (P1 14/14; reload equivalent within 1e-4 on 60 images; BYOD refusals name the failed rule). Defects: the install pattern forces a restart (CN2-M1); a partial re-run of the fine-tune cell leaves the pipeline in a state the export cannot represent (CN2-M4).
- **Promise fulfilment:** all default-path stages run and are reported. The fine-tune is real, but the default comparison cannot demonstrate a fine-tuning effect (CN2-M3). The BYOD dataset branch reaches fine-tune and export but not new-data inference, an equivalence check or a written record (CN2-m1).
- **Scientific validity:** the split ignores 100 duplicate pairs the validator reports, and the prose asserts the independence that the data lacks (CN2-M2). On this pair of classes the leak does not change the numbers (both leak-free subsets also score 1.000), because the task is at ceiling.
- **Learner experience:** clear stage prose, two "What to look for" notes, an explicit "read the loss as optimisation evidence only" note and an honest limits section. The untrained-head expectation is wrong (CN2-M3), re-run guidance is missing (CN2-M4), and the GDL layer is mostly absent (CN2-M5).
- **Spec conformance:** unresolved applicable MUSTs — RUN1, RUN10, ENV6 (CN2-M1); SPL3, SPL5 (CN2-M2); EVAL3 (CN2-M3); DAT14, VER5, DAT19 (CN2-m1); UX12 (CN2-m2); REL12 BYOD evidence absent from the release record. SHOULD deviations: SPL10 (CN2-M2); GDL7, GDL8, GDL14 (CN2-M3); GDL10, UX10 (CN2-M4); GDL1–GDL4, GDL6, GDL9, GDL11–GDL13, UX8 (CN2-M5); SPL2 (CN2-m1).

## 3. Promise and objective tracing

| Claim / objective | Implementation | Observable result | Learner interpretation | Status |
|---|---|---|---|---|
| One-pass `Run all` | cell 3 in-kernel `pip install` + stale-module guard | Kaggle pass 1 `RuntimeError`, restart, pass 2 14/14 | Section 1 says the cell "stops with a restart instruction" | **Not met** (CN2-M1) |
| Digest-verified pinned snapshot | cell 9 | 4/4 files verified at `f4db009e…` | clear | Met |
| Digest-verified dataset, validated before any model | cell 11 | 986,707 B, sha256 `66f90a4f…`; 100 duplicate groups reported | "What to look for" explains 32 px upsampling; duplicate finding not discussed | Met (finding ignored downstream, CN2-M2) |
| Stratified split; "valid here because … independent thumbnails" | cell 13, `split_dataset` | 280/60/60, ID-leakage assert passes; 25/60 held-out and 19/60 unseen pixel copies in train | independence asserted | **Not met** (CN2-M2) |
| Majority and zero-shot baselines | cell 13 | 0.500, 1.000 | zero-shot explained | Met |
| Blank/noise probes before and after | cells 15, 23 | ImageNet head flat; adapted head `frog` 0.949 / 0.882 (CPU), 0.971 / 0.895 (T4) | well explained | Met |
| Untrained head "roughly chance", "the fine-tune — not the re-heading — is what moves the score" | cell 17 | 0.817 at seed 0; 0.217–0.817 across seeds 0–4 | expectation contradicted | **Not met** (CN2-M3) |
| Bounded fine-tune, trainable set and schedule stated | cell 19 | 27,868,034 / 27,868,034 trainable; losses printed | clear | Met |
| Held-out comparison against both baselines | cell 21 | fine-tuned 1.000 = zero-shot 1.000 | "a difference of one or two images is within noise"; no statement that this run shows no gain | Computation met, conclusion unsupported (CN2-M3) |
| Unseen-split inference | cell 23 | 1.000 | — | Met (leak caveat, CN2-M2) |
| Export, fresh reload, equivalence with tolerance | cell 25 | 200 tensors; 60 images equivalent within 1e-4 | "loading succeeding is not the check" | Met on the default path; breaks after a partial re-run (CN2-M4) |
| Outputs with provenance | cell 27 | 5 files; result JSON holds dataset digest, split, fine-tune config, metrics, artifact descriptor, reload check | listed | Met |
| BYOD image | cell 29 | P7: path read, top-5 + `not-measurable`; 5000 px and missing path refused with the rule; text file → `UnidentifiedImageError` naming the path; empty path outside Colab → `ModuleNotFoundError: google.colab` | contract stated | Met (local), one weak message (CN2-m1) |
| BYOD dataset "full adaptation workflow — validate, split, baselines, fine-tune, evaluate, export and reload" | cell 29 | P8: folder and zip reach fine-tune → export → load; no unseen split, no new-data inference, no equivalence check, no metrics written | contract stated | Partly met (CN2-m1) |

| Learning objective (opening cell) | Learner activity | Evidence exercised |
|---|---|---|
| Install, read the carried package, verify the revision | run cells | versions and verified-file count printed |
| Download and validate a dataset before any model runs | run cell, read manifest | duplicate finding printed, then ignored (CN2-M2) |
| Read top-5; build a zero-shot baseline | run, read | outputs readable |
| See what a closed-set classifier answers for blank and noise | run, read | "What to look for" note; strong |
| Split with stratification | run | split is not duplicate-aware (CN2-M2) |
| Fine-tune; compare against both baselines | run, read table | table shows a tie at 1.000; no prompt to interpret it (CN2-M3) |
| Export, reload and verify the adapter | run | met |

Objectives are phrased as actions the code performs (GDL5), and none is followed by a check of the learner's understanding.

## 4. Journeys

| Journey | Basis | Result |
|---|---|---|
| **First-time learner** | Source inspection, all 31 cells | Each section says what runs and why; two "What to look for" notes; the loss and the held-out metrics are labelled as optimisation and tutorial evidence. The learner is told the split is valid because the thumbnails are independent right after being shown 100 duplicate groups (CN2-M2), is told to expect chance from the untrained head and sees 0.817 (CN2-M3), and is not told what a tie between zero-shot and fine-tuned at 1.000 means (CN2-M3). Prerequisites say "Runtimes are not measured in this revision" although the record measured them (CN2-m2). No audience, roadmap, glossary, predictions, checkpoints or troubleshooting (CN2-M5). |
| **Clean default** | Documented (Kaggle T4, reviewed blob) + direct (CPU, install skipped) | Kaggle: pass 1 failed at the install guard after 192.8 s, pass 2 14/14 after a restart (CN2-M1). Direct: 14/14 at defaults, numbers in the table above; five outputs written; reload equivalent. No Colab run. |
| **Active learning** | Direct (P4, P5, P6) | "Try next" 1, as a learner would do it — set `FREEZE_BACKBONE = True` in cell 19 and re-run that cell: training continues from the fully fine-tuned weights, the cell prints `trainable_parameters: 1538`, cell 21 shows 1.000, and cell 25 raises a bare `AssertionError` because the head-only adapter cannot reproduce a modified backbone (CN2-M4). Re-running cells 17–25 instead works: head-only 0.983 held-out in 39.7 s vs full 1.000 in 144.8 s (CPU). "Try next" 2, `PER_CLASS = 10` with cells 11–21 re-run: split 14/4/2, zero-shot 1.000, fine-tuned 1.000 on 4 held-out images — the promised fall "toward the zero-shot baseline" cannot be observed when the baseline is at 1.000 (CN2-M3). |
| **Reuse and recovery** | Direct (P7, P8); Colab upload dialog not verified | BYOD image through `BYOD_IMAGE_PATH`: accepted and classified (`tailed frog` 0.525 on an upscaled CIFAR frog), report `not-measurable`; refused with the rule named: 5000×10 px image, missing file; raw `UnidentifiedImageError` (path named) for a text file named `.png`; empty path on a non-Colab runtime → `ModuleNotFoundError: No module named 'google.colab'`. BYOD dataset: folder and zip of 2×6 images reached validate → split (8/4) → untrained 0.0 → fine-tune → 0.75 → export → load. Refused with the rule named: class with one image, one class, `../` member, undecodable member (named). Accepted silently: a `train/` + `val/` archive (merged by class and re-split). Weak messages: root-level images ignored, then "class_names must hold 2..1000 names, got 1"; a `.tar` path → "is not a directory" (CN2-m1). |

## 5. Findings

### Major

#### CN2-M1 — `Run all` needs a manual restart after the install cell

- **Cell/section:** cell 3, Section 1 (generator `tools/build_notebook.py`, install block lines 50–75); `docs/release-verification.md` record.
- **Observed issue:** the cell `pip install`s seven pins into the running kernel, then raises `RuntimeError: Core dependencies changed while older modules were loaded … Restart the runtime, then rerun from the top.` when a loaded distribution changed. Section 1 prose presents this as expected behaviour.
- **Consequence:** a learner selecting **Run all** on a stock Kaggle/Colab image hits an error in the first code cell and must restart and run again. RUN1, RUN10 and ENV6 forbid this. The record reports the run as "PASSED (default path)" with the restart disclosed in the same cell.
- **Evidence:** documented — Kaggle T4 run of blob `f7d1d0f1`, pass 1 `ok: false` with `cuda-bindings: loaded=12.9.4, installed=13.4.3; numpy: loaded=2.0.2, installed=2.5.3`, pass 2 after restart 14/14. Source — probe static `pip_install_in_kernel: true`, `uses_uv: false`.
- **Recommended correction:** Adopt the fleet's **uv isolated-environment pattern**, which is how the capstone and newer workshop notebooks already run in one pass: the setup cell bootstraps uv, creates an isolated managed interpreter (`uv venv --managed-python --python 3.12.12 <ROOT>/env`), installs a hash-locked `requirements.txt` compiled with `uv pip compile` (`uv pip install --require-hashes --only-binary :all:`), and runs the pinned stages in that environment, so the kernel's preloaded NumPy/torch are never replaced and no restart can be required. Reference implementations on `main`: `ast-audio-classification-pipeline/tutorials/DIMER_Sound_Event_Classification_Workshop.ipynb` and `bioclip2-biodiversity-pipeline/tutorials/DIMER_Philippine_Biodiversity_Field_Survey_Capstone.ipynb`. Do not add another in-kernel install guard or loosen pins to dodge the restart. Implement it in the repository's notebook generator, regenerate, re-qualify with a one-pass hosted Run all, and correct the release record so a restart-dependent run is not reported as a `Run all` PASS.
- **Acceptance check:** a fresh Kaggle or Colab runtime completes every code cell in a single **Run all** with no restart and no error, recorded in `docs/release-verification.md` with the notebook blob id and `restarted: false`; `grep -n "Restart the runtime" tutorials/convnextv2_classification_colab.ipynb` returns nothing.
- **Spec:** RUN1, RUN10, ENV6, REL2.

#### CN2-M2 — The split ignores the 100 duplicate pairs the validator reports, and the prose asserts independence

- **Cell/section:** cell 11 output (`duplicate_groups: 100`), Section 5 prose ("A random split is valid here because CIFAR-10 images are independent thumbnails"), cell 13; `data.py` `split_dataset`. Generator: `tools/notebook_template.py` lines 155 and 169–170; `src/convnextv2_classification_pipeline/data.py` `split_dataset`.
- **Observed issue:** the archive holds `original_images/<class>/image_N.png` and `darkened_images/<class>/image_N.png`; for 100 values of N the two files are pixel-identical. `validate_dataset` reports this as a finding, and its own docstring says a duplicate "inflates the held-out score when one copy lands on each side of a split". `split_dataset` then shuffles individual records; the only leakage check (cell 13) compares record IDs, which always differ between the two copies.
- **Consequence:** 25 of the 60 held-out and 19 of the 60 unseen images have an exact copy in training, so the "held-out" and "unseen" scores are partly scores on training images, and the learner is taught that a random split is valid on data the notebook has just flagged as duplicated. On this class pair the numbers do not move (leak-free subsets: held-out 35 images, unseen 41, both 1.000 for zero-shot and fine-tuned), but the method carries straight into the BYOD branch, where it would inflate a user's score silently.
- **Evidence:** direct (P2) — `held_out_with_pixel_copy_in_train: 25/60`, `unseen_with_pixel_copy_in_train: 19/60`, all groups of size 2, no group spans two labels; documented — the record's caveat (counts 46/60 and 36/60 same-numbered counterparts, not pixel copies). Source — `split_dataset` has no group parameter.
- **Recommended correction:** make the split group-aware: group records by pixel digest (or by `image_N` across the two folders) and assign whole groups to one side, then assert in cell 13 that no pixel digest appears in two splits and print the count. Alternatively drop one of each duplicate pair before splitting and say so. Replace the "independent thumbnails" sentence with the actual reason the split is valid after de-duplication, and apply the same rule in the BYOD branch.
- **Acceptance check:** cell 13 prints a cross-split duplicate count of 0 for held-out and unseen, computed from pixel digests; the Section 5 prose no longer claims the archive is independent; a test in `tests/test_data.py` builds an archive with a duplicate pair and shows both copies land in one split.
- **Spec:** SPL3, SPL5, SPL10, EVAL5.

#### CN2-M3 — The default comparison cannot show a fine-tuning effect, and the prose says it does

- **Cell/section:** Section 7 prose ("**Expect roughly chance.** … This row shows the floor, and that the fine-tune — not the re-heading — is what moves the score"), cell 21 comparison, Interpretation section and "Try next". Generator: `tools/notebook_template.py` line 214 and lines 450–460.
- **Observed issue:** on frog vs truck the zero-shot ImageNet mapping already scores 1.000 on the held-out split, as does the fine-tuned head; the untrained random head scores 0.817 at the default `SEED = 0` and ranges from 0.217 to 0.817 across seeds 0–4. The Interpretation section says the result "was compared with the majority-class, zero-shot and untrained baselines" without saying that the fine-tune did not beat the zero-shot baseline. "Try next" asks the learner to lower `PER_CLASS` and "watch how quickly the fine-tuned model falls back toward the zero-shot baseline", which is 1.000.
- **Consequence:** the central lesson — that fine-tuning, not re-heading, improves the score — is not demonstrated by the default run, and the stated expectation for the untrained head is contradicted by the first number the learner sees. A learner can reasonably conclude that the fine-tune worked (1.000 vs 0.817) when the correct reading is "the task is at ceiling for the pretrained model; this run measures no fine-tuning gain".
- **Evidence:** documented — Kaggle held-out 0.500 / 1.000 / 0.8167 / 1.000; direct — P1 identical; P3 untrained seeds 0–4: 0.817, 0.683, 0.533, 0.433, 0.217 (Wilson 95 % for seed 0: [0.70, 0.89], excludes 0.5); P6 `PER_CLASS = 10`: zero-shot 1.000, fine-tuned 1.000 on 4 held-out images. The release record already states "no measurable gain"; the notebook does not.
- **Recommended correction:** choose a default task where the pretrained head is not at ceiling (for example two CIFAR-10 classes that ImageNet maps poorly, or more classes), or keep frog/truck and say plainly before and after cell 21 that this pair is at ceiling for zero-shot so the fine-tune can only tie. Replace "Expect roughly chance" with "a random head can land anywhere between well below and well above 50 %; it carries no information about the task" and, optionally, report the untrained head over several init seeds. Add a sentence to the Interpretation section that states this run's fine-tune vs zero-shot result, and rewrite the `PER_CLASS` experiment so its expected observation is possible.
- **Acceptance check:** on the default run either the zero-shot held-out accuracy is below the fine-tuned accuracy by more than the stated noise, or the notebook prints/states that the comparison is at ceiling; `grep -n "Expect roughly chance" tutorials/convnextv2_classification_colab.ipynb` returns nothing; the Interpretation section names the fine-tuned vs zero-shot result; every "Try next" item's expected observation is reproducible on the default data.
- **Spec:** EVAL3, GDL7, GDL8, GDL14, UX5.

#### CN2-M4 — Re-running only the edited fine-tune cell trains the already adapted model, misreports it, and breaks the export

- **Cell/section:** cell 19 (`FREEZE_BACKBONE` form field), cells 21 and 25, "Try next" in the Interpretation section; `pipeline.py` `finetune` (mutates `self`), `save_artifact` (omits `frozen_prefixes` tensors). Generator: `tools/notebook_template.py` cell 19 block and line 458.
- **Observed issue:** `adapter` is created in cell 17 and mutated in place by `finetune`. A learner following "Set `FREEZE_BACKBONE = True` and compare" naturally edits cell 19 and re-runs it. That continues training the fully fine-tuned network with its backbone frozen, prints `trainable_parameters: 1538`, and cell 21 reports 1.000 as if it were a head-only result. Cell 25 then saves only the head (the backbone is "frozen", so assumed equal to the base), reloads it onto the base model, and the predictions differ: bare `AssertionError` with no message. No cell says which cells to re-run after changing a field.
- **Consequence:** the notebook's one guided experiment produces a wrong comparison (full + head-only training labelled head-only) and then crashes without a recovery message. The correct procedure (re-run from cell 17) works, but the learner is not told it.
- **Evidence:** direct (P4) — re-run of cell 19 only: losses 0.0008 → 0.0001, `trainable_parameters 1538`, held-out 1.000, cell 25 `AssertionError:`; (P5) re-run of cells 17–25: 1,538 trainable, held-out 0.983, 2-tensor adapter, reload equivalent, fine-tune 39.7 s vs 144.8 s full on CPU.
- **Recommended correction:** make the experiment self-contained: build a fresh pipeline inside the fine-tune cell (or refuse to fine-tune a pipeline whose `adapted` is already `True` with a message saying "re-run from Section 7"), print in the "Try next" text exactly which cells to re-run, and give the reload assertions messages that name the mismatch and the next step.
- **Acceptance check:** with defaults run once, setting `FREEZE_BACKBONE = True` and re-running only cell 19 either trains from the base checkpoint (cell 25 then passes) or stops with a message naming the cells to re-run; every "Try next" item names its field and the cells to re-run.
- **Spec:** GDL10, UX10, UX5, FT5.

#### CN2-M5 — Declared `GUIDED`, but most of the guided layer is absent

- **Cell/section:** opening cells 0–1, every section boundary, cells 3, 5, 7, 9, end of notebook. Generator: `tools/notebook_template.py`, `tools/build_notebook.py`.
- **Observed issue:** no intended-learner statement, no **How to use this notebook**, no roadmap, no Input → Model → Output contract (ImageNet classification and two-class adaptation share one notebook), no glossary (FCMAE, GRN, softmax, balanced accuracy, AdamW, cross-entropy, adapter, zero-shot mapping), no prediction prompts before the zero-shot, untrained-head or fine-tune results, no interpretation checkpoints, no troubleshooting section (restart, Hub download, out-of-memory on CPU, BYOD errors), no conclusion template. The two module cells (1,213 lines) and the manifest cell are not titled **Infrastructure** and are not collapsed (`cellView` absent everywhere). The notebook does have two "What to look for" notes, expectation notes in Sections 7 and 8, and a strong limits section.
- **Consequence:** a self-paced learner gets an accurate, well-explained script but no prompts to commit to a prediction, check understanding or recover from expected failures, and must scroll past 1,213 lines of package code without being told it can be skipped.
- **Evidence:** source inspection; probe static `guided_markers` (How to use / Roadmap / Glossary / Check your reasoning / Troubleshooting / conclusion / audience / Infrastructure / re-run all absent), `cellView_form_cells: []`, `what_to_look_for_count: 2`.
- **Recommended correction:** add the GDL layer in the template following NOTEBOOK_SPEC §25.13's reference notebook: audience and how-to-use, roadmap, task contracts for both capabilities, glossary, a prediction before Sections 5, 7 and 9, a collapsible checkpoint after the comparison table, a Predict → Change one thing → Run → Observe → Explain activity built on CN2-M4's fix, troubleshooting, and a conclusion scaffold; title cells 3, 5, 7 and 9 `# @title Infrastructure: …` with `cellView: form`.
- **Acceptance check:** each of GDL1–GDL4, GDL6, GDL9–GDL14 maps to a named cell in a checklist added to `tutorials/README.md`; cells 3, 5, 7 and 9 carry `cellView: form` with an Infrastructure title.
- **Spec:** GDL1–GDL4, GDL6, GDL9, GDL11–GDL14, UX8.

### Minor

#### CN2-m1 — BYOD dataset branch stops short of the promised workflow; some inputs get weak messages

- **Cell/section:** cell 29; opening cell and Section 13 prose ("the same validate → split → baselines → fine-tune → evaluate → export → reload stages as the sample"). Generator: `tools/notebook_template.py` lines 395–438.
- **Observed issue:** the branch splits train/held-out only (no unseen split or new-data inference), calls `load_artifact` without comparing the reloaded predictions (the default path does compare), writes no metrics, dataset manifest or split record (only the adapter file), and prints `{'majority_baseline', 'untrained', 'fine_tuned'}` mixing accuracy and balanced accuracy without saying so. A `train/` + `val/` archive is merged by class name and re-split silently. Images at the archive root are ignored without a message, so an archive with one class folder plus root images fails with "class_names must hold 2..1000 names, got 1". A `.tar` path fails as "is not a directory". With an empty path on Kaggle or local Jupyter the branch raises `ModuleNotFoundError: No module named 'google.colab'`. The BYOD branch also inherits CN2-M2's split.
- **Consequence:** a user's adaptation result is printed once and not recorded; reload is a load, not a check; pre-split data loses its split; some failures do not tell the user what to fix.
- **Evidence:** direct (P8) — `a_folder_ok`/`b_zip_ok`: split 8/4, untrained 0.0, fine-tuned 0.75, adapter exported and loaded, nothing else written to `outputs/`; `g_presplit_train_val` accepted, 12 records re-split 8/4; `h_root_images_plus_one_class` → `ValueError: class_names must hold 2..1000 names, got 1`; `i_tar_path` → `FileNotFoundError: … is not a directory`. (P7) empty path outside Colab → `ModuleNotFoundError`. Refused with a clear rule: class with one image, one class, `../` member, undecodable member (named), 5000 px image, missing file.
- **Recommended correction:** reuse the default path's reload comparison and output writer for BYOD (`outputs/byod_result.json` with dataset manifest, split, fine-tune config and metrics); label the printed metrics; keep a `train/`/`val/` layout as the split when present (or say it is merged); report ignored root-level files; when the path is empty and `google.colab` is unavailable, stop with "set `BYOD_DATASET_PATH`"; state the `.zip`-or-directory rule in the error.
- **Acceptance check:** a BYOD dataset run writes a result JSON with metrics and a reload check marked `equivalent`; the root-images, `.tar` and empty-path-outside-Colab cases each stop with a message naming the rule and the field to set.
- **Spec:** DAT14, VER5, DAT19, SPL2, UX10, REL12.

#### CN2-m2 — Runtime claim and release record out of step with the evidence

- **Cell/section:** cell 1 Prerequisites ("Runtimes are not measured in this revision"); `docs/release-verification.md` caveat on duplicates.
- **Observed issue:** the record measured 276.9 s wall on T4 (pass 2 84.0 s, fine-tune about 30 s) for this exact blob; the notebook still says runtimes are not measured and gives no CPU estimate although it says CPU works "more slowly" (this review: fine-tune 144.8 s, all cells 172.8 s on a 24-thread CPU). The record's leakage caveat counts 46/60 and 36/60 same-numbered counterparts; the pixel-copy counts are 25/60 and 19/60.
- **Consequence:** the learner cannot plan time; a release reviewer reads a leakage figure that overstates the measured one (the record says so, but does not give the measured figure).
- **Evidence:** documented record; direct P1 and P2 timings and counts.
- **Recommended correction:** state measured times with runtime and date (T4; CPU labelled as an estimate) in the Prerequisites; replace the counterpart count in the record with the pixel-digest count (or remove it once CN2-M2 is fixed).
- **Acceptance check:** the Prerequisites give a measured T4 time with date and runtime; the record's duplicate figure is computed from pixel digests.
- **Spec:** UX12, REL10.

### Suggestions

- **CN2-S1** — Declare `notebook_spec` 2.2 instead of 2.1 once the guided layer lands.
- **CN2-S2** — Show a small grid of held-out images with true label, predicted label and score, including any errors, next to the confusion matrix.
- **CN2-S3** — Report the untrained head as a band over a few init seeds (P3 shows 0.22–0.82) so the "floor" is not read from one draw.
- **CN2-S4** — Record per-stage wall times in `result.json` (the fine-tune dominates: 144.8 of 172.8 s on CPU).

## 6. Readiness

**Needs revision.** Open Majors CN2-M1 to CN2-M5. Remaining gates after the fixes: a one-pass hosted Run all of the regenerated blob (RUN1/RUN10), the REL12 BYOD exercise recorded in `docs/release-verification.md` (one compatible and one incompatible input), and a default run whose held-out and unseen splits contain no pixel copy of a training image.

## 7. Verified versus inferred

- **Verified by direct execution (CPU, install skipped, labelled above):** default path 14/14 with the same metrics as the T4 record; 25/60 held-out and 19/60 unseen pixel copies in train and the 1.000 scores on the leak-free subsets; untrained-head spread across seeds; the partial re-run failure and the working full re-run; `PER_CLASS = 10` outcome; the BYOD image and dataset outcomes in §4.
- **Verified from documented evidence:** the restart on Kaggle pass 1; the T4 metrics and timings.
- **Inferred from source:** that the Colab upload dialog delivers files as the shim did; that GPU runs show the same ceiling (one T4 run exists and agrees).
- **Only Kurt can confirm:** whether frog/truck should stay as the default pair (CN2-M3 accepts either a harder pair or an explicit at-ceiling explanation).
- **Most likely to be wrong:** CN2-M2's severity — on this data the leak does not change any number (the leak-free subsets also score 1.000), so a maintainer could rate it Minor; I rated it Major because the notebook teaches a split rule its own validator contradicts, and the BYOD branch applies it to user data.
