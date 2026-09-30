# Business Entity Resolution — Amazon ML Challenge 2026

Pipeline: **data → blocking (7 layers + L8/L2t) → matching (LightGBM v3) → cross-encoder re-scoring
of unsure pairs + LightGBM stacker → one-owner rule → `matching_results.tsv` / `candidate_pairs.tsv`**.
Final leaderboard F0.5: **0.966** (held-out F0.5 on India/US: 0.9751).

## Environment
Kaggle notebooks (4 CPU cores, 30 GB RAM; GPU T4 where marked), Python 3.12. Pinned versions: `requirements.txt`.
Every script is a Kaggle notebook exported as cells (`# %% [CELL n]`); paste each cell into a notebook
cell in order. Paths are set in each script's first cell (`DATA_ROOT`, `INPUT_ROOT`, `WORK`).
Scripts with a `SMOKE` flag: run once with `SMOKE = True` (quick check), then `SMOKE = False`.
Seeds are fixed (42) everywhere.

## Models used (all open licences, all run locally, no external APIs)
| Model | Licence | Size | Used for |
|---|---|---|---|
| sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | Apache-2.0 | 118M | blocking layer L3 (embeddings); fine-tuned as the cross-encoder |
| ai4bharat/indictrans2-indic-en-dist-200M | MIT | 200M | translating Indian-script business names to English |
| LightGBM | MIT | — | pre-filter, matcher, stacker |

## Run order
| # | Script (src/) | Mode / hardware | Main inputs | Output |
|---|---|---|---|---|
| 1 | `01_blocking/1_blocking_L1_L4.py` | `MODE="test"` and `MODE="train"`, GPU | competition data | `ckpt_<mode>/<country>/pairs.npz, qids.npy, pids.npy` |
| 2 | `01_blocking/2_blocking_phaseB_L5_L7.py` | test + train, GPU | step 1 | `ckpt_b_<mode>/<country>/pairs_b.npz` |
| 3 | `02_translation/translation_indictrans2.py` | test + train, GPU, HF token (gated model) | competition data | `translations_<mode>.parquet` |
| 4 | `01_blocking/3_blocking_phaseC_L8_L2t.py` | test + train, GPU | steps 1, 3 | `ckpt_c_<mode>/<country>/pairs_c.npz` |
| 5 | `03_matching/0_test_text_prep.py` | CPU | step 1 (test) | `prep_test/` |
| 6 | `03_matching/1_train_v3.py` | CPU | steps 1–4 (train) | `stage2_models_v3/` |
| 7 | `03_matching/2_predict_v3.py` | CPU | steps 1–5 (test), 6 | `stage2_v3_output/` (scores in `_parts/`) |
| 8 | `03_matching/3_heldout_predictions_v3.py` | CPU | steps 1–4 (train), 6 | `heldout_v3.parquet` |
| 9 | `04_cross_encoder/1_ce_train_eval.py` | GPU | train data, step 8 | `ce_model/` |
| 10 | `04_cross_encoder/2_ce_stacker_select.py`, then `2b_resave_band_0.02_0.99.py` (same session) | GPU | step 9 | `ce_model/ce_config_v2.json`, `stacker.txt` |
| 11 | `04_cross_encoder/3_export_band_pairs.py` | CPU, in the step-7 notebook | steps 7, 10 | `ce_band_test.parquet` |
| 12 | `04_cross_encoder/4_ce_score_test.py` | GPU | steps 9, 11 | `ce_test.parquet` |
| 13 | `04_cross_encoder/5_apply_final.py` | CPU, in the step-7 notebook | steps 7, 10, 12 | **`stage2_v3ce_output/matching_results.tsv`, `candidate_pairs.tsv`** |

Utilities: `utils/cell0_copy_ckpt_prep.py` (copies step-1/5 outputs into a new notebook),
`utils/validate_streaming.py` (memory-light validator), `utils/notebook1_eda.py` (EDA).
`experiments_not_in_final/`: v4 matcher, probes and analyses that are documented but NOT used for the submission.

## Reproducibility note
In the submitted run, India and US test pairs were scored in step 7 with LightGBM prediction early
stopping at margin 8 (France without it); step 13 therefore clamps the India/US cross-encoder band
to (0.018, 0.982) (`MARGIN8_COUNTRIES`). The included step-7 script uses margin 20 (exact
probabilities); re-running it reproduces the submission up to negligible differences. To reproduce
exactly, set `pred_early_stop_margin=8.0` for India/US in step 7.
Total runtime ≈ 12–14 h across Kaggle sessions (blocking ~4 h, translation ~0.5 h, training ~2 h,
test inference ~2.5 h, cross-encoder ~2 h).
