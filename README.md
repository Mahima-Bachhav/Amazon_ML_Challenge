# Business Entity Resolution — Amazon ML Challenge 2026

End-to-end pipeline that, for every Source-1 (S1) business in the test set, finds all matching Source-2/3
(S2/S3) records, and writes the two submission files:

- `matching_results.tsv`: final matches, one row per test S1 entity (the leaderboard file)
- `candidate_pairs.tsv`: the candidate set the final matcher scored

**Final leaderboard F0.5: 0.966.** Held-out F0.5 on India + US: 0.9751. Team project, 4 members.

```
data ─► cleaning + IndicTrans2 translation ─► blocking (L1–L8, L2t) ─► LightGBM pre-filter (top-N)
     ─► LightGBM matcher v3 ─► cross-encoder on unsure pairs + LightGBM stacker
     ─► one-owner rule + per-country thresholds ─► matching_results.tsv / candidate_pairs.tsv
```

---

## 1. Contents of `src/`

Every file is a Jupyter notebook. Notebooks ending in **`_executed`** are the Kaggle notebooks exactly as run
for the submission, with their outputs (recall reports, held-out scores, validator results). The others hold
the same pipeline code for steps whose executed notebook wasn't kept; they're run the same way.

| Step | Notebook | What it does |
|---|---|---|
| — | `00_eda/eda_executed.ipynb` | Data exploration: sizes, countries, noise types, scripts, match statistics |
| 1a | `01_blocking/1a_blocking_L1_L4_test_executed.ipynb` | Blocking layers L1–L4 on **test** |
| 1b | `01_blocking/1b_blocking_L1_L4_train_executed.ipynb` | Blocking layers L1–L4 on **train** (+ labels, recall report) |
| 2a | `01_blocking/2a_phaseB_L5_L7_test_executed.ipynb` | Phase B blocking (L5–L7) on **test** |
| 2b | `01_blocking/2b_phaseB_L5_L7_train.ipynb` | Phase B blocking on **train** (set `MODE = "train"`) |
| 3 | `02_translation/3_translation_indictrans2.ipynb` | IndicTrans2 translation of Indian-script names (run with `MODE = "test"` and `"train"`) |
| 4a | `01_blocking/4a_phaseC_L8_L2t_test_executed.ipynb` | Phase C blocking (L8 + L2t) on **test** |
| 4b | `01_blocking/4b_phaseC_L8_L2t_train_executed.ipynb` | Phase C blocking on **train** (+ recall report) |
| 5 | `03_matching/5_test_text_prep.ipynb` | Cleans all test names/addresses once (`prep_test/`) |
| 6 | `03_matching/6_train_v3_and_heldout_executed.ipynb` | Trains pre-filter + matcher v3; held-out evaluation; saves `heldout_v3.parquet` |
| 7 | `03_matching/7_predict_v3_export_apply_executed.ipynb` | Test inference with v3; exports unsure pairs; **final apply → submission files** |
| 8 | `04_cross_encoder/8_cross_encoder_train_select_score.ipynb` | Cross-encoder: fine-tuning, held-out evaluation, stacker selection, scoring of the unsure test pairs |
| — | `utils/cell0_copy_ckpt_prep.ipynb` | Copies step-1a/5 outputs into a new notebook's working folder |
| — | `utils/validate_streaming.ipynb` | Memory-light submission validator (same rules as the official one) |
| — | `utils/package_submission.ipynb` | Builds the submission zip and pins `requirements.txt` |
| — | `experiments_not_in_final/` | v4 matcher, probes and analyses: documented, **not** used for the submission |

---

## 2. Environment

- **Platform:** Kaggle notebooks. CPU sessions: 4 cores, 30 GB RAM. **GPU (T4)** for steps 1, 2, 3, 4 and 8.
- **Python 3.12**; pinned versions in `requirements.txt`. Some notebooks `pip install` a missing package on
  first run (rapidfuzz, indic_transliteration, IndicTransToolkit), so keep **Internet ON**.
- **Disk:** `/kaggle/working` is limited to 20 GB. Each notebook cleans up its large intermediate files.
- **Seeds:** fixed at 42 everywhere (splits, sampling, LightGBM, PyTorch).

**Data:** the competition dataset with `train/` and `test/` (`*_source1/2/3.tsv`,
`train_ground_truth.tsv`) and `utils/validate_submission.py`. Each notebook's **first cell** sets the paths:
`DATA_ROOT` (dataset folder), `INPUT_ROOT` (attached inputs, default `/kaggle/input`) and `WORK`
(default `/kaggle/working`).

**Passing outputs between notebooks.** A notebook's outputs become the next notebook's inputs, either by
attaching the earlier notebook's output (**Add Input → Notebooks**) or by uploading its files (zips are fine)
as a Kaggle dataset. Every notebook finds its inputs by **file name** under `INPUT_ROOT` and unpacks zips
automatically, so folder names don't matter.

**Smoke runs.** Notebooks with a `SMOKE` flag: run once with `SMOKE = True` (a small slice: checks every
stage, prints a runtime projection), then set `SMOKE = False` for the full run. Interrupted full runs resume
per country.

---

## 3. Step-by-step reproduction

| Step | Run | Hardware | Inputs | Outputs | Time |
|---|---|---|---|---|---|
| 1a | `1a_blocking_L1_L4_test` (`MODE="test"`) | GPU | data | `ckpt_test/<country>/{pairs.npz, qids.npy, pids.npy, stats.json}` | ~2.5 h |
| 1b | `1b_blocking_L1_L4_train` (`MODE="train"`) | GPU | data | `ckpt_train/...` (+ `label`), `train_queries.csv` | ~1.6 h |
| 2a/2b | Phase B, `MODE="test"` / `"train"` | GPU | data, 1a / 1b | `ckpt_b_<mode>/<country>/pairs_b.npz` | ~1 h / ~45 min |
| 3 | translation, `MODE="test"` and `"train"` | GPU, HF token | data | `translations_<mode>.parquet` | ~10 min each |
| 4a/4b | Phase C, `MODE="test"` / `"train"` | GPU | data, 1a / 1b, 3 | `ckpt_c_<mode>/<country>/pairs_c.npz` | ~45 min / ~30 min |
| 5 | `5_test_text_prep` | CPU | data, 1a | `prep_test/<country>/{q,p}.parquet` | ~10 min |
| 6 | `6_train_v3_and_heldout` | CPU | data, 1b, 2b, 4b, 3 (train) | `stage2_models_v3/`, `heldout_v3.parquet` | ~1.5 h |
| 7 (scoring) | `7_predict_v3_export_apply`: Step 2 v3 cells | CPU | 1a, 5, 2a, 4a, 3 (test), 6 | `stage2_v3_output/` (`_parts/<country>.npz`, `candidate_pairs.tsv`) | ~1.5–2.5 h |
| 7 (export) | same notebook: export cell | CPU | step 7 + step 8's config | `ce_band_test.parquet` | ~3 min |
| 8 | `8_cross_encoder_train_select_score` | GPU | data, 1b, 2b, 4b, 3 (train), 6, 7 (export) | `ce_model/` (model, `ce_config_v2.json`, `stacker.txt`), `ce_test.parquet` | ~2 h |
| 7 (apply) | same notebook as 7: apply cell | CPU | step 7, step 8 outputs | **`stage2_v3ce_output/matching_results.tsv`, `candidate_pairs.tsv`** | ~15 min |

**Notes per step**

1. **Blocking L1–L4:** country-partitioned; TF-IDF (char n-grams) → SVD → exact GPU top-K, plus multilingual
   embeddings and exact address keys. The test run writes the union candidate list; the train run attaches
   labels and prints blocking recall (India 90.0%, US 97.4%).
2. **Phase B:** adds L5 (address-only TF-IDF), L6 (exact name/skeleton keys), L7 (phonetic-skeleton TF-IDF)
   with the same `qids`/`pids` order as step 1, so pairs merge exactly.
3. **Translation:** `ai4bharat/indictrans2-indic-en-dist-200M` is a **gated** Hugging Face model. Request
   access on its model page, create a read token, and add it as the Kaggle secret **`HF_TOKEN`**. The notebook
   includes two compatibility fixes for newer `transformers` releases (a stand-in for the removed
   `transformers.onnx` module, and a `tie_weights` signature fix). It translates each distinct name once.
4. **Phase C:** L8 (exact typed-token search, parallel on all CPU cores) and L2t (TF-IDF over translated
   non-Latin records only, GPU). Train mode reports recall old → +B → +C (India 90.0% → 92.7% → 98.2%).
5. **Test text prep:** cleans 11.7M test records once and benchmarks the matching speed.
6. **Matcher v3:** streams ~190M train pairs in chunks. Pre-filter → keep-N per country (India 80, US 30)
   → string features on translated names → LightGBM matcher → per-country thresholds on held-out (hash
   buckets 0–499). Then the diagnostics cells regenerate the held-out predictions and save
   `heldout_v3.parquet` for the cross-encoder.
7. **Test inference:** scores every test S1 entity country by country, with checkpoints in `_parts/`. If run in
   a new notebook, run `utils/cell0_copy_ckpt_prep` first to place `ckpt_test/` and `prep_test/` in the
   working folder.
8. **Cross-encoder:** fine-tunes `paraphrase-multilingual-MiniLM-L12-v2` as a pair classifier
   (CE-1), compares a logistic blend with a LightGBM stacker over three bands on held-out data (CE-1b), saves
   the chosen configuration (band (0.02, 0.99), stacker), then scores the exported unsure test pairs (CE-2).
   The last cells (CE-1c) are the v4 experiment and aren't needed for the submission.
9. **Final apply** (in step 7's notebook): replaces the matcher score of each unsure pair with the stacker
   score, then applies the strict one-owner rule (exactly one S1 per S2/S3 record; ties broken
   deterministically) and the per-country thresholds (India 0.73, US 0.67; France, unseen in training,
   0.69). Finally it runs the official validator.

---

## 4. Exact-reproduction notes

- **Prediction early stopping.** In the submitted run, India and US were scored in step 7 with LightGBM
  prediction early stopping at `pred_early_stop_margin=8.0`, while France was scored without early stopping.
  Margin 8 makes probabilities exact only inside (0.018, 0.982), so the export and apply cells clamp the
  India/US band to that range (`MARGIN8_COUNTRIES = ["India", "US"]`). The code as shipped uses margin 20
  (exact probabilities for practical purposes). To reproduce the submission exactly, use margin 8 for India
  and US; otherwise the result differs negligibly.
- **Band choice.** Step 8 first selected band (0.005, 0.999); the final configuration was re-saved with band
  (0.02, 0.99), stacker (held-out 0.9751 vs 0.9754, within noise), to match the India/US clamp.
- **Runtime:** about 12–14 hours in total. Independent steps were run in parallel on four Kaggle accounts.

---

## 5. Models and licences

All run locally; no external APIs or data. All are under 8B parameters.

| Model / library | Licence | Size | Use |
|---|---|---|---|
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Apache-2.0 | 118M | L3 embeddings; fine-tuned cross-encoder |
| `ai4bharat/indictrans2-indic-en-dist-200M` | MIT | 200M | Indian-script → English translation |
| LightGBM | MIT | — | pre-filter, matcher, stacker |
| rapidfuzz, scikit-learn, indic_transliteration | MIT / BSD / MIT | — | string similarity, TF-IDF/SVD, transliteration |

---

## 6. Troubleshooting

- **Out of memory in step 6:** lower `MATCHER_NEG_RATE` (Cell 1). Prepared texts are cached, so a rerun is
  cheap.
- **`transformers.onnx` not found** or **`tie_weights() got an unexpected keyword argument`** in step 3: both
  fixes are already in the notebook. Run all its cells in order.
- **"not found" errors:** the named input isn't attached. Check under **Add Input**.
- **Validator killed (out of memory):** use `utils/validate_streaming.ipynb`.
