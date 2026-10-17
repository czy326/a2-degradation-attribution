# A2 — Degradation Attribution for Small-Object Detection (code and results)

Code, pre-registration documents, verdicts and per-condition results for the study

> **Attribution and training-side optimisation of accuracy loss of small-object detection
> under real deployment degradation in remote sensing imagery**

The paper asks three questions and answers them with paired, seed-level statistics:
which degradation source actually costs accuracy, whether widening the *training-time blur
range* helps, and whether the answer survives a change of dataset scale and detector architecture.

Everything reported in the paper can be recomputed from the files in `results/` without
re-running a single training run.

---

## Repository layout

| Path | Contents |
|---|---|
| `code/` | Degradation operators, evaluation harness, training drivers, verdict scripts |
| `prereg/` | Pre-registration documents, written **before** each experiment was launched |
| `verdict/` | Verdict documents produced by the scripts, with the decision rules applied verbatim |
| `results/` | Per-condition mAP, per-seed paired differences, verdict JSON |
| `figures/` | Plot scripts and the exported figures (300 dpi PNG + vector PDF + SVG) |
| `docs/PROTOCOL.md` | The experimental protocol in one page: conditions, size binning, statistics |

## Data

All three datasets are public. We do not redistribute imagery.

| Dataset | Role | Size used here |
|---|---|---|
| **AI-TOD** | main dataset (same-scale) | 2,804 images, 200-image evaluation slice with 7,304 annotated objects |
| **SODA-A** | same-scale replication | 2,513 large scenes tiled into 800 px tiles, 500-tile evaluation slice with 8,363 objects |
| **DIOR** | cross-scale boundary test | 11,725 trainval images, 1,500 train / 500 evaluation |

Labels are converted to YOLO format with the protocol in `docs/PROTOCOL.md`.
The AI-TOD evaluation slice is fixed and reproducible: images come from the same pool used
for training and validation, but a disjoint 200-image slice is carved out for evaluation
(see `code/dcdc_scan.py`, `POOL` / `TRAIN_FRAC` / evaluation slice construction).

## Environment

```
python >= 3.10
torch >= 2.0 (CUDA), ultralytics, opencv-python, numpy, scipy, matplotlib
```

```bash
pip install -r requirements.txt
```

Weights: `yolov8n.pt`, `yolo11n.pt` (Ultralytics), `BSRGAN` and `RealESRGAN` generators
(fetched by `code/probe0_sr.py`; see the comments there for the expected paths).

## Paths you must edit before running

Every path is a module-level constant, so there are only a few of them.

`code/dcdc_scan.py`

```python
SR4SOD  = "/path/to/your/sr4sod"     # dataset root
HR_IMG  = f"{SR4SOD}/_data/val/val/images"
HR_LBL  = "... /labels/val"
```

`code/tv_sweep.py`

```python
SR4SOD    = "/path/to/your/sr4sod"
RUNS_OLD  = "/path/to/a2_runs"       # detector runs
VIEW_ROOT = "/path/to/tv_views"      # cached degraded views
RUN_ROOT  = "/path/to/tv_runs"
W_YOLO    = f"{SR4SOD}/_weights/yolov8n.pt"
```

`dcdc_scan.py` additionally reads `A2_LABEL_DIR`, `A2_VIEW_ROOT`, `A2_RUN_ROOT` from the
environment if you prefer not to edit the file.

## Reproducing the experiments

```bash
# 4 arms x 6 seeds x 16 deployment conditions on AI-TOD (the main table)
python code/tv_sweep.py --only tv_A,tv_B0,tv_B2,tv_D0 \
    --seeds 2026,2027,2028,2029,2030,2031 --jsonl results/tv_sweep.jsonl --scan-n 200

# deployment-operator experiment (Table 6 / Figure 4)
python code/tv_deploy.py

# 2x2 factor decomposition and its verdict
python code/tv_factor2x2.py && python code/tv_factor_verdict.py

# cross-architecture replication on YOLO11n
python code/arch_scan.py && python code/arch_verdict.py

# cross-dataset replication
python code/dior_l1.py && python code/dior_l2.py
python code/soda_tile.py --plan && python code/soda_tile.py --crop && python code/soda_l2.py
```

Each driver is resumable: results are appended to JSON Lines and finished units are skipped.
Training is skipped when `results.csv` already holds 31 rows (header + 30 epochs), which is
stricter than testing for `best.pt` — a run interrupted mid-training leaves an early
`best.pt` that must not be reused.

## Result manifest — where each number in the paper comes from

| Paper item | File |
|---|---|
| Table 3, Table 5 (accuracy loss by condition) | `results/tv_deploy.json` |
| Table 6, Figure 4 (deployment operators vs PSNR) | `results/tv_deploy.json` |
| Table 2 (per-seed blur sensitivity, 24 rows) | `results/tab2_full.csv` |
| Table 7, Table 8 (factor contrasts) | `results/factor_verdict.json` |
| Table 9 (2x2 main effects and interaction, 16 conditions) | `results/tv_factor2x2.json` |
| Table 10, Table 11, Figure 6 (SR engines, dose-response slope) | `results/eng_clean_slope.json`, `results/engine_check.json` |
| Table 13, Table 14 (closure) | `results/tv_deploy.json` |
| Table 15, Figure 7 (crossing between D0 and B2) | `results/tv_deploy.json` |
| §6.4 SODA-A replication | `results/soda_l2.json`, `results/soda_l2_endpoints.json` |
| §6.5 DIOR replication | `results/dior_l1.json`, `results/dior_l2.json` |
| §6.6 YOLO11n replication | `results/tv_sweep.yolo11n.jsonl`, `results/tv_sweep.yolo11n.floor.jsonl`, `results/arch_verdict.json`, `results/tv_deploy.yolo11n.json` |
| §4.3.1 multiple-comparison correction | `results/_an1_bhfdr.json`, `verdict/AN1_BHFDR_REPORT.md` |

`floor` files hold repeated measurements of the same reference condition and are used to
estimate the implementation-noise floor `SD_real`. They are kept separate from the main
JSON Lines on purpose: mixing them in would overwrite the reference cell.

## Statistics in one line

Paired tests at the **seed** level (n = 6, df = 5, |t| = 2.571 at 95 %), 95 % CI, Cohen's
d_z, and Benjamini–Hochberg FDR over the family of 11 contrasts x 16 conditions (m = 176,
q = 0.05). Equivalence-style conclusions use TOST with a pre-declared bound. Decision rules
were written into `prereg/` before the runs and applied verbatim; the verdict scripts contain
no thresholds of their own.

## Figures

`figures/code/` regenerates every figure from `results/`. Figure 2 and Figure 3 use real
image crops produced end-to-end by the same pipeline (degradation → BSRGAN x4 → YOLOv8n), not
schematic drawings; `gen_fig02_assets.py` documents how the crop window was selected
(densest 400 px window, deterministic rule).

## License

MIT for the code. The datasets keep their own licences.
