# Experimental protocol (one page)

Everything below is fixed before the runs and is not tuned afterwards. The corresponding
pre-registration documents live in `../prereg/`.

## 1. Task and pipeline

```
HR patch (800 x 800)
   -> degradation simulator        (blur sigma, additive noise, JPEG round trip)
   -> LR patch (200 x 200)         (4x INTER_AREA downsampling)
   -> SR frontend (BSRGAN x4)      -> 800 x 800
   -> single-stage detector (YOLOv8n, 30 epochs, batch 8)
```

The deployment pipeline is identical. Only the training-time degradation simulator differs
between arms, so the inference graph, the parameter count and the data flow are unchanged.

## 2. Size binning

Short side of the box, computed **after** casting coordinates to float32:

```python
min(x2 - x1, y2 - y1)          # float32 first, then min
```

Bin edges: `[0, 2, 4, 6, 8, 12, 16, 24, 32, 64, inf)` (imported from `probe0_common.py`;
there is exactly one definition in this repository). The cast order matters: with float64
residuals, integer-pixel boxes drift across bin boundaries and bin counts move by about 7 %.

## 3. Sixteen deployment conditions

```
{blur 2.0, blur 0.2} x {noise 0.003, 0.012, 0.024, 0.096}          -> 8 conditions
{noise 0.096, noise 0.003} x {blur 0.0, 0.6, 1.2, 2.4}             -> 8 conditions
```

Degradation details:

| Knob | Setting |
|---|---|
| Blur | Gaussian, `sigma` in pixels |
| Noise | additive Gaussian, standard deviation in [0, 1] intensity |
| **Shot noise** | **disabled** (`shot=False`), so noise is purely additive |
| Downsampling | `cv2.INTER_AREA` |
| JPEG | applied at **training time only**; the evaluation path uses `jpeg=None` |

JPEG is a training-distribution variable, not an evaluation variable. This is why the
16-condition grid varies blur and noise only.

## 4. Four arms

| Arm | Training-time degradation |
|---|---|
| `A` | resampling only (blur sigma = 0) |
| `B0` | blur sigma ~ U(0, 0.05) + noise + JPEG |
| `B2` | blur sigma ~ U(0.2, 2.0) + noise + JPEG |
| `D0` | blur sigma ~ U(0.2, 2.0), no noise, no JPEG |

The 2x2 factor design is `{narrow, wide} x {without, with noise+JPEG}`. The extra arms
`N2` (noise only) and `J2` (JPEG only) isolate the two components of the second factor.

## 5. Statistics

* Unit of replication is the **training run (random seed)**, not the image. Images inside one
  run share a single set of weights, so image-level sampling would fold weight variance into
  sampling variance and inflate significance.
* Six seeds: 2026-2031. Paired across conditions, per seed, never pooled.
* n = 6, df = 5, two-sided 95 % critical value **|t| = 2.571**; 95 % CI; Cohen's d_z.
* Multiple comparisons: Benjamini-Hochberg FDR over the full family (11 contrasts x 16
  conditions, m = 176, q = 0.05).
* Equivalence-type statements use **TOST** with a bound fixed in advance. A non-significant
  paired test is reported as "not significant", not as "no effect".
* `MDE = t_crit * SD_delta / sqrt(n)`; `SNR = |effect| / MDE`.

## 6. Paired comparability

Every comparison between arms or between detectors uses the **same degradation tag**, so the
two models are evaluated on pixel-identical inputs. Degradation randomness is derived from
`arm_rng(stem, tag)`, therefore changing the tag changes the images and breaks pairing.

## 7. Implementation-noise floor

For each (arm, seed) the reference condition is measured three times with different tag
bases. The spread of these repeats gives `SD_real`, the implementation-noise floor. It is
reported next to every effect size so that a small effect can be compared against the noise
of the measurement itself rather than against zero.

## 8. Known limitation in the SODA-A evaluation

`eval_arm()` skips images that carry no ground-truth box (`if not gt: continue`). On SODA-A
250 of the 500 selected tiles are empty, so those tiles contribute neither true positives nor
false positives. Consequences:

* differences between arms are **unbiased**, because all arms see the same tile set;
* absolute mAP is **optimistically biased** and must not be compared across datasets;
* the background false-positive rate is a relative, arm-to-arm quantity only.
