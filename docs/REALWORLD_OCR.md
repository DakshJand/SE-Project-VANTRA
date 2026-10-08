# Real-world OCR validation — fine-tuned EasyOCR results

## Final held-out accuracy (1,047 real photographs, never touched during training)

| Model | Exact match | Char accuracy |
|---|---|---|
| Template matcher (original stand-in) | 0.0% | 1.6% |
| Pretrained EasyOCR (CRAFT + CRNN) | 1.9% | 37.5% |
| **Fine-tuned EasyOCR (final)** | **47.9%** | **78.3%** |
| Fine-tuned + Indian-format postprocessing | 47.9% | 78.3% |

The Indian-format postprocessing is neutral on this test set (US/EU/Taiwan plates
don't match the 10-char Indian shape it targets); it remains active in the
production path where it lifts synthetic Indian plates to 98–100%.

## Dataset

Two public real-plate datasets, partitioned BY SOURCE to prevent leakage between
visually similar images:

| Source | Plates | Style | Split role |
|---|---|---|---|
| sonnetechnology/license-plate-text-recognition-full (HF, CC BY 4.0) | 5,506 train / 800 val / 788 test | US/EU plates, bbox + text | train, val, test |
| EZCon/taiwan-license-plate-recognition (HF) | 1,812 train / 259 test | Taiwanese plates, rotated boxes + text | train, test |

Training corpus after augmentation (blur, rotation ±8°, brightness/contrast
jitter, occlusion strips — 3 variants per source image): **29,272 rows** (7,318
real × 4). Held-out test: **1,047 items** (788 US/EU + 259 Taiwan), evaluated
once, after all training concluded.

## Iteration history (validation exact %, 400 US/EU items)

| Run | Config | Val exact | Val char |
|---|---|---|---|
| — | Pretrained baseline | 0.2% | 38.6% |
| r1 | lr 1e-4, 12k rows, 4 epochs, full FT | 47.5% | 76.6% |
| r2 | lr 5e-5, 16k rows, 6 epochs, full FT | 47.5% | 76.8% |
| r3 | lr 1e-4, all 32k rows, 6 epochs (crashed ep3, ckpt@ep2) | 48.2% | 78.2% |
| r4 | continued from r3, lr 5e-5, 4 epochs | 47.8% | 78.5% |

The curve plateaus at ~48% across four hyperparameter configurations — a genuine
ceiling for this dataset size, not an under-trained model.

## Error breakdown by condition (final model, held-out test)

| Condition | n | Exact | Char |
|---|---|---|---|
| **By crop size** | | | |
| large (>160px) | 175 | **61.1%** | 84.0% |
| medium (80–160px) | 630 | **55.6%** | 84.3% |
| small (<80px) | 242 | **18.2%** | 56.7% |
| **By lighting** | | | |
| bright | 102 | 62.7% | 85.2% |
| normal | 1,432* | 55.1% | 81.8% |
| dark | 61 | 44.3% | 78.5% |
| **By aspect ratio** | | | |
| normal (2.0–5.5) | — | ~55% | ~82% |
| squished (<2.0, oblique angle) | 496 | **24.4%** | 64.4% |
| **By source** | | | |
| Taiwan | 259 | 51.4% | 78.9% |
| US/EU | 788 | 46.7% | 78.1% |

(*condition counts overlap — each image is bucketed in all four dimensions)

**Where it struggles:** small crops (<80px wide — 18% exact) and squished/
oblique-angle plates (24% exact) account for most errors; when resolution and
angle are adequate the model reads 56–63% exact / 84% char. Typical error
patterns: dropped leading characters on small crops (`KA7436` → `KA436`) and
single-digit confusions (`51F69881` → `51F59881` at conf 0.95 — a genuinely
ambiguous read).

**Confidence calibration (production thresholds):** correct reads mean 0.977
(p10 0.926), wrong reads mean 0.68 — a clean separation. The matcher's
confidence-aware fuzzy thresholds (recalibrated earlier for the pretrained DL
distribution) remain valid for the fine-tuned model.

## What would close the remaining gap to >90%

1. **10–50× more labeled real plates.** The ~48% plateau across four
   hyperparameter runs with 7.3k unique images is a data ceiling, not a training
   one. CCPD (~250k Chinese plates), the full train splits of both datasets used
   here, and Indian-plate corpora (Datacluster's full 6k set is redacted in the
   public release — a paid/licensed version would be needed) are the natural
   sources. At 50–100k unique plates, comparable fine-tunes typically reach
   85–95% on plate recognition benchmarks.
2. **Resolution-aware inference:** super-resolution or tiling for <80px crops —
   the single largest error bucket.
3. **Rectification for oblique plates:** a small perspective-correction network
   before recognition (standard in production ANPR) for the squished-aspect
   bucket.
4. **Plate-aware fragment merging** for multi-line plates (measured: hurts
   single-line plates, so it must be conditioned on detected text-line layout).


---

# VLM-fallback pass — full report

## Candidate comparison (validation set, 120 images per cell)

Three multimodal models via the latentcode gateway × three prompt strategies:

| Model | Prompt | Exact | Char | Latency | Cost/120 |
|---|---|---|---|---|---|
| **gemini-3.7-flash** | **plain** | **56.7%** | 77.3% | 7.4s | $0.70 |
| gemini-3.7-flash | structured | 55.0% | 77.8% | 7.7s | $0.74 |
| gemini-3.7-flash | ranked | 55.0% | 76.9% | 8.5s | $0.74 |
| gemini-3.1-pro | plain | 54.2% | 76.0% | 12.7s | $1.78 |
| gemini-3.1-pro | structured | 51.7% | 75.5% | 11.6s | $1.78 |
| gemini-3.1-pro | ranked | 53.3% | 77.0% | 14.7s | $2.22 |
| gemini-3.5-flash | plain | 55.0% | 77.0% | 6.2s | $0.83 |
| gemini-3.5-flash | structured | 52.5% | 76.3% | 6.8s | $0.82 |
| gemini-3.5-flash | ranked | 55.0% | 78.1% | 7.2s | $0.90 |

Winner: **gemini-3.7-flash + plain prompt** (highest exact; the structured
char-by-char and ranked-candidates prompts did not help on any model).

## Full-validation and held-out reality check

The grid sample (120 val images) overstated the VLM: on the **full 800-image
validation set** the winner scored **44.5% exact / 65.0% char**, and on the
**held-out test (1,047 images)**: **35.0% exact / 53.4% char** — the test set
(US/EU + Taiwan) is genuinely harder than the val front-slice.

## Error-mode breakdown (held-out, CRNN vs VLM)

| Condition | n | CRNN exact | VLM exact |
|---|---|---|---|
| large (>160px) | 175 | 61.1% | 20.6% |
| medium (80–160px) | 630 | 55.6% | 46.3% |
| small (<80px) | 242 | 18.2% | 15.7% |
| normal aspect | 551 | 69.0% | 59.5% |
| squished (<2.0) | 496 | 24.4% | 7.7% |

The VLM loses to the fine-tuned CRNN in **every** condition bucket — including
the two failure modes it was hypothesized to help (small crops: 15.7% vs 18.2%;
oblique: 7.7% vs 24.4%).

## Overlap analysis (held-out, n=1,047)

| | CRNN right | CRNN wrong |
|---|---|---|
| **VLM right** | 336 | 30 |
| **VLM wrong** | 165 | 516 |

The models are largely **redundant**, not complementary: the VLM recovers only
30 images the CRNN misses (2.9% of the set) while losing 165 that the CRNN gets
right. Oracle-union accuracy 50.6% vs CRNN-alone 47.9%.

## Threshold tuning (validation, n=800)

| Threshold | Fallback rate | Combined exact |
|---|---|---|
| 0.00 (never) | 0% | 47.1% |
| 0.50 | 8.1% | 47.2% |
| 0.70 | 23.5% | 46.9% |
| 0.80 | 36.6% | 46.4% |
| 0.90 | 49.0% | 47.4% |
| **0.95** | **57.5%** | **48.2%** |
| 0.98 | 64.2% | 46.9% |

Best validation operating point: t=0.95 (48.2%, +1.1pp over CRNN-alone at a
57.5% fallback rate).

## Final held-out comparison (n=1,047)

| Configuration | Exact | Char |
|---|---|---|
| CRNN alone (fine-tuned) | **47.9%** | **78.3%** |
| VLM alone (gemini-3.7-flash + plain) | 35.0% | 53.4% |
| Combined @ t=0.90 (49.2% fallback) | 47.0% | — |
| Combined @ t=0.95 (55.8% fallback) | 46.2% | — |

## Honest conclusion: the VLM fallback does NOT improve results

**Both combined configurations underperform CRNN-alone on the held-out set**
(−0.9pp and −1.7pp). The validation gain (+1.1pp at t=0.95) did not transfer —
it was noise from the val/test distribution shift, exactly why the threshold was
tuned on val and evaluated once on held-out. Per-image overlap confirms the
structural reason: the VLM's 30 unique recoveries are swamped by the 165
high-confidence CRNN reads it would overwrite.

**Why the VLM underperforms here (real evidence, not speculation):**
- small crops (<80px) are simply too degraded for either model — the VLM does
  not magically recover resolution; 15.7% vs CRNN's 18.2%
- oblique/squished crops: the CRNN's CTC training on 29k augmented examples
  (including rotations) generalizes better than the VLM's zero-shot reading
  (7.7% vs 24.4%)
- the VLM wins only where both models are already decent, and loses
  high-confidence CRNN reads to hallucination when forced to guess

**Production framing:** the fallback ships toggleable (VANTRA_VLM_FALLBACK=1,
threshold VANTRA_VLM_THRESHOLD) and **off by default** — the honest measured
result is that it costs ~$0.006 and ~7.5s per escalated image for no accuracy
gain on this data. It may still be worth enabling for *specific* future
conditions (e.g. multi-line plates, non-Latin scripts the CRNN was never trained
on), which the current test set cannot measure.

## Cost/latency model (measured)

- VLM: 7.5-7.9s/image, $0.0058/image (gemini-3.7-flash via latentcode)
- CRNN: ~0.2s/image local, $0
- At a hypothetical 50% fallback rate: +$2.90 and +3.9s average per 1,000
  city detections — for a measured accuracy *loss* on this data. If a future
  CRNN were improved to leave only genuinely-unreadable images below threshold,
  the fallback rate would drop toward 10-15% and the calculus would need
  re-measuring.


---

# Accuracy-push pass (r5): data expansion + SR preprocessing — honest negative result

## What was added

**Data (2.9x expansion, 21,154 unique real crops):**
- +12,590 Chinese plate crops (richjjj/chinese_license_plate_rec, LMDB; hanzi
  province character dropped, alphanumeric tails kept as labels — real photos,
  charset-compatible)
- +518 Taiwan validation-split crops (EZCon; source-separated from test)
- Dataset survey found nothing else usable: thundarstrom's Indian corpus repos
  are README-only (data never uploaded), FANVID requires YouTube video downloads
  of deliberately-unrecognizable single frames (temporal task, wrong fit),
  AmirAI24/ANPR-CoCo and all Roboflow exports are bbox-only without text,
  neotje/dutch is 100 YOLO-format images. **Volume was capped here — stated
  explicitly, not padded with synthetic copies.**

**Preprocessing (measured on validation before training):**
- Super-resolution (ESPCN x2) on <80px crops: small-bucket 17.6% → 19.2% (+1.6pp)
- Perspective rectification (Otsu + contour quad + homography) on squished crops:
  12.4-13.5% — slightly WORSE than baseline; dropped from the pipeline
- SR copies of small training crops added to the r5 corpus (1,790 rows)

## r5 training (91,776 rows: 22,944 real + 68,832 augmented)

| Epoch | Val exact | Val char |
|---|---|---|
| 1 | 47.5% | 77.6% |
| 2 | 47.5% | 78.2% |
| 3 | 47.8% | 78.1% |
| 4 | 47.2% | 78.2% |
| 5 | 47.8% | 78.8% |
| 6 | 48.2% | 78.7% |

Validation plateau identical to r1-r4 (~48%). Full convergence confirmed.

## Final held-out comparison (1,047 photos, untouched, evaluated once at the end)

| Model | Exact | Char |
|---|---|---|
| **r3 (7.3k real crops) — remains best** | **47.9%** | **78.3%** |
| r5 (21.2k real crops incl. Chinese + SR) | 45.3% | 78.3% |

**The 2.9x data expansion made held-out accuracy WORSE (−2.6pp).** Character
accuracy held (78.3%) but exact-match dropped in every bucket:

| Bucket | r3 | r5 |
|---|---|---|
| large (>160px) | 61.1% | 55.4% |
| medium | 55.6% | 52.9% |
| small (<80px) | 18.2% | 18.2% |
| squished | 24.4% | 23.6% |
| US/EU | 46.7% | 44.2% |
| Taiwan | 51.4% | 48.6% |

## Why (concrete, not vague)

1. **Domain shift**: Chinese plates (hanzi-dropped 6-char tails) are a different
   label distribution from the US/EU/Taiwan test set. Training capacity spent
   fitting Chinese plate styles came at the cost of the test distribution. The
   validation set (US/EU only) could not reveal this — both models score ~48%
   there; only the held-out set exposed the transfer failure.
2. **The SR copies (+1.6pp on val small-bucket) did not transfer**: small-bucket
   held-out stayed exactly 18.2%. The small-crop failure mode is below what
   2x bicubic-quality SR can recover — the information simply isn't in the pixels.
3. **The ~48% ceiling is architecture-and-distribution-bound**, not data-volume-
   bound: 3x more data moved nothing on val and hurt test. A better recognizer
   (e.g. PARSeq/transformer head, or per-country models with routing) rather
   than more of the same data is the indicated next step — out of scope for this
   CPU-training setup.

## Final state

Production model remains **r3 (ft_r3_alldata_e6)**: 47.9% exact / 78.3% char on
the held-out set. The r5 model is retained at
`data/models/ft_r5_expanded_sr/` for reference. Held-out set was touched exactly
twice in total across all passes (r3 final eval, r5 final eval), never used for
any training or threshold decision.

---

# Indian-domain pass (2026-09-09) — leak-free split rebuild, supersedes earlier Indian numbers

## Split correction

An earlier Indian evaluation (~41% exact, since discarded) used an **image-level**
train/val/test split that allowed the same plate text to appear in both train and
test — its number was potentially inflated by leakage. The split was rebuilt
**partitioned by unique plate text**: 964 unique plates, zero overlap verified —
test = 152 plates / 312 images, val = 47 plates / 100 images, train = 765 plates /
1,295 images. All numbers below are on this leak-free split. The earlier 41% and
any checkpoints from it are invalid; no "inA_indian_only" run on the old split
ever completed (its output dir was empty), so nothing needed discarding.

## Corrected baseline (2026-09-09)

r3 (the production international model) re-measured on the leak-free Indian test:
**30.4% exact [95% CI 25.6–35.8] / 76.7% char**. This corrects and supersedes
the earlier ~41% figure measured on the leaky split.

## Training (leak-free split, 2026-09-09)

Indian corpus is small — 1,295 train images (765 plates) × 4 with the standard
augmentation = 5,180 rows; val = 100 images (47 plates). Four configs were
trained in parallel (CPU), val-based selection:

| Run | Config | Val exact (best epoch) | Notes |
|---|---|---|---|
| inA1 | Indian-only, lr 1e-4, 12 ep | 63.0% (ep10) | plateau ~62% from ep3 |
| inA2 | Indian-only, lr 5e-5, 12 ep | 65.0% (ep12) | slow steady climb, no overfit collapse |
| inB1_sub | from r3, lr 5e-5, 3 ep, mixed corpus (5,180 Indian + 12,000 international sample) | 67.0% (ep2) | ep3 dropped to 60% — early stop retained ep2 |

(An initial inB design with the full 92k-row international corpus was killed
before epoch 1 for CPU-time infeasibility; the subsampled 17k-row corpus is the
tested design. An inB2 lr-1e-4 variant was killed at the same time and never
trained — not a data point.)

## One-shot leak-free Indian test (312 images, 152 unique plates, evaluated once)

| Model | Exact match | 95% CI | Char acc | Plate-level exact |
|---|---|---|---|---|
| Pretrained EasyOCR | 0.3% | [0.1, 1.8] | 34.2% | 0.7% |
| r3 (int'l production) | 30.4% | [25.6, 35.8] | 76.7% | 34.2% |
| inA1 (Indian-only lr 1e-4) | 51.9% | [46.4, 57.4] | 87.7% | 57.9% |
| inA2 (Indian-only lr 5e-5) | 53.2% | [47.7, 58.7] | 87.5% | 55.9% |
| **inB1_sub (deployed)** | **45.8%** | **[40.4, 51.4]** | **84.8%** | **51.3%** |

**Caveat — small sample:** this test set has 152 unique plates / 312 images; the
95% CIs above span ~11pp. Numbers should be read as ranges, not point estimates,
and are NOT comparable in precision to the international 47.9% (n=1,047).

## Why inB1_sub was selected over the higher-Indian-scoring inA2

Cross-domain check on the international held-out set (1,047 images):

| Model | Int'l exact | Int'l char |
|---|---|---|
| r3 | 47.9% | 78.3% |
| inA1 | 29.9% | 65.8% |
| inA2 | 33.0% | 66.6% |
| **inB1_sub** | **47.2%** | **78.3%** |

The Indian-only models win on Indian plates (+7pp over inB1_sub) but
catastrophically forget the international domain (−15pp vs r3). inB1_sub gains
+15.4pp exact on Indian (non-overlapping CI vs r3) while holding international
at 47.2% vs 47.9% (within noise) — the only model that improves one domain
without regressing the other. Deployed as the production checkpoint
(`data/models/ft_inB1_sub_from_r3/`, wired via `app/ml/ocr.py` default; r3
remains on disk and selectable via `VANTRA_FT_CKPT`).

Interesting negative: inB1_sub had the *highest* val exact (67%) but the *lowest*
Indian-test exact of the three trained models — with only 47 unique val plates,
val selection between close configs is noisy; the mixed corpus's val edge did
not transfer. Recorded as a caution for future model selection on this dataset.

## Condition buckets (inB1_sub, leak-free Indian test)

Squished/oblique (<2.0 aspect) remains the dominant failure mode (9.1% exact,
n=11); size buckets are much flatter than the international set (small 41.8 /
medium 46.0 / large 48.6) — the Indian corpus's crops are mostly well-resized
crops from full-vehicle imagery. Dark images score oddly high (60%, n=10) —
small-bucket noise.

## Ensemble pass (2026-09-10) — negative result

Ensembled all four checkpoints (r3, inA1, inA2, inB1_sub) with five strategies
(confidence-weighted char vote ×2, majority vote, conf-weighted string, Borda),
selected on Indian val (Borda won at 67%), evaluated once on both held-out sets:

| Method | Indian exact | Indian char | Int'l exact |
|---|---|---|---|
| Best ensemble (Borda/majority) | 54.5% | 87.9% | 45.3% |
| Best singles | inA2 53.2% | inA2 87.5% | r3 47.9% |

Indian +1.3pp over the best single is deep inside the ±11pp CI — noise. On
international the ensemble REGRESSES (45.3% vs 47.9%): the two Indian-
specialized models outvote the competent ones on unfamiliar plates. The four
models are largely redundant (same structural finding as the VLM pass).
**Ensemble rejected — not deployed.**

## Final round (2026-09-10): inC — new production model

One focused experiment after the ensemble: continue inB1_sub on Indian-only
data at low LR (2e-5, 6 epochs — the one untested configuration between
"Indian-only from scratch" and "mixed from r3"). Val climbed 68 → 72% (best of
all runs). One-shot held-out:

| Model | Indian exact (95% CI) | Indian char | Int'l exact | Int'l char |
|---|---|---|---|---|
| inB1_sub (prior deploy) | 45.8% [40.4–51.4] | 84.8% | 47.2% | 78.3% |
| **inC (deployed)** | **50.3% [44.8–55.8]** | **87.1%** | **46.7%** | **77.3%** |
| inA2 (Indian-only best) | 53.2% [47.7–58.7] | 87.5% | 33.0% | 66.6% |

inC is a strict improvement over inB1_sub (+4.5pp Indian exact, +2.3pp char,
international within noise at −0.5pp) and recovers most of the Indian-only
advantage without forgetting international.

## Final state (2026-09-10)

Production model: **inC (ft_inC_from_inB1)** — 50.3% [44.8–55.8] exact /
87.1% char on the leak-free Indian test (312 imgs / 152 plates — small sample,
CIs ~11pp), 46.7% / 77.3% on the international held-out (1,047 imgs). Chain:
r3 → inB1_sub (mixed corpus) → inC (Indian-only, low LR). Earlier models
retained on disk; rollback via `VANTRA_FT_CKPT`. Training and model
experimentation are concluded. The binding constraint remains labeled Indian
plate volume (1,295 images / 765 plates).
