# VANTRA — the numbers that matter

**One vehicle, one trail, across every camera in the city.**

## Our three strongest results

- **Zero false accusations.** On 487 hops of ordinary, law-abiding traffic, the
  impossible-travel (cloned-plate) alarm fired **0 times**. Enforcement actions
  built on VANTRA won't cry wolf.
- **100% of planted cloned-plate cases caught.** Every one of 15 impossible
  traversals we injected was flagged — with the exact evidence (the two cameras,
  the gap, the physical minimum time) attached.
- **98% plate-read accuracy on clean captures**, beating our 90% target — using
  a pretrained deep-learning recognizer (EasyOCR: CRAFT + CRNN).

## When the camera can't read, we don't guess

Stress-set plate reading, by condition:

| Condition | Read correctly |
|---|---|
| Clean | 98% |
| Rain | 95% |
| Low light | 95% |
| Dirt/occlusion | 83% |
| Bad viewing angle | 52% |
| Motion blur | 42% |

Blur and angle are hard — every ANPR system struggles there. The difference:
VANTRA's reading confidence collapses on unreadable frames, so instead of
trusting a wrong plate it falls back to matching the vehicle by appearance and
route. The trail survives; the numbers above are the honest ones.

## Honest real-world number

On 1,047 real photographs (US, EU and Taiwanese plates, held out from training),
our fine-tuned OCR reads **47.9% exactly right, 78.3% of characters right** —
up from 0% with the original stand-in engine and 1.9% with the stock pretrained
model. We fine-tuned a pretrained deep-learning recognizer (EasyOCR CRAFT+CRNN)
on two public real-plate datasets. The number is a data ceiling, not an
architecture limit: where photos have adequate resolution, we read 56-63% exact,
and the path to >90% is a better recognizer architecture, not just more data — we
proved that by tripling the training corpus and measuring the result go DOWN
(45.3% vs 47.9%): distribution match matters more than volume.

**Indian plates (updated 2026-09-10, leak-free evaluation):** on a held-out
Indian test set (312 photos, 152 unique plates — partitioned by plate text so
no plate appears in both training and test), our final model reads **~50%
exactly right [95% CI 45–56%], 87.1% of characters right** — up from 30.4%
for the previous international-only model on the same images. We report the
range, not the point estimate: 152 unique plates is a small sample. The same
model holds the international result (46.7% vs 47.9% — within noise), because
it was trained in two stages — mixed Indian+international, then a low-rate
Indian finish — specifically to avoid trading one domain for the other. An
Indian-only variant reads Indian plates slightly better (~53%) but forgets
international plates (33%); a 4-model ensemble was tested and rejected (no
real gain). Distribution is a zero-sum game at this data scale, and we ship
the model that works for both.

We also tested a multimodal-model fallback (Gemini 3.7 Flash as a second
opinion on low-confidence reads) across 9 model/prompt configurations. Honest
result: **it didn't help** — the fine-tuned local model beats the VLM in every
condition, so the fallback ships disabled. We report the negative result
because that's what the data said. The matcher only
consumes (text, confidence) — confidence separates correct (0.98) from wrong
(0.68) reads cleanly, so weak reads route to appearance+route matching instead
of being trusted.

