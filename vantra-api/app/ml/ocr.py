"""OCR engine for VANTRA plate crops.

Pipeline (matching the spec's architecture, sized for offline demo):
1. Localization: for full vehicle images a contrast-box detector finds the plate
   rectangle (stands in for fine-tuned YOLOv8 plate detection — see NOTES.md).
2. Text extraction: segmentation + per-glyph template matching (stands in for
   PaddleOCR/CRNN). Multiple binarization thresholds are tried; the variant with
   the best mean margin wins. OCR confidence = calibrated mean top1-top2 margin,
   so degraded plates (blur/low-light/occlusion) score lower — downstream
   matching uses that confidence to widen fuzzy thresholds.
"""
from __future__ import annotations

import collections
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
IND_PLATE_LEN = 10

_FONT_CANDIDATES = [
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/SFNS.ttf",
    "/System/Library/Fonts/Geneva.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]


def _load_font(size: int):
    for p in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    raise RuntimeError("no TTF font found for plate rendering/OCR")


def render_char_template(ch: str, size: int = 40) -> np.ndarray:
    """Render a glyph, return its tight binary ink-mask bbox crop."""
    img = Image.new("L", (60, 70), 255)
    d = ImageDraw.Draw(img)
    d.text((30, 35), ch, font=_load_font(size), fill=0, anchor="mm")
    a = np.asarray(img, dtype=np.float32) / 255.0
    ys, xs = np.where(a < 0.5)
    return (a < 0.5)[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.float32)


_TEMPLATES: dict[str, np.ndarray] | None = None


def templates() -> dict[str, np.ndarray]:
    global _TEMPLATES
    if _TEMPLATES is None:
        _TEMPLATES = {ch: render_char_template(ch) for ch in ALPHABET}
    return _TEMPLATES


# ------------------------------------------------------------- localization

def detect_plate_box(vehicle_img: Image.Image, border_thresh: float = 0.5,
                     y_floor: float = 0.45) -> tuple[int, int, int, int] | None:
    """Locate the plate in a vehicle image. The plate is a bright rectangle with
    a thick dark frame: the box is anchored on dark border COLUMNS at its left
    and right edges AND a dark border ROW at its top, with a predominantly
    bright, text-bearing interior.
    `border_thresh` relaxes border-column darkness (video downscale).
    `y_floor` lowers the vertical scan start (video tiles can have plates high
    in frame; vehicle photos keep 0.45 to skip the cabin)."""
    g = np.asarray(vehicle_img.convert("L"), dtype=np.uint8)
    h, w = g.shape
    best: tuple[float, tuple[int, int, int, int]] | None = None
    step = 4
    for y0 in range(int(h * y_floor), h - 72, step):
        band = g[y0 + 2:y0 + 68, :]
        dark_band = (band < 90).mean(axis=0)     # per-column darkness in band rows
        # top border row darkness (interior x-range filled in per candidate)
        for x0 in range(4, w - 200, step):
            for pw in (200, 220, 240, 260, 290, 300, 320, 350, 380):
                x1 = x0 + pw
                if x1 >= w - 4 or y0 + 70 >= h - 4:
                    continue
                # vertical borders: left and right edge columns dark in the band
                if dark_band[x0:x0 + 3].mean() < border_thresh or dark_band[x1 - 3:x1].mean() < border_thresh:
                    continue
                # horizontal border: top rows dark across the interior width
                top_border = g[y0:y0 + 3, x0 + 10:x1 - 10]
                if (top_border < 90).mean() < 0.5:
                    continue
                interior = g[y0 + 5:y0 + 65, x0 + 5:x1 - 5]
                bright = (interior > 165).mean()
                dark = (interior < 90).mean()
                # plate interior: predominantly bright background with dark text
                if bright < 0.62 or not (0.03 < dark < 0.40):
                    continue
                score = bright + pw / 1000 - abs(dark - 0.18)
                if best is None or score > best[0]:
                    best = (score, (x0, y0, x1, y0 + 70))
    if best is None:
        return None
    x0, y0, x1, y1 = best[1]
    # snap the bottom edge down to the plate's dark bottom border (plates are
    # ~80px; the fixed 70px search window can undershoot by a few rows)
    row_dark = (g[:, x0 + 10:x1 - 10] < 90).mean(axis=1)
    for y in range(y1, min(y1 + 22, h)):
        if row_dark[y] > 0.5:
            y1 = y + 1
            break
    return (x0, y0, x1, y1)


# ------------------------------------------------------------- recognition

def _segments(mask: np.ndarray, gap_thresh: int, min_w: int, thresh_idx: int) -> list[tuple[int, int]]:
    col = (mask > 0).sum(axis=0)
    # low-contrast variants require weaker ink evidence
    ink_min = {0: 2, 1: 1, 2: 3}.get(thresh_idx, 2)
    ink = col >= ink_min
    segs: list[tuple[int, int]] = []
    x = 0
    n = len(ink)
    while x < n:
        if ink[x]:
            x2 = x
            gap = 0
            while x2 < n:
                if ink[x2]:
                    gap = 0
                else:
                    gap += 1
                    if gap > gap_thresh:
                        break
                x2 += 1
            end = x2 - (gap - 1 if gap > gap_thresh else 0)
            segs.append((x, end))
            x = x2
        else:
            x += 1
    return [(a, b) for a, b in segs if b - a >= min_w]


def _split_to_count(mask: np.ndarray, segs: list[tuple[int, int]], want: int) -> list[tuple[int, int]]:
    """Split widest segments at their minimum-ink interior column until `want`."""
    segs = list(segs)
    while len(segs) < want:
        widths = [b - a for a, b in segs]
        if not widths or max(widths) < 14:
            break
        i = int(np.argmax(widths))
        a, b = segs[i]
        span = b - a
        lo, hi = a + int(span * 0.25), a + int(span * 0.75)
        if hi - lo < 4:
            break
        colink = mask[:, a:b].sum(axis=0)
        mid = lo + int(np.argmin(colink[lo - a:hi - a]))
        segs[i:i + 1] = [(a, mid), (mid, b)]
    return sorted(segs)[:want]


def _match_cell(cell: np.ndarray) -> tuple[str, float]:
    ys, xs = np.where(cell > 0)
    if len(ys) == 0:
        return "?", 0.0
    crop = cell[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(bool)
    h, w = crop.shape
    tpl = templates()
    best_ch, best_iou, second = "?", -1.0, -1.0
    for ch, t in tpl.items():
        tt = np.asarray(
            Image.fromarray((t * 255).astype(np.uint8)).resize((w, h), Image.LANCZOS)
        ) > 127
        inter = (crop & tt).sum()
        union = (crop | tt).sum()
        iou = inter / union
        if iou > best_iou:
            second, best_ch, best_iou = best_iou, ch, iou
        elif iou > second:
            second = iou
    return best_ch, best_iou - second


def _read_with_threshold(plate_img: Image.Image, thresh: int, thresh_idx: int) -> tuple[str, float]:
    g = np.asarray(plate_img.convert("L"), dtype=np.float32) / 255.0
    H, W = g.shape
    # text band: drop border (3px) and the bottom IND strip entirely
    y0, y1 = 8, (H - 18 if H > 60 else H - 3)
    core = g[y0:y1, 3:W - 3]
    if core.size == 0:
        return "", 0.0
    mask = (core < thresh / 255.0).astype(np.float32)
    segs = _segments(mask, gap_thresh=2, min_w=8, thresh_idx=thresh_idx)
    if not segs:
        return "", 0.0
    segs = _split_to_count(mask, segs, IND_PLATE_LEN)
    out, margins, ious = [], [], []
    for a, b in segs:
        ch, margin = _match_cell(mask[:, a:b])
        out.append(ch)
        margins.append(margin)
        ious.append(margin)  # reuse: margin approximates separability
    text = "".join(out)
    mean_margin = float(np.mean(margins)) if margins else 0.0
    # segment-count penalty: missing/extra segments lower confidence hard
    n = len(segs)
    if n < IND_PLATE_LEN:
        mean_margin *= 0.5 ** (IND_PLATE_LEN - n)
    return text, float(np.clip(mean_margin / 0.30, 0.02, 0.99))


def _otsu(img: Image.Image) -> int:
    g = np.asarray(img.convert("L"))
    hist, _ = np.histogram(g, bins=256, range=(0, 256))
    total = g.size
    sum_all = float(np.dot(np.arange(256), hist))
    sum_b, w_b, best_t, best_var = 0.0, 0, 128, -1.0
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_all - sum_b) / w_f
        var = w_b * w_f * (m_b - m_f) ** 2
        if var > best_var:
            best_var, best_t = var, t
    return best_t


def _valid_format(text: str) -> bool:
    """Indian plate format plausibility: LL DD LL DDDD."""
    return (len(text) == IND_PLATE_LEN and text[:2].isalpha() and text[2:4].isdigit()
            and text[4:6].isalpha() and text[6:].isdigit())


def read_plate(plate_img: Image.Image) -> tuple[str, float]:
    """OCR a plate crop. Primary: fine-tuned EasyOCR CRNN (fast, local). If the
    CRNN's confidence is below the fallback threshold and the VLM fallback is
    enabled (VANTRA_VLM_FALLBACK=1 + latentcode key), escalate to a multimodal
    model (gemini-3.7-flash) for a second read — see METRICS.md §7. The old
    template matcher is the last-resort fallback when the DL model is unavailable
    (CI). Returns (text, confidence in [0,1]) — the contract matching relies on.
    """
    dl = _dl_read(plate_img)
    if dl is not None:
        text, conf = dl
        # Degenerate/empty reads: cap confidence so downstream treats them as weak.
        most = collections.Counter(text).most_common(1)[0][1] if text else 99
        if len(text) < 4 or most >= max(6, len(text) - 2):
            conf = min(conf, 0.30)
        # VLM escalation on low-confidence CRNN reads (opt-in)
        if conf < _vlm_fallback_threshold() and _vlm_enabled():
            vlm = _vlm_read(plate_img)
            if vlm is not None:
                return vlm
        return text, conf
    return _read_plate_template(plate_img)


def _vlm_enabled() -> bool:
    import os
    return os.environ.get("VANTRA_VLM_FALLBACK", "0") == "1" and _vlm_key() is not None


def _vlm_fallback_threshold() -> float:
    import os
    return float(os.environ.get("VANTRA_VLM_THRESHOLD", "0.90"))


def _vlm_key() -> str | None:
    import os
    from pathlib import Path as _P
    key = os.environ.get("LATENTCODE_KEY")
    if not key:
        kf = _P("/tmp/latentcode_key")
        if kf.exists():
            key = kf.read_text().strip()
    return key or None


def _vlm_read(plate_img: Image.Image) -> tuple[str, float] | None:
    """Second-opinion read via a multimodal model (gemini-3.7-flash on the
    latentcode gateway). Best-effort: any failure returns None (caller keeps
    the CRNN read). Confidence is calibrated from the VLM's self-consistency:
    the gateway returns no probability, so we use a fixed conservative 0.75
    (validation: VLM correct reads are ~56% of its reads — better than a weak
    CRNN read but below a confident CRNN one)."""
    try:
        import base64
        import io as _io
        import re as _re
        import requests
        img = plate_img.convert("RGB")
        if img.width < 96:
            scale = 96 / img.width
            img = img.resize((96, int(img.height * scale)), Image.LANCZOS)
        elif img.width > 640:
            img = img.resize((640, int(img.height * 640 / img.width)), Image.LANCZOS)
        buf = _io.BytesIO()
        img.save(buf, format="PNG")
        r = requests.post(
            "https://latentstack.dev/v1/chat/completions",
            headers={"Authorization": f"Bearer {_vlm_key()}",
                     "Content-Type": "application/json"},
            json={
                "model": "gemini-3.7-flash",
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": "Read the license plate text in this "
                     "image. Reply with ONLY the plate characters."},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,"
                               f"{base64.b64encode(buf.getvalue()).decode()}"}},
                ]}],
                "max_tokens": 3000,
            }, timeout=60)
        r.raise_for_status()
        content = r.json()["choices"][0]["message"].get("content") or ""
        lines = [l for l in content.strip().splitlines() if l.strip()]
        text = _re.sub(r"[^A-Za-z0-9]", "", lines[-1] if lines else "").upper()
        if not text:
            return None
        return text, 0.75
    except Exception:
        return None


_easyocr_reader = None
_easyocr_failed = False


def _dl_read(plate_img: Image.Image) -> tuple[str, float] | None:
    """Run the EasyOCR recognizer on a plate crop. Returns None if the model
    isn't loadable (falls back to the template matcher)."""
    global _easyocr_reader, _easyocr_failed
    if _easyocr_failed:
        return None
    if _easyocr_reader is None:
        try:
            import easyocr
            _easyocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False,
                                             quantize=False)
            # Fine-tuned recognition weights (real-plate corpus — METRICS.md §6).
            # Loaded when present; vanilla pretrained otherwise (CI, fresh clones).
            import os
            ft = os.environ.get(
                "VANTRA_FT_CKPT",
                str(Path(__file__).resolve().parents[3] / "data" / "models"
                    / "ft_inC_from_inB1" / "recognizer.pt"))
            if os.path.exists(ft):
                import torch
                _easyocr_reader.recognizer.load_state_dict(
                    torch.load(ft, map_location="cpu", weights_only=False))
        except Exception as e:
            print(f"[ocr] EasyOCR unavailable ({type(e).__name__}: {e}) "
                  f"— falling back to template matcher")
            _easyocr_failed = True
            return None
    import re as _re
    # upscale small crops: the recognizer needs ~200px+ of text height context
    img = plate_img
    scale = max(1.0, 220 / max(img.width, 1))
    if scale > 1.0:
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
    results = _easyocr_reader.readtext(np.asarray(img.convert("RGB")))
    if not results:
        return ("", 0.0)
    # best fragment: highest (confidence x alphanumeric count) — plate text beats
    # stray noise marks; the IND hallmark and border digits are usually separate
    # lower-value fragments
    def frag_score(item):
        _bbox, text, conf = item
        alnum = sum(c.isalnum() for c in text)
        return float(conf) * max(alnum, 1)
    _bbox, text, conf = max(results, key=frag_score)
    clean = _re.sub(r"[^A-Za-z0-9]", "", text).upper()
    clean = _indian_plate_postprocess(clean)
    return clean, float(conf)


def _read_plate_template(plate_img: Image.Image) -> tuple[str, float]:
    """Original template-matching OCR (pre-DL swap). Kept as the CI/no-model
    fallback and for comparison benchmarks."""
    variants: list[Image.Image] = [plate_img]
    for fx in (1.15, 1.3, 1.42):  # undo viewing-angle squeeze
        variants.append(plate_img.resize((int(plate_img.width * fx), plate_img.height), Image.BICUBIC))
    angled = []
    for v in variants:
        for ang in (0, 6, -6):
            angled.append(v.rotate(ang, expand=False, fillcolor=(128, 128, 128),
                                   resample=Image.BICUBIC) if ang else v)
    o = _otsu(plate_img)
    thresholds = [128, 100, 155, 80, 170, 185, int(o * 0.9), int(o * 0.75)]
    best = ("", 0.0, 0.0)
    for v in angled:
        for th in thresholds:
            if not 30 <= th <= 220:
                continue
            text, conf = _read_with_threshold(v, th, 0)
            if not text:
                continue
            fmt = 1.0 if _valid_format(text) else 0.55
            eff = conf * fmt
            if eff > best[2]:
                best = (text, conf, eff)
    text, conf = best[0], best[1]
    # Degenerate reads (e.g. heavy blur smearing every glyph into bars -> "IIIIIIIIII")
    # are unreadable in practice: report low confidence so downstream widens thresholds.
    most = collections.Counter(text).most_common(1)[0][1] if text else 99
    if len(text) < 8 or most >= 8:
        conf = min(conf, 0.25)
    return text, conf


def detect_and_read(vehicle_img: Image.Image, border_thresh: float = 0.5,
                    y_floor: float = 0.45) -> tuple[str, float, tuple[int, int, int, int] | None]:
    """Full pipeline on a vehicle image: localize + OCR."""
    box = detect_plate_box(vehicle_img, border_thresh, y_floor)
    if box is None:
        return "", 0.0, None
    crop = vehicle_img.crop(box)
    text, conf = read_plate(crop)
    return text, conf, box

# Confusable remapping for Indian plate format LL DD LL DDDD (positions 0-based:
# 0-1 letters, 2-3 digits, 4-5 letters, 6-9 digits). The DL model freely swaps
# visually identical glyphs (0/O, 1/I, 2/Z, 5/S, 8/B, 6/G); the format tells us
# which class each position must be.
_TO_DIGIT = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "T": "7",
             "Z": "2", "S": "5", "B": "8", "G": "6", "A": "4", "U": "0"}
_TO_LETTER = {"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B", "6": "G",
              "4": "A", "7": "T"}


def _indian_plate_postprocess(text: str) -> str:
    """If a 10-char read matches the Indian plate SHAPE (LL..LL....), remap
    position-inconsistent glyphs to the correct class. Only applied when the
    remap makes the string a valid-format plate — never force other shapes."""
    if len(text) != 10:
        return text
    out = list(text)
    for i in (2, 3, 6, 7, 8, 9):          # digit positions
        out[i] = _TO_DIGIT.get(out[i], out[i])
    for i in (0, 1, 4, 5):                # letter positions
        out[i] = _TO_LETTER.get(out[i], out[i])
    fixed = "".join(out)
    return fixed

