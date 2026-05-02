import base64
import html
import statistics
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from banger.frames import Frame

THUMB_LONG_EDGE = 480
THUMB_JPEG_QUALITY = 78


@dataclass
class Row:
    frame: Frame
    score: float
    thumb_b64: str


def encode_thumbnail(preview: np.ndarray) -> str:
    h, w = preview.shape[:2]
    scale = THUMB_LONG_EDGE / max(h, w)
    if scale < 1.0:
        preview = cv2.resize(
            preview,
            (int(round(w * scale)), int(round(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
    ok, buf = cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, THUMB_JPEG_QUALITY])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def write_report(out_path: Path, rows: list[Row], threshold: float) -> None:
    rows = sorted(rows, key=lambda r: r.score, reverse=True)
    scores = [r.score for r in rows]
    kept = sum(1 for s in scores if s >= threshold)
    summary = {
        "total": len(rows),
        "kept": kept,
        "rejected": len(rows) - kept,
        "threshold": threshold,
        "min": min(scores) if scores else 0.0,
        "median": statistics.median(scores) if scores else 0.0,
        "max": max(scores) if scores else 0.0,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_render(rows, summary), encoding="utf-8")


def _render(rows: list[Row], summary: dict) -> str:
    cards = []
    divider_emitted = False
    for r in rows:
        is_keep = r.score >= summary["threshold"]
        if not is_keep and not divider_emitted:
            cards.append(
                f'<div class="divider">↓ below threshold ({summary["threshold"]:.0f}) ↓</div>'
            )
            divider_emitted = True
        cards.append(_card(r, is_keep))
    cards_html = "\n".join(cards)
    return _TEMPLATE.format(
        total=summary["total"],
        kept=summary["kept"],
        rejected=summary["rejected"],
        threshold=summary["threshold"],
        min_=summary["min"],
        median=summary["median"],
        max_=summary["max"],
        cards=cards_html,
    )


def _card(row: Row, is_keep: bool) -> str:
    flag_class = "keep" if is_keep else "reject"
    flag_label = "KEEP" if is_keep else "REJECT"
    stem = html.escape(row.frame.stem)
    kind = html.escape(row.frame.kind)
    return (
        f'<figure class="card {flag_class}">'
        f'<img loading="lazy" src="data:image/jpeg;base64,{row.thumb_b64}" alt="{stem}">'
        f'<figcaption>'
        f'<span class="stem">{stem}</span>'
        f'<span class="score">{row.score:.1f}</span>'
        f'<span class="badge {flag_class}">{flag_label}</span>'
        f'<span class="kind">{kind}</span>'
        f'</figcaption>'
        f'</figure>'
    )


_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>banger sharpness report</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font-family: ui-sans-serif, system-ui, sans-serif; margin: 1rem; background: #111; color: #eee; }}
  header {{ margin-bottom: 1rem; }}
  h1 {{ margin: 0 0 .25rem; font-size: 1.1rem; }}
  .summary {{ font-size: .85rem; color: #aaa; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: .75rem; }}
  .card {{ margin: 0; background: #1a1a1a; border-radius: 6px; overflow: hidden; border: 1px solid #2a2a2a; }}
  .card.reject {{ opacity: .55; }}
  .card img {{ width: 100%; display: block; aspect-ratio: 3/2; object-fit: cover; }}
  figcaption {{ padding: .4rem .55rem; display: grid; grid-template-columns: 1fr auto; gap: .15rem .5rem; font-size: .8rem; align-items: baseline; }}
  .stem {{ font-family: ui-monospace, monospace; }}
  .score {{ font-variant-numeric: tabular-nums; color: #ddd; text-align: right; }}
  .badge {{ font-size: .65rem; padding: .05rem .35rem; border-radius: 3px; letter-spacing: .03em; }}
  .badge.keep {{ background: #1e3a1e; color: #8fdc8f; }}
  .badge.reject {{ background: #3a1e1e; color: #dc8f8f; }}
  .kind {{ font-size: .7rem; color: #888; text-align: right; }}
  .divider {{ grid-column: 1 / -1; padding: .6rem; text-align: center; color: #c97; border-top: 1px dashed #555; border-bottom: 1px dashed #555; margin: .25rem 0; font-size: .85rem; letter-spacing: .05em; }}
</style>
</head>
<body>
<header>
  <h1>banger sharpness report</h1>
  <div class="summary">
    {total} frames &middot; {kept} kept &middot; {rejected} rejected &middot;
    threshold = {threshold:.1f} &middot;
    min {min_:.1f} / median {median:.1f} / max {max_:.1f}
  </div>
</header>
<main class="grid">
{cards}
</main>
</body>
</html>
"""
