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
    sharpness: float
    aesthetic: float | None
    aesthetic_breakdown: dict[str, float] | None
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
    kept = [r for r in rows if r.sharpness >= threshold]
    rejected = [r for r in rows if r.sharpness < threshold]
    kept.sort(
        key=lambda r: (r.aesthetic if r.aesthetic is not None else r.sharpness),
        reverse=True,
    )
    rejected.sort(key=lambda r: r.sharpness, reverse=True)
    ordered = kept + rejected

    sharp_scores = [r.sharpness for r in rows]
    aesthetic_scores = [r.aesthetic for r in rows if r.aesthetic is not None]
    summary = {
        "total": len(rows),
        "kept": len(kept),
        "rejected": len(rejected),
        "threshold": threshold,
        "sharp_min": min(sharp_scores) if sharp_scores else 0.0,
        "sharp_median": statistics.median(sharp_scores) if sharp_scores else 0.0,
        "sharp_max": max(sharp_scores) if sharp_scores else 0.0,
        "aesthetic_min": min(aesthetic_scores) if aesthetic_scores else None,
        "aesthetic_median": statistics.median(aesthetic_scores) if aesthetic_scores else None,
        "aesthetic_max": max(aesthetic_scores) if aesthetic_scores else None,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_render(ordered, summary, len(kept)), encoding="utf-8")


def _render(rows: list[Row], summary: dict, divider_after: int) -> str:
    cards = []
    for i, r in enumerate(rows):
        if i == divider_after and summary["rejected"]:
            cards.append(
                f'<div class="divider">↓ below sharpness threshold '
                f'({summary["threshold"]:.0f}) ↓</div>'
            )
        cards.append(_card(r, is_keep=(i < divider_after)))
    if summary["aesthetic_min"] is not None:
        aesthetic_line = (
            f'aesthetic min {summary["aesthetic_min"]:.2f} / '
            f'median {summary["aesthetic_median"]:.2f} / '
            f'max {summary["aesthetic_max"]:.2f}'
        )
    else:
        aesthetic_line = "aesthetic: —"
    return _TEMPLATE.format(
        total=summary["total"],
        kept=summary["kept"],
        rejected=summary["rejected"],
        threshold=summary["threshold"],
        sharp_min=summary["sharp_min"],
        sharp_median=summary["sharp_median"],
        sharp_max=summary["sharp_max"],
        aesthetic_line=aesthetic_line,
        cards="\n".join(cards),
    )


def _card(row: Row, is_keep: bool) -> str:
    flag_class = "keep" if is_keep else "reject"
    flag_label = "KEEP" if is_keep else "REJECT"
    stem = html.escape(row.frame.stem)
    kind = html.escape(row.frame.kind)
    if row.aesthetic is not None:
        primary = f'<span class="primary">{row.aesthetic:.2f}</span>'
    else:
        primary = '<span class="primary muted">—</span>'
    breakdown_html = _breakdown(row.aesthetic_breakdown) if row.aesthetic_breakdown else ""
    return (
        f'<figure class="card {flag_class}">'
        f'<img loading="lazy" src="data:image/jpeg;base64,{row.thumb_b64}" alt="{stem}">'
        f'<figcaption>'
        f'<div class="head"><span class="stem">{stem}</span>{primary}</div>'
        f'<div class="meta">'
        f'<span class="kind">{kind}</span>'
        f'<span class="extras">sharp {row.sharpness:.0f} '
        f'<span class="badge {flag_class}">{flag_label}</span></span>'
        f'</div>'
        f'{breakdown_html}'
        f'</figcaption>'
        f'</figure>'
    )


def _breakdown(b: dict[str, float]) -> str:
    rows = []
    for prompt, sim in b.items():
        rows.append(
            f'<tr><td>{html.escape(prompt)}</td>'
            f'<td class="num">{sim:.4f}</td></tr>'
        )
    return (
        '<details class="breakdown"><summary>prompt sims</summary>'
        f'<table>{"".join(rows)}</table>'
        '</details>'
    )


_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>banger report</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font-family: ui-sans-serif, system-ui, sans-serif; margin: 1rem; background: #111; color: #eee; }}
  header {{ margin-bottom: 1rem; }}
  h1 {{ margin: 0 0 .25rem; font-size: 1.1rem; }}
  .summary {{ font-size: .8rem; color: #aaa; line-height: 1.5; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: .75rem; }}
  .card {{ margin: 0; background: #1a1a1a; border-radius: 6px; overflow: hidden; border: 1px solid #2a2a2a; }}
  .card.reject {{ opacity: .55; }}
  .card img {{ width: 100%; display: block; aspect-ratio: 3/2; object-fit: cover; }}
  figcaption {{ padding: .4rem .55rem; font-size: .8rem; }}
  figcaption .head, figcaption .meta {{ display: flex; justify-content: space-between; align-items: baseline; gap: .5rem; }}
  figcaption .meta {{ font-size: .7rem; color: #888; margin-top: .2rem; }}
  .stem {{ font-family: ui-monospace, monospace; }}
  .primary {{ font-variant-numeric: tabular-nums; font-weight: 600; color: #ddd; }}
  .primary.muted {{ color: #555; font-weight: normal; }}
  .badge {{ font-size: .65rem; padding: .05rem .35rem; border-radius: 3px; letter-spacing: .03em; margin-left: .35rem; }}
  .badge.keep {{ background: #1e3a1e; color: #8fdc8f; }}
  .badge.reject {{ background: #3a1e1e; color: #dc8f8f; }}
  .divider {{ grid-column: 1 / -1; padding: .6rem; text-align: center; color: #c97; border-top: 1px dashed #555; border-bottom: 1px dashed #555; margin: .25rem 0; font-size: .85rem; letter-spacing: .05em; }}
  .breakdown {{ margin-top: .35rem; font-size: .65rem; color: #aaa; }}
  .breakdown summary {{ cursor: pointer; color: #888; }}
  .breakdown table {{ width: 100%; border-collapse: collapse; margin-top: .2rem; }}
  .breakdown td {{ padding: .1rem 0; }}
  .breakdown td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
</style>
</head>
<body>
<header>
  <h1>banger report</h1>
  <div class="summary">
    {total} frames &middot; {kept} kept &middot; {rejected} rejected (threshold {threshold:.0f})<br>
    sharpness min {sharp_min:.0f} / median {sharp_median:.0f} / max {sharp_max:.0f}<br>
    {aesthetic_line}
  </div>
</header>
<main class="grid">
{cards}
</main>
</body>
</html>
"""
