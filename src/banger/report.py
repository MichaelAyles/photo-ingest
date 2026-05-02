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
    aesthetic_source: str | None  # "prompts" | "head"
    thumb_b64: str
    cluster_id: int | None = None  # None if not in a multi-frame cluster
    cluster_size: int = 1
    cluster_best: bool = True  # False = suppressed sibling of a burst best
    scene_preset: str | None = None  # picked darktable preset name
    scene_score: float | None = None
    scene_fell_back: bool = False


def encode_thumbnail_bytes(preview: np.ndarray) -> bytes:
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
    return buf.tobytes()


def encode_thumbnail(preview: np.ndarray) -> str:
    return base64.b64encode(encode_thumbnail_bytes(preview)).decode("ascii")


def write_report(out_path: Path, rows: list[Row], threshold: float) -> None:
    kept = [r for r in rows if r.sharpness >= threshold and r.cluster_best]
    suppressed = [r for r in rows if r.sharpness >= threshold and not r.cluster_best]
    rejected = [r for r in rows if r.sharpness < threshold]
    kept.sort(
        key=lambda r: (r.aesthetic if r.aesthetic is not None else r.sharpness),
        reverse=True,
    )
    suppressed.sort(
        key=lambda r: (r.cluster_id or 0, -(r.aesthetic if r.aesthetic is not None else 0))
    )
    rejected.sort(key=lambda r: r.sharpness, reverse=True)
    ordered = kept + suppressed + rejected

    sharp_scores = [r.sharpness for r in rows]
    aesthetic_scores = [r.aesthetic for r in rows if r.aesthetic is not None]
    summary = {
        "total": len(rows),
        "kept": len(kept),
        "suppressed": len(suppressed),
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
    out_path.write_text(
        _render(ordered, summary, len(kept), len(kept) + len(suppressed)),
        encoding="utf-8",
    )


def _render(rows: list[Row], summary: dict, kept_after: int, suppress_after: int) -> str:
    cards = []
    for i, r in enumerate(rows):
        if i == kept_after and summary["suppressed"]:
            cards.append(
                f'<div class="divider">↓ suppressed by burst dedup '
                f'({summary["suppressed"]}) ↓</div>'
            )
        if i == suppress_after and summary["rejected"]:
            cards.append(
                f'<div class="divider">↓ below sharpness threshold '
                f'({summary["threshold"]:.0f}) ↓</div>'
            )
        if i < kept_after:
            section = "keep"
        elif i < suppress_after:
            section = "suppressed"
        else:
            section = "reject"
        cards.append(_card(r, section=section))
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
        suppressed=summary["suppressed"],
        rejected=summary["rejected"],
        threshold=summary["threshold"],
        sharp_min=summary["sharp_min"],
        sharp_median=summary["sharp_median"],
        sharp_max=summary["sharp_max"],
        aesthetic_line=aesthetic_line,
        cards="\n".join(cards),
    )


def _card(row: Row, section: str = "keep") -> str:
    if section == "keep":
        flag_class, flag_label = "keep", "KEEP"
    elif section == "suppressed":
        flag_class, flag_label = "suppressed", "DUPE"
    else:
        flag_class, flag_label = "reject", "REJECT"
    stem = html.escape(row.frame.display_name)
    kind = html.escape(row.frame.kind)
    if row.aesthetic is not None:
        source_tag = (
            f'<span class="src-{row.aesthetic_source}">{row.aesthetic_source}</span>'
            if row.aesthetic_source
            else ""
        )
        primary = f'<span class="primary">{row.aesthetic:.2f}</span>{source_tag}'
    else:
        primary = '<span class="primary muted">—</span>'
    breakdown_html = _breakdown(row.aesthetic_breakdown) if row.aesthetic_breakdown else ""
    cluster_html = ""
    if row.cluster_size > 1:
        if row.cluster_best:
            cluster_html = (
                f'<span class="cluster">best of {row.cluster_size} burst</span>'
            )
        else:
            cluster_html = (
                f'<span class="cluster">dupe of cluster {row.cluster_id} '
                f'(n={row.cluster_size})</span>'
            )
    scene_html = ""
    if row.scene_preset:
        suffix = " (default)" if row.scene_fell_back else ""
        scene_html = (
            f'<span class="scene">{html.escape(row.scene_preset)}{suffix}</span>'
        )
    return (
        f'<figure class="card {flag_class}">'
        f'<img loading="lazy" src="data:image/jpeg;base64,{row.thumb_b64}" alt="{stem}">'
        f'<figcaption>'
        f'<div class="head"><span class="stem">{stem}</span>{primary}</div>'
        f'<div class="meta">'
        f'<span class="kind">{kind}{cluster_html}{scene_html}</span>'
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
  .card.suppressed {{ opacity: .65; border-color: #3a3a4a; }}
  .badge.suppressed {{ background: #2e2e3a; color: #c0c0e0; }}
  .cluster {{ font-size: .65rem; color: #aaa; margin-left: .35rem; padding: 0 .35rem; background: #222; border-radius: 2px; }}
  .scene {{ font-size: .65rem; color: #c9c; margin-left: .35rem; padding: 0 .35rem; background: #2a1f2a; border-radius: 2px; font-family: ui-monospace, monospace; }}
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
  .src-prompts {{ font-size: .55rem; color: #888; margin-left: .35rem; text-transform: uppercase; letter-spacing: .04em; }}
  .src-head {{ font-size: .55rem; color: #c9d; margin-left: .35rem; text-transform: uppercase; letter-spacing: .04em; }}
</style>
</head>
<body>
<header>
  <h1>banger report</h1>
  <div class="summary">
    {total} frames &middot; {kept} kept &middot; {suppressed} suppressed (dupes) &middot; {rejected} rejected (threshold {threshold:.0f})<br>
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
