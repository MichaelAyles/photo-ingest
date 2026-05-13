"""Hand-rolled darktable XMP sidecar writer.

No new dep. The XMP format is well-documented and small: it's RDF-XML
with a handful of attributes we care about (rating, label, hierarchical
subjects). darktable picks these up on its next library scan, Lightroom
and other DAMs read them too.

What we write:
- xmp:Rating       0-5 stars. Derived from the combined quality score's
                   rank in the input batch (top 10% = 5, next 20% = 4,
                   next 30% = 3, next 30% = 2, bottom 10% = 1). Frames
                   that failed any gate get 0.
- xmp:Label        a darktable colour label string ("red"/"yellow"/
                   "green"/"blue"/"purple") encoding the scene cluster.
- dc:subject       a flat tag list: banger/source/<aesthetic_source>,
                   banger/scene/<preset>, banger/rank/<N>. Lets the user
                   filter in any DAM.

The "no copying" angle is the headline workflow win over facet/Aftershoot:
the source folder stays untouched and the .xmp lives next to its source
file, so a roundtrip through Lightroom or darktable picks up our triage
without ever moving a JPEG.
"""

from __future__ import annotations

import logging
import xml.sax.saxutils as sx
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from banger.report import Row

log = logging.getLogger("banger")


# Eight colour labels we have to play with — darktable supports five.
# Map cluster ids modulo 5 onto them so any k from KMeans gets a colour.
CLUSTER_COLOURS = ["red", "yellow", "green", "blue", "purple"]


def stars_from_rank(rank: int, total: int) -> int:
    """Top 10% → 5 stars, next 20% → 4, next 30% → 3, next 30% → 2, last 10% → 1."""
    if total <= 0 or rank < 0 or rank >= total:
        return 0
    pct = rank / total  # 0 = best
    if pct < 0.10:
        return 5
    if pct < 0.30:
        return 4
    if pct < 0.60:
        return 3
    if pct < 0.90:
        return 2
    return 1


def _xmp_for(rating: int, label: str | None, subjects: list[str]) -> str:
    """Render the smallest darktable-compatible XMP I can get away with."""
    label_attr = f' xmp:Label="{sx.escape(label)}"' if label else ""
    if subjects:
        items = "\n".join(f"      <rdf:li>{sx.escape(s)}</rdf:li>" for s in subjects)
        subject_block = (
            "    <dc:subject>\n"
            "     <rdf:Bag>\n"
            f"{items}\n"
            "     </rdf:Bag>\n"
            "    </dc:subject>\n"
        )
    else:
        subject_block = ""
    return (
        '<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>\n'
        '<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="banger">\n'
        ' <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">\n'
        '  <rdf:Description rdf:about=""\n'
        '    xmlns:xmp="http://ns.adobe.com/xap/1.0/"\n'
        '    xmlns:dc="http://purl.org/dc/elements/1.1/"\n'
        f'    xmp:Rating="{rating}"{label_attr}>\n'
        f"{subject_block}"
        "  </rdf:Description>\n"
        " </rdf:RDF>\n"
        "</x:xmpmeta>\n"
        '<?xpacket end="w"?>\n'
    )


def _sidecar_path(src: Path) -> Path:
    """darktable expects FILE.JPG.xmp (preserve original extension + .xmp)."""
    return src.with_suffix(src.suffix + ".xmp")


def write_sidecar(
    row: "Row",
    rank: int,
    total: int,
    cluster_id: int | None = None,
) -> Path | None:
    """Write the XMP next to the source frame. Returns the path written, or None."""
    from banger import metrics as metrics_mod

    src = row.frame.classify_path
    rating = stars_from_rank(rank, total)
    if row.aesthetic is None or row.sharpness < 0:
        rating = 0

    label: str | None = None
    if cluster_id is not None:
        label = CLUSTER_COLOURS[cluster_id % len(CLUSTER_COLOURS)]
    elif row.scene_preset:
        # No cluster info (legacy prompt-based scenes path). Derive a stable
        # colour from a hash of the preset name so the same preset always
        # gets the same colour across runs.
        label = CLUSTER_COLOURS[hash(row.scene_preset) % len(CLUSTER_COLOURS)]

    subjects: list[str] = []
    if row.aesthetic_source:
        subjects.append(f"banger/source/{row.aesthetic_source}")
    if row.scene_preset:
        subjects.append(f"banger/scene/{row.scene_preset}")
    subjects.append(f"banger/rank/{rank + 1:03d}")
    if row.metrics:
        combined = metrics_mod.combined_score(row.metrics, row.aesthetic)
        subjects.append(f"banger/quality/{combined:.1f}")

    payload = _xmp_for(rating, label, subjects)
    out = _sidecar_path(src)
    try:
        out.write_text(payload, encoding="utf-8")
    except OSError as e:
        log.warning("xmp write failed for %s: %s", src, e)
        return None
    return out


def write_for_rows(
    rows: list["Row"],
    cluster_ids: dict[str, int] | None = None,
) -> int:
    """Rank `rows` by their effective score (taste head or sharpness fallback)
    and emit an XMP per row. Returns count written.
    """
    # Sort descending so rank 0 = best.
    def _score(r):
        if r.aesthetic is not None:
            return r.aesthetic
        # Frames that failed a gate landed with aesthetic=None; use a sentinel
        # so they fall to the bottom but still get rating=0.
        return -1e9

    ordered = sorted(rows, key=_score, reverse=True)
    total = len(ordered)
    written = 0
    for rank, r in enumerate(ordered):
        cid = None
        if cluster_ids is not None:
            cid = cluster_ids.get(r.frame.display_name)
        if write_sidecar(r, rank, total, cluster_id=cid) is not None:
            written += 1
    return written
