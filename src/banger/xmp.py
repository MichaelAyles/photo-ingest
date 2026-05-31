"""Hand-rolled darktable XMP sidecar writer.

No new dep. The XMP format is well-documented and small: it's RDF-XML
with a handful of attributes we care about (rating, label, hierarchical
subjects). darktable picks these up on its next library scan, Lightroom
and other DAMs read them too.

What we write:
- xmp:Rating       0-5 stars. Two sources, in priority order:
                   (1) the user's own taste label for the frame, if one
                   exists (a -5..+5 score mapped onto 0-5 stars), so a
                   roundtrip reflects the user's judgement, not just the
                   batch; (2) otherwise the combined quality score's rank
                   in the input batch (top 10% = 5, next 20% = 4, next
                   30% = 3, next 30% = 2, bottom 10% = 1). Frames that
                   failed any gate get 0.
- xmp:Label        a colour label string. By default this encodes the
                   scene cluster ("red"/"yellow"/"green"/"blue"/"purple"),
                   but callers can switch the source to the pick state so
                   the colour reflects Pick/Reject instead.
- dc:subject       a keyword Bag. Always carries the flat banger/* tags
                   (banger/source/<aesthetic_source>, banger/scene/<preset>,
                   banger/rank/<N>, banger/quality/<score>) plus, when the
                   caller supplies them, the frame's face NAMES and CLIP
                   TAGS and a Pick/Reject keyword. Lets the user filter in
                   any DAM.
- lr:hierarchicalSubject   the same names/tags as a Lightroom-style
                   hierarchy (People|Beth, Scene|<preset>, Tag|<tag>) so
                   Lightroom and Capture One build a proper keyword tree.

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


# Eight colour labels we have to play with, darktable supports five.
# Map cluster ids modulo 5 onto them so any k from KMeans gets a colour.
CLUSTER_COLOURS = ["red", "yellow", "green", "blue", "purple"]

# The taste-label score scale used by banger.state (kept local so this
# module never imports state — avoids a cycle and lets callers pass values
# straight through). Mirror of state.SCORE_MIN / state.SCORE_MAX.
LABEL_MIN = -5
LABEL_MAX = 5

# Colour-label source selectors for the writer's `color_label_source`
# param. "scene" = the historical scene-cluster colour (default);
# "pick" = colour driven by the Pick/Reject state.
COLOR_FROM_SCENE = "scene"
COLOR_FROM_PICK = "pick"

# Colours used when the label encodes pick state instead of scene cluster.
PICK_COLOURS = {"pick": "green", "reject": "red"}


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


def stars_from_label(score: int) -> int:
    """Map a user taste label (-5..+5) onto 0-5 stars.

    The taste scale is symmetric around 0 (neutral); stars are 0-5 with no
    negatives, so we shift and rescale linearly: -5 → 0, 0 → ~3, +5 → 5.
    Out-of-range scores clamp to the valid star band.
    """
    s = max(LABEL_MIN, min(LABEL_MAX, int(score)))
    span = LABEL_MAX - LABEL_MIN  # 10
    stars = round((s - LABEL_MIN) / span * 5)
    return max(0, min(5, stars))


def _hier_keywords(face_names, tags, scene_preset):
    """Build (flat, hierarchical) keyword lists from names/tags/scene.

    Flat entries are the bare names/tags (DAM-portable in dc:subject);
    hierarchical entries use Lightroom's pipe syntax (People|Beth) so a
    proper keyword tree is reconstructed on import.
    """
    flat: list[str] = []
    hier: list[str] = []
    seen: set[str] = set()

    def _add(flat_kw, hier_kw):
        if flat_kw and flat_kw not in seen:
            seen.add(flat_kw)
            flat.append(flat_kw)
            hier.append(hier_kw)

    for name in face_names or []:
        name = str(name).strip()
        if name:
            _add(name, f"People|{name}")
    if scene_preset:
        _add(str(scene_preset), f"Scene|{scene_preset}")
    for tag in tags or []:
        tag = str(tag).strip()
        if tag:
            _add(tag, f"Tag|{tag}")
    return flat, hier


def _bag(tag_open, tag_close, items):
    """Render an rdf:Bag block for a dc/lr property, or "" when empty."""
    if not items:
        return ""
    lis = "\n".join(f"      <rdf:li>{sx.escape(s)}</rdf:li>" for s in items)
    return (
        f"    {tag_open}\n"
        "     <rdf:Bag>\n"
        f"{lis}\n"
        "     </rdf:Bag>\n"
        f"    {tag_close}\n"
    )


def _xmp_for(
    rating: int,
    label: str | None,
    subjects: list[str],
    hierarchical: list[str] | None = None,
) -> str:
    """Render the smallest darktable-compatible XMP I can get away with.

    `subjects` go into dc:subject (flat, DAM-portable keywords);
    `hierarchical`, when given, additionally goes into lr:hierarchicalSubject
    so Lightroom/Capture One rebuild a keyword tree.
    """
    label_attr = f' xmp:Label="{sx.escape(label)}"' if label else ""
    subject_block = _bag("<dc:subject>", "</dc:subject>", subjects)
    hier_block = _bag(
        "<lr:hierarchicalSubject>", "</lr:hierarchicalSubject>", hierarchical or []
    )
    lr_ns = (
        '    xmlns:lr="http://ns.adobe.com/lightroom/1.0/"\n' if hier_block else ""
    )
    return (
        '<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>\n'
        '<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="banger">\n'
        ' <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">\n'
        '  <rdf:Description rdf:about=""\n'
        '    xmlns:xmp="http://ns.adobe.com/xap/1.0/"\n'
        '    xmlns:dc="http://purl.org/dc/elements/1.1/"\n'
        f"{lr_ns}"
        f'    xmp:Rating="{rating}"{label_attr}>\n'
        f"{subject_block}"
        f"{hier_block}"
        "  </rdf:Description>\n"
        " </rdf:RDF>\n"
        "</x:xmpmeta>\n"
        '<?xpacket end="w"?>\n'
    )


def _sidecar_path(src: Path) -> Path:
    """darktable expects FILE.JPG.xmp (preserve original extension + .xmp)."""
    return src.with_suffix(src.suffix + ".xmp")


def write_sidecar(
    row: Row,
    rank: int,
    total: int,
    cluster_id: int | None = None,
    taste_label: int | None = None,
    pick_state: str | None = None,
    face_names: list[str] | None = None,
    tags: list[str] | None = None,
    color_label_source: str = COLOR_FROM_SCENE,
) -> Path | None:
    """Write the XMP next to the source frame. Returns the path written, or None.

    Optional, all additive and backward-compatible:
    - taste_label: the user's own -5..+5 judgement for this frame. When
      present (and the frame didn't fail a gate) the rating comes from the
      user's label instead of the batch rank. Pass None to fall back to rank.
    - pick_state: "pick" or "reject" (case-insensitive). Writes a portable
      Pick/Reject keyword and, when color_label_source is "pick", drives the
      xmp:Label colour too.
    - face_names / tags: identity names and CLIP tags to emit as keywords
      (both flat dc:subject and hierarchical lr:hierarchicalSubject).
    - color_label_source: "scene" (default, historical scene-cluster colour)
      or "pick" (colour from the Pick/Reject state).
    """
    from banger import metrics as metrics_mod

    src = row.frame.classify_path

    gate_failed = row.aesthetic is None or row.sharpness < 0
    if gate_failed:
        rating = 0
    elif taste_label is not None:
        # The user told us what they think of this frame — honour it over
        # the batch rank (the whole point of the local-first taste loop).
        rating = stars_from_label(taste_label)
    else:
        rating = stars_from_rank(rank, total)

    pick = pick_state.lower() if pick_state else None
    if pick not in (None, "pick", "reject"):
        log.warning("xmp: ignoring unknown pick_state %r", pick_state)
        pick = None

    label: str | None = None
    if color_label_source == COLOR_FROM_PICK:
        if pick is not None:
            label = PICK_COLOURS.get(pick)
    else:  # COLOR_FROM_SCENE (default / anything else)
        if cluster_id is not None:
            label = CLUSTER_COLOURS[cluster_id % len(CLUSTER_COLOURS)]
        elif row.scene_preset:
            # No cluster info (legacy prompt-based scenes path). Derive a
            # stable colour from a hash of the preset name so the same preset
            # always gets the same colour across runs.
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
    if pick == "pick":
        subjects.append("banger/pick")
    elif pick == "reject":
        subjects.append("banger/reject")

    # Human-facing keywords (names + scene + CLIP tags) on top of the
    # machine banger/* tags, in both flat and hierarchical form.
    extra_flat, extra_hier = _hier_keywords(face_names, tags, row.scene_preset)
    subjects.extend(extra_flat)

    payload = _xmp_for(rating, label, subjects, hierarchical=extra_hier)
    out = _sidecar_path(src)
    try:
        out.write_text(payload, encoding="utf-8")
    except OSError as e:
        log.warning("xmp write failed for %s: %s", src, e)
        return None
    return out


def write_for_rows(
    rows: list[Row],
    cluster_ids: dict[str, int] | None = None,
    taste_labels: dict[str, int] | None = None,
    pick_states: dict[str, str] | None = None,
    face_names: dict[str, list[str]] | None = None,
    tags: dict[str, list[str]] | None = None,
    color_label_source: str = COLOR_FROM_SCENE,
) -> int:
    """Rank `rows` by their effective score (taste head or sharpness fallback)
    and emit an XMP per row. Returns count written.

    The optional per-frame maps are keyed by frame.display_name (same key as
    `cluster_ids`) and are all additive / backward-compatible:
    - taste_labels: display_name -> user -5..+5 score (rating source override)
    - pick_states:  display_name -> "pick" | "reject"
    - face_names:   display_name -> list of identity names (keywords)
    - tags:         display_name -> list of CLIP tags (keywords)
    - color_label_source: passed through to write_sidecar.
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
        key = r.frame.display_name
        cid = cluster_ids.get(key) if cluster_ids is not None else None
        label = taste_labels.get(key) if taste_labels is not None else None
        pick = pick_states.get(key) if pick_states is not None else None
        names = face_names.get(key) if face_names is not None else None
        frame_tags = tags.get(key) if tags is not None else None
        if write_sidecar(
            r,
            rank,
            total,
            cluster_id=cid,
            taste_label=label,
            pick_state=pick,
            face_names=names,
            tags=frame_tags,
            color_label_source=color_label_source,
        ) is not None:
            written += 1
    return written
