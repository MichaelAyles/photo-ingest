"""Copy every photo from test_photos/camera and test_photos/camera2 that does
NOT appear to contain a human into ./test-photos/.

Uses banger's own CLIP tagger + insightface face detector (belt + suspenders).
Embeddings are cached on disk, so re-runs are fast.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

from banger import aesthetic, face_id, state
from banger.frames import discover_frames
from banger.preview import load_preview
from banger.tags import tag_from_embedding

SRC_DIRS = [Path("test_photos/camera"), Path("test_photos/camera2")]
DST = Path("test-photos")

HUMAN_TAGS = {
    "person", "people", "group of people",
    "child", "baby", "family",
    "man", "woman", "couple", "selfie",
    "portrait shot",
}
HUMAN_TAG_THRESHOLD = 0.22


def is_human(preview, sha):
    emb = state.load_embedding(sha)
    if emb is None:
        emb = aesthetic.encode_image(preview)
        state.cache_embedding(sha, emb)
    tags = tag_from_embedding(emb, top_n=20, min_sim=0.0)
    top_human = max(
        ((t, s) for t, s in tags if t in HUMAN_TAGS),
        key=lambda kv: kv[1],
        default=(None, 0.0),
    )
    if top_human[1] >= HUMAN_TAG_THRESHOLD:
        return True, f"tag={top_human[0]}@{top_human[1]:.2f}"
    try:
        dets = face_id.extract_face_detections(preview)
        if dets:
            return True, f"faces={len(dets)}"
    except Exception as e:
        print(f"  face_id err: {e}", file=sys.stderr)
    return False, f"top_human={top_human[0]}@{top_human[1]:.2f}"


def main():
    DST.mkdir(exist_ok=True)
    kept = 0
    rejected = 0
    failed = 0
    for src_dir in SRC_DIRS:
        if not src_dir.is_dir():
            print(f"skip missing: {src_dir}")
            continue
        frames = discover_frames(src_dir, recursive=True)
        print(f"\n{src_dir}: {len(frames)} frames")
        for i, f in enumerate(frames, 1):
            try:
                preview = load_preview(f.classify_path)
            except Exception as e:
                print(f"  [{i}/{len(frames)}] {f.display_name} preview FAIL: {e}")
                failed += 1
                continue
            sha = state.sha256_of(f.classify_path)
            human, reason = is_human(preview, sha)
            tag = "SKIP" if human else "KEEP"
            print(f"  [{i}/{len(frames)}] {tag} {f.display_name} ({reason})")
            if human:
                rejected += 1
                continue
            for src in (f.jpeg, f.raw):
                if src is None:
                    continue
                dst = DST / src.name
                if dst.exists():
                    continue
                shutil.copy2(src, dst)
            kept += 1
    print(f"\nkept={kept} rejected={rejected} failed={failed} -> {DST.resolve()}")


if __name__ == "__main__":
    main()
