"""End-to-end smoke test for `cmd_run` with mocked CLIP + scene classification.

Verifies the orchestration around sharpness, dedup, scenes, and output writing
without loading the real CLIP model. The mocks return constant embeddings /
breakdowns so the test focuses on flow control, not learned behaviour.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from banger import aesthetic, scenes


@pytest.fixture
def cli_env(tmp_path: Path, monkeypatch, isolated_state, make_jpeg):
    """Build a tmp input dir with mixed sharpness frames; mock CLIP-touching calls."""
    make_jpeg(name="A.JPG", sharpness="high")
    make_jpeg(name="B.JPG", sharpness="medium")
    make_jpeg(name="C.JPG", sharpness="low")
    make_jpeg(name="D.JPG", sharpness="blank")
    make_jpeg(name="E.JPG", subdir="sub", sharpness="high")

    # Use the first pixel of the preview to seed a unique embedding per frame —
    # k-means selection in cmd_run needs distinct embeddings to cluster.
    def _stub_encode(preview):
        seed = int(preview[0, 0, 0]) * 17 + int(preview[0, 0, 1]) * 31
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(8).astype(np.float32)
        v /= np.linalg.norm(v)
        return v

    monkeypatch.setattr(aesthetic, "encode_image", _stub_encode)
    monkeypatch.setattr(
        aesthetic,
        "score_from_embedding",
        lambda emb: (1.5, {p: 0.2 for p in (aesthetic.POSITIVE_PROMPTS + aesthetic.NEGATIVE_PROMPTS)}),
    )

    fake_match = scenes.SceneMatch(
        prompt=scenes.DEFAULT_PROMPT,
        preset=scenes.DEFAULT_PRESET,
        score=0.25,
        top_score=0.25,
        breakdown={p: 0.2 for p in scenes.SCENE_PROMPTS},
        fell_back=False,
    )
    monkeypatch.setattr(scenes, "classify", lambda emb: fake_match)

    return tmp_path


def test_cmd_run_writes_report_and_output(cli_env, tmp_path):
    from banger.cli import cmd_run

    input_dir = cli_env
    report_path = tmp_path / "report.html"
    output_dir = tmp_path / "output"

    rc = cmd_run(
        input_dir=input_dir,
        recursive=True,
        report_path=report_path,
        output_dir=output_dir,
        top_n=2,
    )
    assert rc == 0
    assert report_path.exists()
    html = report_path.read_text(encoding="utf-8")
    assert "KEEP" in html
    assert "REJECT" in html

    # Output should have top-2 plus manifest.
    assert (output_dir / "manifest.json").exists()
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert len(manifest) == 2
    assert manifest[0]["rank"] == 1
    # Culler-only mode: originals copied verbatim, listed in "files".
    for entry in manifest:
        assert entry["files"]
        for name in entry["files"]:
            assert (output_dir / name).exists()


def test_cmd_run_no_report_uses_cache_path(cli_env, isolated_state):
    """Two consecutive runs without --report should hit the metadata cache."""
    from banger.cli import cmd_run

    rc1 = cmd_run(input_dir=cli_env, recursive=True, report_path=None)
    assert rc1 == 0
    rc2 = cmd_run(input_dir=cli_env, recursive=True, report_path=None)
    assert rc2 == 0

    # After two runs, every survivor's metadata is on disk.
    md_files = list(isolated_state.METADATA_DIR.glob("*.json"))
    assert len(md_files) >= 1


def test_cmd_run_face_gate_does_not_crash(cli_env):
    from banger.cli import cmd_run

    rc = cmd_run(input_dir=cli_env, recursive=True, report_path=None, face_gate=True)
    assert rc == 0


def test_cmd_run_returns_2_on_missing_dir(tmp_path):
    from banger.cli import cmd_run

    missing = tmp_path / "does_not_exist"
    rc = cmd_run(input_dir=missing, recursive=False, report_path=None)
    assert rc == 2
