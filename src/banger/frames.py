from dataclasses import dataclass
from pathlib import Path

from banger.preview import JPEG_SUFFIXES, RAW_SUFFIXES


@dataclass(frozen=True)
class Frame:
    """One photo, possibly present as both a RAW and a sibling JPEG."""

    stem: str
    jpeg: Path | None
    raw: Path | None

    @property
    def classify_path(self) -> Path:
        # Prefer the camera-written JPEG (faster to read, higher quality than the in-RAW thumb);
        # fall back to the RAW so RAW-only shoots still score.
        path = self.jpeg if self.jpeg is not None else self.raw
        assert path is not None
        return path

    @property
    def develop_path(self) -> Path:
        # darktable-cli wants the RAW for headroom; only fall back to JPEG if there is no RAW.
        path = self.raw if self.raw is not None else self.jpeg
        assert path is not None
        return path

    @property
    def kind(self) -> str:
        if self.jpeg and self.raw:
            return "raw+jpeg"
        return "jpeg" if self.jpeg else "raw"


def discover_frames(input_dir: Path) -> list[Frame]:
    by_stem: dict[str, dict[str, Path]] = {}
    for p in input_dir.iterdir():
        if not p.is_file():
            continue
        if p.suffix in JPEG_SUFFIXES:
            by_stem.setdefault(p.stem, {})["jpeg"] = p
        elif p.suffix in RAW_SUFFIXES:
            by_stem.setdefault(p.stem, {})["raw"] = p
    frames = [
        Frame(stem=stem, jpeg=files.get("jpeg"), raw=files.get("raw"))
        for stem, files in by_stem.items()
    ]
    return sorted(frames, key=lambda f: f.stem)
