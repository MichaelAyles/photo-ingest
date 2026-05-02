from dataclasses import dataclass
from pathlib import Path

from banger.preview import JPEG_SUFFIXES, RAW_SUFFIXES


@dataclass(frozen=True)
class Frame:
    """One photo, possibly present as both a RAW and a sibling JPEG."""

    stem: str
    subdir: str  # path relative to the discovery root, "" if at root
    jpeg: Path | None
    raw: Path | None

    @property
    def classify_path(self) -> Path:
        path = self.jpeg if self.jpeg is not None else self.raw
        assert path is not None
        return path

    @property
    def develop_path(self) -> Path:
        path = self.raw if self.raw is not None else self.jpeg
        assert path is not None
        return path

    @property
    def kind(self) -> str:
        if self.jpeg and self.raw:
            return "raw+jpeg"
        return "jpeg" if self.jpeg else "raw"

    @property
    def display_name(self) -> str:
        return f"{self.subdir}/{self.stem}" if self.subdir else self.stem


def discover_frames(input_dir: Path, recursive: bool = False) -> list[Frame]:
    by_key: dict[tuple[str, str], dict[str, Path]] = {}
    iterator = input_dir.rglob("*") if recursive else input_dir.iterdir()
    for p in iterator:
        if not p.is_file():
            continue
        if p.suffix in JPEG_SUFFIXES:
            kind = "jpeg"
        elif p.suffix in RAW_SUFFIXES:
            kind = "raw"
        else:
            continue
        rel_parent = p.parent.relative_to(input_dir)
        subdir = "" if rel_parent == Path(".") else str(rel_parent).replace("\\", "/")
        key = (subdir, p.stem)
        by_key.setdefault(key, {})[kind] = p
    frames = [
        Frame(stem=stem, subdir=subdir, jpeg=files.get("jpeg"), raw=files.get("raw"))
        for (subdir, stem), files in by_key.items()
    ]
    return sorted(frames, key=lambda f: (f.subdir, f.stem))
