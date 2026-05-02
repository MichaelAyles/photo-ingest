"""Final develop stage: darktable-cli when available, copy-with-manifest fallback.

`find_darktable()` returns the resolved binary path (PATH first, then standard
Windows install dirs) or None. `develop_to_jpeg(...)` is the full per-frame
operation: run darktable-cli with the matched preset's .xmp, or fall back to
copying the source JPEG. The fallback always produces an output file so the
pipeline never silently drops a kept frame.

CLAUDE.md spec: "If darktable-cli fails, fall back to the embedded JPEG with
the preset's tone curve approximated via PIL." For v0 we just copy the JPEG;
PIL tone-curve approximation can come later (v0.5 / v1).
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("banger")

DARKTABLE_BINARY_NAMES = ("darktable-cli", "darktable-cli.exe")
DARKTABLE_WINDOWS_FALLBACK_PATHS = (
    Path("C:/Program Files/darktable/bin/darktable-cli.exe"),
    Path("C:/Program Files (x86)/darktable/bin/darktable-cli.exe"),
)


@dataclass
class DevelopResult:
    success: bool
    used: str  # "darktable" | "copy_fallback" | "skipped"
    note: str = ""


def find_darktable() -> Path | None:
    """Locate darktable-cli; return None if absent so the caller can fall back."""
    for name in DARKTABLE_BINARY_NAMES:
        resolved = shutil.which(name)
        if resolved:
            return Path(resolved)
    for p in DARKTABLE_WINDOWS_FALLBACK_PATHS:
        if p.exists():
            return p
    return None


def run_darktable(
    darktable_cli: Path,
    src: Path,
    dst: Path,
    style_xmp: Path | None = None,
    timeout_sec: int = 120,
) -> DevelopResult:
    """Invoke darktable-cli once. Caller is responsible for picking the preset."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd: list[str] = [str(darktable_cli)]
    if style_xmp is not None:
        cmd.append(str(style_xmp))
    cmd.append(str(src))
    cmd.append(str(dst))
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)
    except FileNotFoundError as e:
        return DevelopResult(False, "skipped", f"darktable-cli not executable: {e}")
    except subprocess.TimeoutExpired:
        return DevelopResult(False, "skipped", f"timeout after {timeout_sec}s")
    if result.returncode != 0 or not dst.exists():
        return DevelopResult(False, "skipped", result.stderr.strip() or f"exit {result.returncode}")
    return DevelopResult(True, "darktable")


def copy_fallback(src: Path, dst: Path) -> DevelopResult:
    """Verbatim copy when darktable isn't available (or fails)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return DevelopResult(True, "copy_fallback", "darktable not available")


def develop_to_jpeg(
    src: Path,
    dst: Path,
    preset_name: str | None,
    presets_dir: Path,
    darktable_cli: Path | None = None,
) -> DevelopResult:
    """Develop one frame. Resolve preset → XMP path → darktable, else copy."""
    style: Path | None = None
    if preset_name and presets_dir.is_dir():
        candidate = presets_dir / f"{preset_name}.xmp"
        if candidate.exists():
            style = candidate
        else:
            log.info("preset %s.xmp not in %s — copy fallback", preset_name, presets_dir)
    if darktable_cli is None:
        return copy_fallback(src, dst)
    if style is None:
        # darktable can develop without a style, applying its defaults; but
        # without our intentional style the output is generic. Prefer to copy
        # the camera JPEG so the user gets predictable v0 output.
        if src.suffix.lower() in (".jpg", ".jpeg"):
            return copy_fallback(src, dst)
        # ARW with no preset: let darktable produce a default-rendered JPEG.
    result = run_darktable(darktable_cli, src, dst, style_xmp=style)
    if not result.success and src.suffix.lower() in (".jpg", ".jpeg"):
        log.warning("darktable failed for %s: %s — copy fallback", src.name, result.note)
        return copy_fallback(src, dst)
    return result


def write_manifest(out_path: Path, entries: list[dict]) -> None:
    out_path.write_text(json.dumps(entries, indent=2, default=str), encoding="utf-8")
