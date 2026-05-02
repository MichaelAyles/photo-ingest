"""Frame discovery: pairing logic and recursive walk."""

from banger.frames import Frame, discover_frames


def test_solo_jpeg_is_a_frame(make_jpeg, tmp_path):
    make_jpeg(name="DSC00001.JPG")
    frames = discover_frames(tmp_path)
    assert len(frames) == 1
    f = frames[0]
    assert f.stem == "DSC00001"
    assert f.subdir == ""
    assert f.kind == "jpeg"
    assert f.classify_path == f.develop_path == f.jpeg


def test_raw_jpeg_pair_groups_by_stem(make_jpeg, tmp_path):
    make_jpeg(name="DSC00002.JPG")
    # Synthesise a placeholder ARW alongside (content doesn't matter for pairing).
    (tmp_path / "DSC00002.ARW").write_bytes(b"not a real raw")
    frames = discover_frames(tmp_path)
    assert len(frames) == 1
    f = frames[0]
    assert f.kind == "raw+jpeg"
    assert f.classify_path == f.jpeg  # classify prefers JPEG
    assert f.develop_path == f.raw  # develop prefers RAW


def test_recursive_walk_includes_subdirs(make_jpeg, tmp_path):
    make_jpeg(name="A.JPG", subdir="trip1")
    make_jpeg(name="B.JPG", subdir="trip2")
    make_jpeg(name="C.JPG")

    flat = discover_frames(tmp_path, recursive=False)
    assert {f.stem for f in flat} == {"C"}

    deep = discover_frames(tmp_path, recursive=True)
    assert {f.stem for f in deep} == {"A", "B", "C"}
    by_stem = {f.stem: f for f in deep}
    assert by_stem["A"].subdir == "trip1"
    assert by_stem["B"].subdir == "trip2"
    assert by_stem["C"].subdir == ""


def test_same_stem_in_two_subdirs_does_not_collide(make_jpeg, tmp_path):
    make_jpeg(name="DSC00001.JPG", subdir="trip1")
    make_jpeg(name="DSC00001.JPG", subdir="trip2")
    frames = discover_frames(tmp_path, recursive=True)
    assert len(frames) == 2
    assert {(f.subdir, f.stem) for f in frames} == {("trip1", "DSC00001"), ("trip2", "DSC00001")}


def test_unsupported_extensions_ignored(make_jpeg, tmp_path):
    make_jpeg(name="DSC00001.JPG")
    (tmp_path / "notes.txt").write_text("ignore me")
    (tmp_path / "video.mp4").write_bytes(b"binary")
    frames = discover_frames(tmp_path)
    assert len(frames) == 1


def test_display_name_falls_back_to_stem_at_root():
    f_root = Frame(stem="DSC00001", subdir="", jpeg=None, raw=None)
    f_sub = Frame(stem="DSC00001", subdir="trip1", jpeg=None, raw=None)
    assert f_root.display_name == "DSC00001"
    assert f_sub.display_name == "trip1/DSC00001"
