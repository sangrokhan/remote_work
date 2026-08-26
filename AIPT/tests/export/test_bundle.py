"""aipt.export.bundle -- zip everything a run produced (DESIGN.md 4.6: "세
CSV + pcap을 기존 bundle.zip 방식으로 묶어서 다운로드하는 구조는 유지").
"""

from __future__ import annotations

import io
import zipfile

from aipt.export.bundle import build_bundle_zip, bundle_zip_name, slugify


def test_slugify_lowercases_and_strips_unsafe_chars():
    assert slugify("Public AI: Gemini/cached!") == "public-ai-gemini-cached"


def test_slugify_empty_falls_back_to_default():
    assert slugify("", default="run") == "run"
    assert slugify("!!!", default="run") == "run"


def test_bundle_zip_name_uses_slug():
    assert bundle_zip_name("mock:baseline") == "aipt_mock-baseline_bundle.zip"


def test_build_bundle_zip_contains_all_three_csvs():
    data = build_bundle_zip(
        label="mock:baseline",
        connection_csv="label,host\n",
        turns_csv="backend,arm\n",
        packets_csv="index,ts\n",
    )
    zf = zipfile.ZipFile(io.BytesIO(data))
    names = set(zf.namelist())
    assert names == {
        "mock-baseline_cwnd.csv",
        "mock-baseline_turns.csv",
        "mock-baseline_packets.csv",
    }
    assert zf.read("mock-baseline_turns.csv").decode() == "backend,arm\n"


def test_build_bundle_zip_omits_missing_csvs():
    data = build_bundle_zip(label="x", turns_csv="backend,arm\n")
    zf = zipfile.ZipFile(io.BytesIO(data))
    assert zf.namelist() == ["x_turns.csv"]


def test_build_bundle_zip_includes_pcap_file(tmp_path):
    pcap = tmp_path / "capture.pcap"
    pcap.write_bytes(b"\xd4\xc3\xb2\xa1" + b"\x00" * 20)  # minimal fake header
    data = build_bundle_zip(label="x", turns_csv="a,b\n", pcap_paths=[pcap])
    zf = zipfile.ZipFile(io.BytesIO(data))
    assert "capture.pcap" in zf.namelist()


def test_build_bundle_zip_skips_missing_pcap_silently(tmp_path):
    missing = tmp_path / "gone.pcap"
    data = build_bundle_zip(label="x", turns_csv="a,b\n", pcap_paths=[missing])
    zf = zipfile.ZipFile(io.BytesIO(data))
    assert zf.namelist() == ["x_turns.csv"]


def test_build_bundle_zip_extra_files():
    data = build_bundle_zip(label="x", extra_files={"run.json": '{"ok": true}'})
    zf = zipfile.ZipFile(io.BytesIO(data))
    assert zf.read("run.json").decode() == '{"ok": true}'


def test_build_bundle_zip_empty_call_still_valid_zip():
    data = build_bundle_zip()
    zf = zipfile.ZipFile(io.BytesIO(data))
    assert zf.namelist() == []
    assert zf.testzip() is None
