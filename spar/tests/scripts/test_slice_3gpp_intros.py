import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))


def test_parse_spec_number_standard():
    from slice_3gpp_intros import parse_spec_number
    assert parse_spec_number("29502-i40.md") == "29.502"
    assert parse_spec_number("38300-i30.md") == "38.300"
    assert parse_spec_number("23501-i20.md") == "23.501"


def test_parse_spec_number_unknown():
    from slice_3gpp_intros import parse_spec_number
    assert parse_spec_number("10.2 TT.md") == ""
    assert parse_spec_number("foo.md") == ""
    assert parse_spec_number("README.md") == ""


def test_slice_writes_limited_lines(tmp_path):
    from slice_3gpp_intros import slice_file
    src = tmp_path / "29502-i40.md"
    src.write_text("\n".join(f"line {i}" for i in range(2000)), encoding="utf-8")
    dst = tmp_path / "out.md"
    slice_file(src, dst, line_limit=1000)
    result_lines = dst.read_text(encoding="utf-8").splitlines()
    assert len(result_lines) == 1000
    assert result_lines[0] == "line 0"
    assert result_lines[999] == "line 999"


def test_slice_short_file(tmp_path):
    from slice_3gpp_intros import slice_file
    src = tmp_path / "short.md"
    src.write_text("hello\nworld\n", encoding="utf-8")
    dst = tmp_path / "out.md"
    slice_file(src, dst, line_limit=1000)
    assert dst.read_text(encoding="utf-8") == "hello\nworld\n"
