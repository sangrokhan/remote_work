"""bundle.py -- zip everything a run produced into one download, the way
``tcp_congestion/tcp_congestion/app.py``'s ``download_bundle_zip`` route did
(``zipfile.ZIP_DEFLATED``, in-memory ``io.BytesIO``, one entry per CSV plus
the raw pcap).

DESIGN.md 4.6: "세 CSV + pcap을 기존 bundle.zip 방식으로 묶어서 다운로드하는
구조는 유지" -- this module is that structure, generalized so any caller
(a web route, a CLI, a test) can build the same zip without duplicating the
``zipfile`` bookkeeping. It does not know about runs, backends, or HTTP: it
takes already-rendered CSV text plus optional pcap paths and returns bytes.
"""

from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

# Mirrors tcp_congestion/tcp_congestion/app.py's `_SAFE_SLUG`: a label used
# to build a zip entry/file name must not carry anything a filesystem or a
# zip reader would choke on, and must never be substituted silently for a
# different one (aipt/core/capture.py's `_SAFE_LABEL` makes the same call
# for pcap filenames, for the same reason -- see its module docstring).
_SAFE_SLUG = re.compile(r"[^a-z0-9_-]+")


def slugify(label: str, default: str = "run") -> str:
    """A label, made safe for a zip entry name or filename.

    Never guesses a substitute for an unspellable label beyond the default:
    an empty result (nothing alphanumeric survived) falls back to
    ``default`` rather than an empty filename, which is the one case a zip
    writer cannot represent at all.
    """
    slug = _SAFE_SLUG.sub("-", (label or "").lower()).strip("-")
    return slug or default


def build_bundle_zip(
    *,
    label: str = "run",
    connection_csv: str | None = None,
    turns_csv: str | None = None,
    packets_csv: str | None = None,
    pcap_paths: list[str | Path] | None = None,
    extra_files: dict[str, str | bytes] | None = None,
) -> bytes:
    """One zip holding whichever of the three export-layer CSVs the caller
    has, plus any pcap files, plus arbitrary extras (e.g. the run's JSON).

    Every argument is optional and independently omittable -- a caller
    without congestion monitoring on, say, passes ``connection_csv=None``
    and gets a bundle without a ``cwnd.csv`` entry rather than an empty one
    with a header and no rows claiming to have looked. ``extrasaction`` at
    the CSV layer already handles "measured nothing"; this layer handles
    "never asked".

    Entry names are prefixed with the slugified ``label`` (matching
    ``tcp_congestion``'s ``{slug}_cwnd.csv`` / ``{slug}_turns.csv``
    convention) so a bundle downloaded for one arm/run never collides with
    another sitting in the same downloads folder.
    """
    slug = slugify(label)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        if connection_csv is not None:
            zf.writestr(f"{slug}_cwnd.csv", connection_csv)
        if turns_csv is not None:
            zf.writestr(f"{slug}_turns.csv", turns_csv)
        if packets_csv is not None:
            zf.writestr(f"{slug}_packets.csv", packets_csv)

        for pcap_path in pcap_paths or []:
            path = Path(pcap_path)
            if path.exists():
                zf.write(path, arcname=path.name)
            # A missing pcap (capture unavailable, or a mock run that made
            # no real traffic -- docs/outputs.md's note on mock captures) is
            # not an error at this layer: the bundle simply has one fewer
            # file, same as an omitted CSV above.

        for name, content in (extra_files or {}).items():
            zf.writestr(name, content)

    return buf.getvalue()


def bundle_zip_name(label: str) -> str:
    """The download filename for a bundle -- ``aipt_{slug}_bundle.zip``,
    parallel to ``tcp_congestion``'s ``tcp_congestion_{algo}_{label}.zip``.
    """
    return f"aipt_{slugify(label)}_bundle.zip"
