"""The suite must not write into the operator's run history.

store.py resolves GEMINI_DATA_DIR once, at import, and the route tests drive real
runs through the Flask app -- so without this every `pytest` left a mock execution
behind in data/runs, next to the live runs the experiment is actually about. They
accumulated: 122 files, 17 MB, most of them synthetic, all of them indistinguishable
from real ones in the history panel.

The env is set here, at collection time, because that is the only point that is
guaranteed to run before a test module imports store.
"""

import os
import tempfile

_TMP = tempfile.mkdtemp(prefix="gemini_token_test_")

# Runs, pcaps, and any other artifact a test happens to produce go to a temp dir
# that nothing but the test process reads.
os.environ.setdefault("GEMINI_DATA_DIR", os.path.join(_TMP, "runs"))
os.environ.setdefault("PCAP_DIR", os.path.join(_TMP, "pcaps"))
# No test may reach the network. Mock mode is the default so that a missing
# monkeypatch fails as a wrong assertion rather than as a billable API call.
os.environ.setdefault("GEMINI_MOCK", "1")
os.environ.setdefault("GEMINI_API_KEY", "")
