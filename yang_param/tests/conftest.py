import pytest
from pathlib import Path

YANG_DIR = str(Path(__file__).parent.parent / "data" / "yang")

@pytest.fixture
def yang_dir():
    return YANG_DIR
