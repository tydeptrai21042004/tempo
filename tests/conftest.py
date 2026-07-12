from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from realtime_watermark.core import make_demo_host, make_demo_watermark


@pytest.fixture(scope="session")
def host128():
    return make_demo_host(128, 128)


@pytest.fixture(scope="session")
def wm16():
    return make_demo_watermark(16)


@pytest.fixture(scope="session")
def host192():
    return make_demo_host(192, 192)
