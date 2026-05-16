"""Test fixtures + global mocks for pycuda/tensorrt (not installable on Windows host)."""
import sys
from pathlib import Path
from unittest.mock import MagicMock

# Mock GPU libs before any test imports scripts.nano_server
sys.modules.setdefault('pycuda', MagicMock())
sys.modules.setdefault('pycuda.driver', MagicMock())
sys.modules.setdefault('tensorrt', MagicMock())

# Make `scripts/` importable as a top-level package for tests
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
