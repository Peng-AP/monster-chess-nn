"""Pytest configuration: ensure src/ is importable."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
