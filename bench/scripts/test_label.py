#!/usr/bin/env python3
"""Unit tests for bench/scripts/label.py (no GPU required)."""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LABEL = HERE / "label.py"


def run_label(*args: str) -> dict:
    out = subprocess.check_output([sys.executable, str(LABEL), *args], text=True)
    assert out.startswith("RESULT_JSON ")
    return json.loads(out[len("RESULT_JSON "):])


def test_accepts_valid_improvement():
    res = run_label("170", "164", "366", "1.0", "0.14", "abc1234")
    assert res["pass"] is True
    assert res["label"] in {"XL", "L", "M", "S", "XS", "none"}


def test_rejects_llama_correctness_failure():
    res = run_label("170", "164", "366", "0.85", "0.14", "abc1234")
    assert res["pass"] is False
    assert res["label"] == "REJECT"
    assert "llama.cpp" in res["reason"]


def test_rejects_baseline_drift():
    res = run_label("170", "164", "366", "1.0", "0.14", "abc1234",
                    "0.95", "0.02", "base001")
    assert res["pass"] is False
    assert res["label"] == "REJECT"
    assert "baseline drift" in res["reason"]
    assert res["baseline_top1"] == 0.95


def test_accepts_baseline_match():
    res = run_label("170", "164", "366", "1.0", "0.14", "abc1234",
                    "1.0", "0.0", "base001")
    assert res["pass"] is True
    assert res["baseline_top1"] == 1.0
    assert res["baseline_selfkl"] == 0.0


def test_baseline_gate_without_llama_failure():
    # llama gate passes; baseline fails
    res = run_label("170", "0", "0", "0.95", "0.1", "abc1234",
                    "0.98", "0.005", "base001")
    assert res["pass"] is False
    assert "baseline drift" in res["reason"]


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
