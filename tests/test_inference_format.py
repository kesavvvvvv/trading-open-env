from __future__ import annotations

import re
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_inference_file_exists():
    path = ROOT / "inference.py"
    assert path.exists(), "inference.py must exist at repository root"


def test_inference_contains_required_log_tokens():
    text = (ROOT / "inference.py").read_text(encoding="utf-8")
    assert "[START]" in text
    assert "[STEP]" in text
    assert "[END]" in text
    assert "OPENAI_API_KEY" in text or "API_BASE_URL" in text or "MODEL_NAME" in text
    assert "OpenAI" in text


def test_inference_output_format_contract_is_documented():
    text = (ROOT / "inference.py").read_text(encoding="utf-8")
    pattern = re.compile(r"\[START\].*\[STEP\].*\[END\]", re.DOTALL)
    assert pattern.search(text), "inference.py should clearly implement the required log sequence"


def test_inference_can_be_imported_or_has_main_guard():
    text = (ROOT / "inference.py").read_text(encoding="utf-8")
    assert "if __name__ == \"__main__\"" in text or "if __name__ == '__main__'" in text
