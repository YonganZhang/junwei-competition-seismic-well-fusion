from __future__ import annotations

from pathlib import Path


TRACK_DIR = Path(__file__).resolve().parents[1]
EVIDENCE_PATH = TRACK_DIR / "_outputs" / "agent_chapter" / "evidence.md"
README_PATH = TRACK_DIR / "README.md"


def test_agent_chapter_evidence_exists_and_covers_all_targets():
    text = EVIDENCE_PATH.read_text(encoding="utf-8")

    for target in ["T1", "T2", "T3", "T4", "T5", "T6", "T7"]:
        assert target in text

    assert "未验证" in text
    assert "not_feasible" in text
    assert "blocked" in text
    assert "test.h5" in text
    assert "frozen holdout" in text


def test_agent_chapter_does_not_leak_secrets_or_readme_mentions_it():
    text = EVIDENCE_PATH.read_text(encoding="utf-8")
    readme = README_PATH.read_text(encoding="utf-8")

    assert "DEEPSEEK_KEY" not in text
    assert "Authorization: Bearer" not in text
    assert "_outputs/agent_chapter/evidence.md" in readme
