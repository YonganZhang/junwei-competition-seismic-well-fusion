from __future__ import annotations

import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import agent_chapter as ac  # noqa: E402

def test_discover_latest_result_files_points_to_real_outputs() -> None:
    files = ac.discover_latest_result_files()
    assert set(files) == {
        "metrics.json",
        "run_manifest.json",
        "build_report.json",
        "split_manifest.json",
        "checkpoints/history.json",
    }
    for item in files.values():
        assert item.path.is_file()
        assert item.size_bytes > 0
        assert item.sha256
        assert "test.h5" not in item.path.as_posix()


def test_generate_evidence_uses_real_development_inputs_and_redacts_secret(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(ac, "call_deepseek", lambda prompt, **kwargs: "DeepSeek: placeholder analysis\n- 未验证: needs retraining\n")
    report = ac.generate_evidence(tmp_path / "evidence.md")
    output = tmp_path / "evidence.md"
    text = output.read_text(encoding="utf-8")
    assert output.is_file()
    assert report["output_path"] == output.as_posix()
    assert report["prompt_chars"] > 1000
    assert report["analysis_chars"] > 0
    assert "# 智能体分析章节" in text
    assert "DeepSeek: placeholder analysis" in text
    assert "未验证" in text
    assert "DEEPSEEK_KEY" not in text
    assert "_data/processed/reservoir/train.h5" in text
    assert "_outputs/guard.npz" in text
