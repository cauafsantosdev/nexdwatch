"""Tests for recommendation artifact CLI commands."""

from pathlib import Path

from typer.testing import CliRunner

import manage
from app.ml.faiss_index import FaissIndexBuildResult


def test_build_index_command_reports_artifact_details(monkeypatch, tmp_path) -> None:
    result = FaissIndexBuildResult(
        film_count=123,
        dimension=32,
        output_path=tmp_path / "retrieval.faiss",
    )
    monkeypatch.setattr(manage, "rebuild_faiss_index", lambda _: result)

    command_result = CliRunner().invoke(manage.app, ["build-index"])

    assert command_result.exit_code == 0
    assert "123 films" in command_result.output
    assert "dimension 32" in command_result.output
    assert str(result.output_path) in command_result.output


def test_build_index_command_returns_nonzero_on_failure(monkeypatch) -> None:
    def fail(_: Path) -> None:
        raise ValueError("invalid artifacts")

    monkeypatch.setattr(manage, "rebuild_faiss_index", fail)

    command_result = CliRunner().invoke(manage.app, ["build-index"])

    assert command_result.exit_code == 1
    assert "Index rebuild failed: invalid artifacts" in command_result.output
