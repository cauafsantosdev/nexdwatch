"""Automatic activation watcher and startup recovery coverage."""

import json
import os
import shutil
import signal
from unittest.mock import Mock

import pytest

from app.ml.model_registry import (
    CURRENT_POINTER_FILENAME,
    model_root,
    promote_model_bundle,
    read_current_version,
    resolve_serving_model,
    rollback_model,
)
from app.services import model_activation
from app.services.model_activation import ModelPointerWatcher, resolve_startup_model
from tests.test_model_registry import _bundle


def test_unchanged_pointer_does_no_validation_or_recycle(tmp_path, monkeypatch) -> None:
    current = _bundle(tmp_path, 1)
    promote_model_bundle(tmp_path, current.name)
    recycle = Mock()
    validate = Mock(side_effect=AssertionError("unchanged pointer must be cheap"))
    monkeypatch.setattr("app.services.model_activation.resolve_serving_model", validate)
    watcher = ModelPointerWatcher(
        tmp_path, current.name, interval_seconds=30, recycle=recycle
    )

    assert not watcher.check_once()
    validate.assert_not_called()
    recycle.assert_not_called()


def test_recycle_uses_graceful_sigterm_for_current_api_process(monkeypatch) -> None:
    kill = Mock()
    monkeypatch.setattr(model_activation.os, "kill", kill)

    model_activation.request_graceful_process_recycle()

    kill.assert_called_once_with(os.getpid(), signal.SIGTERM)


def test_new_valid_pointer_requests_exactly_one_recycle(tmp_path) -> None:
    old = _bundle(tmp_path, 1)
    new = _bundle(tmp_path, 2)
    promote_model_bundle(tmp_path, old.name)
    recycle = Mock()
    watcher = ModelPointerWatcher(
        tmp_path, old.name, interval_seconds=30, recycle=recycle
    )

    promote_model_bundle(tmp_path, new.name)
    assert watcher.check_once()
    assert not watcher.check_once()
    recycle.assert_called_once_with()


def test_malformed_or_invalid_pointer_keeps_healthy_process(tmp_path) -> None:
    old = _bundle(tmp_path, 1)
    promote_model_bundle(tmp_path, old.name)
    recycle = Mock()
    watcher = ModelPointerWatcher(
        tmp_path, old.name, interval_seconds=30, recycle=recycle
    )
    pointer = model_root(tmp_path) / CURRENT_POINTER_FILENAME

    pointer.write_text("not json", encoding="utf-8")
    assert not watcher.check_once()
    pointer.write_text(
        json.dumps(
            {
                "model_version": "20300102T010203Z-00000002",
                "previous_version": old.name,
            }
        ),
        encoding="utf-8",
    )
    assert not watcher.check_once()
    recycle.assert_not_called()


def test_rollback_pointer_change_requests_recycle(tmp_path) -> None:
    old = _bundle(tmp_path, 1)
    current = _bundle(tmp_path, 2)
    promote_model_bundle(tmp_path, old.name)
    promote_model_bundle(tmp_path, current.name)
    recycle = Mock()
    watcher = ModelPointerWatcher(
        tmp_path, current.name, interval_seconds=30, recycle=recycle
    )

    rollback_model(tmp_path)
    assert watcher.check_once()
    assert read_current_version(tmp_path) == old.name
    recycle.assert_called_once_with()


def test_failed_promotion_causes_no_pointer_change_or_recycle(
    tmp_path, monkeypatch
) -> None:
    old = _bundle(tmp_path, 1)
    candidate = _bundle(tmp_path, 2)
    promote_model_bundle(tmp_path, old.name)
    (candidate / "item_embeddings.npy").write_bytes(b"corrupt")
    recycle = Mock()
    watcher = ModelPointerWatcher(
        tmp_path, old.name, interval_seconds=30, recycle=recycle
    )

    with pytest.raises(ValueError, match="checksum"):
        promote_model_bundle(tmp_path, candidate.name)
    assert read_current_version(tmp_path) == old.name
    assert not watcher.check_once()
    recycle.assert_not_called()


def test_startup_activation_failure_restores_previous_valid_bundle(tmp_path) -> None:
    old = _bundle(tmp_path, 1)
    candidate = _bundle(tmp_path, 2)
    promote_model_bundle(tmp_path, old.name)
    promote_model_bundle(tmp_path, candidate.name)
    (candidate / "item_embeddings.npy").write_bytes(b"corrupt after promotion")

    recovered = resolve_startup_model(tmp_path)

    assert recovered.model_version == old.name
    assert read_current_version(tmp_path) == old.name
    assert resolve_serving_model(tmp_path).model_version == old.name


def test_failed_first_versioned_activation_restores_complete_legacy(tmp_path) -> None:
    candidate = _bundle(tmp_path, 1)
    for filename in ("item_embeddings.npy", "film_index.json", "retrieval.faiss"):
        shutil.copy2(candidate / filename, tmp_path / filename)
    (tmp_path / "candidates").mkdir()
    shutil.copy2(candidate / "popularity.json", tmp_path / "candidates/popularity.json")
    promote_model_bundle(tmp_path, candidate.name)
    (candidate / "retrieval.faiss").write_bytes(b"corrupt after promotion")

    recovered = resolve_startup_model(tmp_path)

    assert recovered.model_version == "legacy-flat"
    assert read_current_version(tmp_path) is None
