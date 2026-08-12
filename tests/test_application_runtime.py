"""Production import-boundary checks for the application and worker."""

import subprocess
import sys
from pathlib import Path

from app.core.config import Settings


def test_api_startup_import_does_not_load_research_ml_dependencies() -> None:
    script = (
        "import sys; import app.main; "
        "blocked={'torch','lightgbm','experiments.neural_retrieval.training',"
        "'experiments.neural_retrieval.service','experiments.ranker.training'}; "
        "assert blocked.isdisjoint(sys.modules), blocked.intersection(sys.modules)"
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_api_startup_does_not_activate_internal_categorized_policy() -> None:
    script = (
        "import sys; import app.main; "
        "assert 'app.services.categorized_recommendation_service' not in sys.modules; "
        "from app.services.recommendation_service import RECOMMENDATION_STRATEGY; "
        "assert RECOMMENDATION_STRATEGY == 'SVD_Mean_Pooling'"
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_standard_requirements_exclude_research_only_dependencies() -> None:
    requirements = Path("requirements.txt").read_text(encoding="utf-8").lower()

    assert "torch" not in requirements
    assert "lightgbm" not in requirements
    ranker_requirements = (
        Path("requirements-ranker.txt").read_text(encoding="utf-8").lower()
    )
    assert "-r requirements.txt" in ranker_requirements
    assert "lightgbm==4.7.0" in ranker_requirements


def test_application_settings_have_no_neural_or_backend_selector() -> None:
    setting_names = {name.upper() for name in Settings.model_fields}

    assert not any(name.startswith("NCF_") for name in setting_names)
    assert "RECOMMENDATION_BACKEND" not in setting_names


def test_readme_describes_one_live_backend_and_isolates_neural_research() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "live `SVD_Mean_Pooling` service" in readme
    assert "experiments/neural_retrieval/" in readme
    assert "**Neural Option:**" not in readme
    assert "Both backends" not in readme
    assert "explicitly selectable" not in readme
