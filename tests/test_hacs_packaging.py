import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
COMPONENT_ROOT = REPO_ROOT / "custom_components" / "home_energy_planner"


def test_hacs_json_shape():
    payload = json.loads((REPO_ROOT / "hacs.json").read_text(encoding="utf-8"))
    assert payload["name"] == "Home Energy Planner"
    assert "homeassistant" in payload


def test_manifest_shape():
    payload = json.loads((COMPONENT_ROOT / "manifest.json").read_text(encoding="utf-8"))
    assert payload["domain"] == "home_energy_planner"
    assert payload["config_flow"] is True
    assert payload["version"]
    assert payload["documentation"].startswith("https://")


def test_translations_parse():
    payload = json.loads(
        (COMPONENT_ROOT / "translations" / "en.json").read_text(encoding="utf-8")
    )
    assert "config" in payload
    assert "options" in payload


def test_component_files_exist():
    for name in ("__init__.py", "config_flow.py", "const.py", "coordinator.py", "pricing.py", "sensor.py"):
        assert (COMPONENT_ROOT / name).is_file(), name


def test_pure_core_has_no_ha_imports():
    source = (COMPONENT_ROOT / "pricing.py").read_text(encoding="utf-8")
    assert "homeassistant" not in source
