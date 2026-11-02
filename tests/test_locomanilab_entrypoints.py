"""CPU checks for previous converter entry points; no policy packages required."""

import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
ENTRY_POINTS = [('scripts/data/soma_sonic_to_psi_lerobot.py', 'scripts/data/locomanilab_sonic_to_psi_lerobot.py'), ('scripts/data/soma_heft_to_psi_lerobot.py', 'scripts/data/locomanilab_heft_to_psi_lerobot.py'), ('scripts/data/soma_amo_to_psi_lerobot.py', 'scripts/data/locomanilab_amo_to_psi_lerobot.py')]


@pytest.mark.parametrize("old, canonical", ENTRY_POINTS)
def test_previous_cli_forwards_arguments_and_exit_code(tmp_path, old, canonical):
    previous = tmp_path / Path(old).name
    target = tmp_path / Path(canonical).name
    shutil.copy2(ROOT / old, previous)
    target.write_text(
        "import json, sys\n"
        "from pathlib import Path\n"
        "def main():\n"
        "    print(json.dumps([Path(__file__).name, sys.argv[1:]]))\n"
        "    raise SystemExit(7)\n"
    )
    result = subprocess.run(
        [sys.executable, str(previous), "--example", "value with spaces"],
        cwd=tmp_path.parent, text=True, capture_output=True,
    )
    assert result.returncode == 7
    assert json.loads(result.stdout) == [target.name, ["--example", "value with spaces"]]
    assert target.name in result.stderr


@pytest.mark.parametrize("old, canonical", ENTRY_POINTS)
def test_previous_import_reuses_the_canonical_module(tmp_path, old, canonical):
    previous = tmp_path / Path(old).name
    target = tmp_path / Path(canonical).name
    shutil.copy2(ROOT / old, previous)
    target.write_text("class Converter: pass\ndef main(): return 9\n")
    code = (
        "import importlib, sys; sys.path.insert(0, sys.argv[1]); "
        "previous = importlib.import_module(sys.argv[2]); "
        "canonical = importlib.import_module(sys.argv[3]); "
        "assert previous.Converter is canonical.Converter; "
        "assert previous.main is canonical.main; "
        "assert previous.main() == 9"
    )
    subprocess.run(
        [sys.executable, "-c", code, str(tmp_path), previous.stem, target.stem],
        cwd=tmp_path.parent, check=True,
    )
