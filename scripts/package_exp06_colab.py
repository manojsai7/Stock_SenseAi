"""Package verified StockSense EXP-06 source code and configuration for Google Colab."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]

INCLUDED_FILES = [
    "ml/__init__.py",
    "ml/features/stationary.py",
    "ml/preprocessing/stationary_scaler.py",
    "ml/research_v2/__init__.py",
    "ml/research_v2/evaluation.py",
    "ml/research_v2/models.py",
    "ml/research_v2/pipeline.py",
    "ml/research_v2/readiness.py",
    "ml/research_v2/experiment_lock.py",
    "ml/research_v2/runner.py",
    "research/EXP-06/config/experiment_config.json",
    "research/EXP-06/config/exploratory_config.json",
    "research/EXP-06/requirements-colab.txt",
    "research/EXP-06/SELECTION_PROTOCOL.md",
    "research/EXP-06/EXPERIMENT_PREREGISTRATION.md",
    "research/EXP-06/EXPLORATORY_DEVELOPMENT_ADDENDUM.md",
    "research/EXP-06/HOLDOUT_DESIGN.md",
    "research/EXP-06/DATA_QUALITY_AUDIT.md",
    "research/EXP-06/TRAINING_GATE_REPORT.md",
    "research/EXP-05/DATA_PROVENANCE_INVESTIGATION.md",
]


def build_package(destination_zip: Path):
    destination_zip.parent.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for relative_path in INCLUDED_FILES:
        full_path = ROOT / relative_path
        if not full_path.is_file():
            raise FileNotFoundError(f"Missing required package file: {full_path}")
        manifest[relative_path] = hashlib.sha256(full_path.read_bytes()).hexdigest()

    with zipfile.ZipFile(destination_zip, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for relative_path in INCLUDED_FILES:
            z.write(ROOT / relative_path, arcname=relative_path)
        z.writestr("source_manifest.json", json.dumps(manifest, indent=2) + "\n")

    print(f"Package created at: {destination_zip}")
    print(f"Files bundled: {len(INCLUDED_FILES)} + source_manifest.json")
    print(f"Package sha256: {hashlib.sha256(destination_zip.read_bytes()).hexdigest()}")
    return destination_zip


if __name__ == "__main__":
    out = ROOT / "research/EXP-06/artifacts/stocksense_exp06_source.zip"
    build_package(out)
