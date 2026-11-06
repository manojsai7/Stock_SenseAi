"""Package shared EXP-06 code for Colab without credentials or production weights."""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    paths = [ROOT / 'ml/__init__.py', ROOT / 'ml/features/stationary.py',
             ROOT / 'ml/preprocessing/stationary_scaler.py']
    paths += sorted((ROOT / 'ml/research_v2').glob('*.py'))
    paths += sorted((ROOT / 'research/EXP-06/config').glob('*.json'))
    paths += [ROOT / 'research/EXP-06/requirements-colab.txt',
              ROOT / 'research/EXP-06/SELECTION_PROTOCOL.md',
              ROOT / 'research/EXP-05/DATA_PROVENANCE_INVESTIGATION.md']
    if not (ROOT / 'ml/research_v2/runner.py').exists():
        raise RuntimeError('Shared experiment runner is missing')
    destination = ROOT / 'scratch/stocksense_exp06_source.zip'
    destination.parent.mkdir(exist_ok=True)
    manifest = {}
    with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            name = path.relative_to(ROOT).as_posix()
            content = path.read_bytes()
            archive.writestr(name, content)
            manifest[name] = hashlib.sha256(content).hexdigest()
        archive.writestr('source_manifest.json', json.dumps(manifest, indent=2))
    print(destination)


if __name__ == '__main__':
    main()
