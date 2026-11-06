"""Run approved EXP-06 development folds; never scores the reserved holdout."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'research/EXP-06/config/experiment_config.json')
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from ml.research_v2.runner import run
    run(json.loads(args.config.read_text(encoding='utf-8')), args.dataset, args.output)


if __name__ == '__main__':
    main()
