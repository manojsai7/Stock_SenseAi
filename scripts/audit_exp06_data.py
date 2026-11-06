"""Audit an immutable RELIANCE.NS input snapshot and write a new output directory."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml.research_v2.readiness import audit_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--calendar", type=Path, help="Authoritative daily session CSV with Date column")
    parser.add_argument("--evidence", type=Path, help="Snapshot and calendar provenance JSON")
    args = parser.parse_args()
    result = audit_dataset(args.input, args.output, calendar_path=args.calendar, evidence_path=args.evidence)
    print(f"{result['status']} | {result['candidate_rows']} candidate rows | {result['quarantined_rows']} quarantined")
    for blocker in result["blockers"]:
        print(f"  {blocker}")


if __name__ == "__main__":
    main()
