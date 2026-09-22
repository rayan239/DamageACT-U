import argparse
from damageactu.utils.frozen_source import verify_frozen_source


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frozen-project-root", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.seed != 42:
        raise RuntimeError("The paper event-track release contains only completed expert Seed 42. Additional seeds are new experiments.")
    verify_frozen_source(args.frozen_project_root)
    raise SystemExit(
        "Frozen source verified. Full expert retraining is intentionally delegated to the archived "
        "Phase7D-B v7 resumable trainer until its training loop is parity-ported into this clean CLI. "
        "This prevents an unverified rewrite from being presented as the canonical experiment."
    )


if __name__ == "__main__":
    main()
