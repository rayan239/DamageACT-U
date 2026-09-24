#!/usr/bin/env python3
from pathlib import Path
import argparse, subprocess, sys, json

ap=argparse.ArgumentParser(description="Public DamageACT-U frozen-result reproduction entry point.")
ap.add_argument("--assets-root",type=Path,default=None,
                help="Optional external-assets root. If supplied, assets are verified/installed first.")
args=ap.parse_args()
repo=Path(__file__).resolve().parents[1]

def run(cmd):
    print("\n$", " ".join(str(x) for x in cmd), flush=True)
    subprocess.check_call([str(x) for x in cmd], cwd=repo)

if args.assets_root is not None:
    run([sys.executable, "scripts/install_external_assets.py",
         "--assets-root", str(args.assets_root), "--repo-root", str(repo)])

run([sys.executable, "scripts/verify_external_assets.py", "--root", str(repo)])
run([sys.executable, "-m", "pytest", "-q"])
run([sys.executable, "scripts/verify_restored_evidence_v2.py"])
run([sys.executable, "scripts/reproduce_results_v2_core.py"])
run([sys.executable, "scripts/reproduce_results_v2_bootstrap.py"])

core=json.loads((repo/"results/reproduction_core_v2.json").read_text(encoding="utf-8"))
boot=json.loads((repo/"results/reproduction_bootstrap_v2.json").read_text(encoding="utf-8"))

# Do not infer a stronger conclusion than the frozen evidence.
if core.get("all_precommitted_criteria_pass", core.get("ALL_PRECOMMITTED_CRITERIA_PASS")) is True:
    raise RuntimeError("Frozen all-precommitted flag unexpectedly became TRUE.")
# Bootstrap schema can vary; textual scripts already enforce exact frozen criteria.

print("\nDAMAGEACT-U PUBLIC FROZEN REPRODUCTION: PASS")
print("Scope: saved artifacts / checkpoint inference / frozen router and statistical reproduction.")
print("Training-from-scratch reproduction is NOT claimed by this command.")
print("Final event-track evidence remains single-initialization (Seed 42).")
