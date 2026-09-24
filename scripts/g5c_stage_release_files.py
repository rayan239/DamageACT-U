#!/usr/bin/env python3
from pathlib import Path
import csv, json, subprocess, sys

APPROVED={"GIT_CANONICAL","GIT_HISTORICAL_PROVENANCE"}

def require(c,m):
    if not c: raise RuntimeError(m)

repo=Path(".").resolve()
summary=repo/"audit/g5_release_manifest/g5c_finalize_release_tree_summary.json"
manifest=repo/"audit/g5_release_manifest/g5c_final_release_disposition.csv"
require(summary.is_file() and manifest.is_file(),"Run G5C finalization first.")
s=json.loads(summary.read_text(encoding="utf-8"))
require(s.get("status")=="PASS" and s.get("review_required_count")==0,"G5C summary is not final PASS.")

with open(manifest,newline="",encoding="utf-8") as f:
    rows=list(csv.DictReader(f))
paths=sorted({r["path"] for r in rows if r["final_disposition"] in APPROVED})

# G5C-generated/modified release metadata must also be versioned.
paths += [
    ".gitignore",
    "audit/g5_release_manifest/g5c_final_release_disposition.csv",
    "audit/g5_release_manifest/g5c_finalize_release_tree_summary.json",
    "release/README.md",
    "release/external_assets_manifest.json",
    "release/SHA256SUMS.external.txt",
]
paths=sorted(set(paths))

for p in paths:
    require((repo/p).is_file(),f"Approved Git path missing: {p}")

# Force is intentional only for paths explicitly approved by the frozen manifest
# (e.g. configs/data and the small neural .pt checkpoint).
for i in range(0,len(paths),25):
    subprocess.check_call(["git","add","-f","--",*paths[i:i+25]],cwd=repo)

staged=subprocess.check_output(["git","diff","--cached","--name-only"],cwd=repo,text=True).splitlines()
unexpected=sorted(set(staged)-set(paths))
missing=sorted(set(paths)-set(staged))
require(not unexpected,f"Unexpected staged paths: {unexpected}")
require(not missing,f"Approved paths not staged: {missing}")

print("G5C ALLOWLIST STAGING: PASS")
print("Approved/staged files:",len(paths))
print("No external/local/generated asset was staged.")
for p in staged:
    print(p)
