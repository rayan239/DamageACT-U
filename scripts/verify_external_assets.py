#!/usr/bin/env python3
from pathlib import Path
import argparse, hashlib, json

MIB=1024*1024
def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(4*MIB),b""): h.update(b)
    return h.hexdigest()

ap=argparse.ArgumentParser()
ap.add_argument("--root",type=Path,default=Path("."),
                help="Root containing assets at their repository-relative paths.")
args=ap.parse_args()
repo=Path(__file__).resolve().parents[1]
manifest=json.loads((repo/"release/external_assets_manifest.json").read_text(encoding="utf-8"))
root=args.root.resolve()
bad=[]
for a in manifest["assets"]:
    p=root/a["path"]
    if not p.is_file():
        bad.append((a["path"],"MISSING")); continue
    if p.stat().st_size!=a["bytes"]:
        bad.append((a["path"],f"SIZE {p.stat().st_size} != {a['bytes']}")); continue
    got=sha(p)
    if got!=a["sha256"]: bad.append((a["path"],f"SHA {got} != {a['sha256']}"))
if bad:
    print("EXTERNAL ASSET VERIFICATION: FAIL")
    for x in bad: print(" ",x[0],x[1])
    raise SystemExit(1)
print("EXTERNAL ASSET VERIFICATION: PASS")
print("Verified assets:",len(manifest["assets"]))
