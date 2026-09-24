#!/usr/bin/env python3
from pathlib import Path
import argparse, hashlib, json, shutil

MIB=1024*1024
def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(4*MIB),b""): h.update(b)
    return h.hexdigest()

ap=argparse.ArgumentParser(description="Install frozen DamageACT-U external assets into a clone.")
ap.add_argument("--assets-root",type=Path,required=True)
ap.add_argument("--repo-root",type=Path,default=Path("."))
ap.add_argument("--overwrite",action="store_true")
args=ap.parse_args()
repo=args.repo_root.resolve()
assets=args.assets_root.resolve()
manifest=json.loads((repo/"release/external_assets_manifest.json").read_text(encoding="utf-8"))

installed=0
for a in manifest["assets"]:
    src=assets/a["path"]
    if not src.is_file():
        raise RuntimeError(f"Missing external asset: {src}")
    if src.stat().st_size != int(a["bytes"]) or sha(src) != a["sha256"]:
        raise RuntimeError(f"External asset identity mismatch: {a['path']}")
    dst=repo/a["path"]
    dst.parent.mkdir(parents=True,exist_ok=True)
    if dst.exists():
        if dst.stat().st_size==int(a["bytes"]) and sha(dst)==a["sha256"]:
            continue
        if not args.overwrite:
            raise RuntimeError(f"Destination exists with wrong identity: {dst}; use --overwrite only after review.")
    shutil.copy2(src,dst)
    if dst.stat().st_size != int(a["bytes"]) or sha(dst) != a["sha256"]:
        raise RuntimeError(f"Installed asset verification failed: {a['path']}")
    installed += 1

print("EXTERNAL ASSET INSTALL: PASS")
print("Manifest assets:",len(manifest["assets"]))
print("Newly copied:",installed)
