#!/usr/bin/env python3
from pathlib import Path
import argparse, hashlib, json, shutil

MIB=1024*1024
def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(4*MIB),b""): h.update(b)
    return h.hexdigest()

ap=argparse.ArgumentParser()
ap.add_argument("--output",type=Path,required=True)
args=ap.parse_args()
repo=Path(__file__).resolve().parents[1]
manifest=json.loads((repo/"release/external_assets_manifest.json").read_text(encoding="utf-8"))
out=args.output.resolve()
out.mkdir(parents=True,exist_ok=True)

for a in manifest["assets"]:
    src=repo/a["path"]
    if not src.is_file():
        raise RuntimeError(f"Missing source asset: {a['path']}")
    if src.stat().st_size!=a["bytes"] or sha(src)!=a["sha256"]:
        raise RuntimeError(f"Source asset identity mismatch: {a['path']}")
    dst=out/a["path"]
    dst.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(src,dst)
    if dst.stat().st_size!=a["bytes"] or sha(dst)!=a["sha256"]:
        raise RuntimeError(f"Copied asset identity mismatch: {a['path']}")
(out/"external_assets_manifest.json").write_text(
    json.dumps(manifest,indent=2,sort_keys=True)+"\n",encoding="utf-8")
(out/"SHA256SUMS.txt").write_text(
    "".join(f"{a['sha256']}  {a['path']}\n" for a in manifest["assets"]),
    encoding="utf-8")
print("EXTERNAL ASSET PACKAGE: PASS")
print("Output:",out)
print("Assets:",len(manifest["assets"]))
