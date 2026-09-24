#!/usr/bin/env python3
from pathlib import Path
import csv, hashlib, json, subprocess
from collections import Counter

EXPECTED_G5B_COMMIT="98b88bd"
EXPECTED_PROTOCOL_SHA256="802b8392795b93bcec3152af06eae500aaf66a953c6443dd57baea694e38d79f"
CROP_SHA="44e2cce1e36214635639cc0b9e2e981a6b1be94e813c78699c5503d86580077f"
LABELS_SHA="13f8b6a7890128e6f0002c5a2c37d15436b60d8a3d5d289803199dd1e835c969"
EXPECTED_G5B_PROTOCOL_SHA="03fd80ad7189a8ab7882347e26f4ed45aad0948073159036dfab0cc708017bb2"
EXPECTED_COUNTS={
    "EXTERNAL_CANONICAL_ASSET":11,
    "EXTERNAL_PROVENANCE_ASSET":1,
    "GENERATED_EXCLUDE":1,
    "GIT_CANONICAL":46,
    "GIT_HISTORICAL_PROVENANCE":5,
    "LOCAL_PROVENANCE_ONLY":30,
    "REVIEW_REQUIRED":0,
}
MIB=1024*1024

def require(c,m):
    if not c: raise RuntimeError(m)

def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(4*MIB),b""): h.update(b)
    return h.hexdigest()

def git(repo,*args):
    return subprocess.check_output(["git",*args],cwd=repo,text=True,stderr=subprocess.STDOUT).strip()

def git_ignored(repo,path):
    r=subprocess.run(["git","check-ignore","-q","--",path],cwd=repo)
    return r.returncode==0

def load_csv(path):
    with open(path,newline="",encoding="utf-8") as f:
        return list(csv.DictReader(f))

def write_csv(path,rows,fields):
    with open(path,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def main():
    repo=Path(".").resolve()
    require((repo/".git").exists(),"Run from repository root.")
    proto=repo/"audit/g5c_finalize_release_tree_protocol.json"
    require(proto.is_file(),"Missing G5C protocol.")
    require(sha(proto)==EXPECTED_PROTOCOL_SHA256,"G5C protocol SHA mismatch.")

    head=git(repo,"rev-parse","HEAD")
    rc=subprocess.run(["git","merge-base","--is-ancestor",EXPECTED_G5B_COMMIT,head],
                      cwd=repo,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode
    require(rc==0,"Required frozen G5B commit 98b88bd is not an ancestor of HEAD.")
    require(subprocess.run(["git","diff","--quiet"],cwd=repo).returncode==0,
            "Tracked unstaged changes exist before G5C.")
    require(subprocess.run(["git","diff","--cached","--quiet"],cwd=repo).returncode==0,
            "Staged changes exist before G5C.")

    g5b_summary=repo/"audit/g5_release_manifest/g5b_release_manifest_summary.json"
    g5b_csv=repo/"audit/g5_release_manifest/g5b_release_disposition.csv"
    require(g5b_summary.is_file() and g5b_csv.is_file(),"Frozen G5B manifests missing.")
    s=json.loads(g5b_summary.read_text(encoding="utf-8"))
    require(s.get("status")=="PASS_WITH_REVIEW_REQUIRED","Unexpected G5B status.")
    require(s.get("protocol_sha256")==EXPECTED_G5B_PROTOCOL_SHA,"Unexpected G5B protocol identity.")
    require(s.get("review_required_count")==2,"G5B review-item count changed.")
    review_paths=sorted(x["path"] for x in s.get("review_required_files",[]))
    require(review_paths==["configs/data/crop.yaml","configs/data/labels.yaml"],
            f"Unexpected G5B review set: {review_paths}")

    crop=repo/"configs/data/crop.yaml"; labels=repo/"configs/data/labels.yaml"
    require(crop.is_file() and labels.is_file(),"Resolved config files missing.")
    require(crop.stat().st_size==111 and sha(crop)==CROP_SHA,"crop.yaml identity mismatch.")
    require(labels.stat().st_size==116 and sha(labels)==LABELS_SHA,"labels.yaml identity mismatch.")

    rows=load_csv(g5b_csv)
    require(len(rows)==94,f"Expected 94 G5B rows, found {len(rows)}")
    found=set()
    for r in rows:
        if r["path"] in ("configs/data/crop.yaml","configs/data/labels.yaml"):
            require(r["final_disposition"]=="REVIEW_REQUIRED",
                    f"Review item already changed unexpectedly: {r['path']}")
            r["final_disposition"]="GIT_CANONICAL"
            r["reason"]="Resolved G5C: frozen small data configuration; accidental nested data/ ignore."
            found.add(r["path"])
    require(found=={"configs/data/crop.yaml","configs/data/labels.yaml"},"Could not resolve both config rows.")

    counts=Counter(r["final_disposition"] for r in rows)
    for k,v in EXPECTED_COUNTS.items():
        require(counts.get(k,0)==v,f"Final disposition count mismatch {k}: {counts.get(k,0)} != {v}")
    require(not [r for r in rows if r["final_disposition"]=="REVIEW_REQUIRED"],
            "Unresolved release items remain.")

    # Narrow .gitignore repair.
    gi=repo/".gitignore"
    original=gi.read_text(encoding="utf-8")
    lines=original.splitlines()
    exact_data=[i for i,x in enumerate(lines) if x.strip()=="data/"]
    require(len(exact_data)==1,f"Expected exactly one unanchored data/ line, found {len(exact_data)}")
    lines[exact_data[0]]="/data/"

    marker="# --- DamageACT-U release asset policy (G5C) ---"
    require(marker not in original,"G5C release block already present unexpectedly.")
    block=[
        "",
        marker,
        "# External canonical assets: retained locally / published separately, not ordinary Git.",
        "/checkpoints/experts/",
        "/manifests/buildings/",
        "/predictions/development/*.csv.gz",
        "/predictions/heldout_events/*.csv.gz",
        "/DamageACTU_G3_Parity_Execution_Bundle.bin",
        "",
        "# Local execution/distribution provenance; hashes are frozen in audit manifests.",
        "/dist/",
        "",
        "# Transitional generated report; canonical v2 reproduction outputs are separately tracked.",
        "/results/reproduction_report.json",
        "",
        "# Small frozen router checkpoint is intentionally versioned.",
        "!/checkpoints/routers/",
        "!/checkpoints/routers/final_neural_safe_router.pt",
        "# --- end DamageACT-U G5C release asset policy ---",
    ]
    gi.write_text("\n".join(lines+block)+"\n",encoding="utf-8")

    # Verify ignore semantics after repair.
    for p in [
        "configs/data/crop.yaml",
        "configs/data/labels.yaml",
        "vendor/frozen_phase7d_source/src/data/build_manifest.py",
        "vendor/frozen_phase7d_source/src/data/crop_utils.py",
        "vendor/frozen_phase7d_source/src/data/dataset.py",
        "checkpoints/routers/final_neural_safe_router.pt",
    ]:
        require(not git_ignored(repo,p),f"Git-canonical path still ignored after G5C: {p}")

    for p in [
        "checkpoints/experts/post_seed42_best_state_dict.pt",
        "checkpoints/experts/siamese_seed42_best_state_dict.pt",
        "manifests/buildings/event_train_buildings.csv.gz",
        "predictions/heldout_events/event_test_clean_expert_predictions.csv.gz",
        "DamageACTU_G3_Parity_Execution_Bundle.bin",
        "dist/DamageACTU_G3_Parity_Execution_Bundle.zip",
        "results/reproduction_report.json",
    ]:
        if (repo/p).exists():
            require(git_ignored(repo,p),f"External/local/generated path not ignored after G5C: {p}")

    out=repo/"audit/g5_release_manifest"
    final_csv=out/"g5c_final_release_disposition.csv"
    fields=list(rows[0].keys())
    write_csv(final_csv,rows,fields)

    external=[r for r in rows if r["final_disposition"] in
              ("EXTERNAL_CANONICAL_ASSET","EXTERNAL_PROVENANCE_ASSET")]
    ext_payload={
        "schema_version":1,
        "status":"FROZEN",
        "storage_target":"Zenodo or GitHub Release assets",
        "note":"URLs/DOI are intentionally null until publication; verify exact SHA256 after download.",
        "asset_count":len(external),
        "assets":[{
            "path":r["path"],
            "bytes":int(r["bytes"]),
            "sha256":r["sha256"],
            "role":r["final_disposition"],
            "download_url":None
        } for r in external]
    }
    release_dir=repo/"release"; release_dir.mkdir(exist_ok=True)
    ext_json=release_dir/"external_assets_manifest.json"
    ext_json.write_text(json.dumps(ext_payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    sums=release_dir/"SHA256SUMS.external.txt"
    sums.write_text("".join(f"{a['sha256']}  {a['path']}\n" for a in ext_payload["assets"]),
                    encoding="utf-8")

    release_md=release_dir/"README.md"
    release_md.write_text(
f"""# DamageACT-U external release assets

The Git repository intentionally keeps large frozen assets outside ordinary Git.

Expected external assets: **{len(external)}**

After obtaining the release/Zenodo asset package, restore the files to the exact
relative paths listed in `external_assets_manifest.json`, then run:

```bash
python scripts/verify_external_assets.py
```

The verifier checks file size and SHA256. No scientific result should be accepted
from an asset whose checksum does not match.

The canonical paper pipeline remains:

Phase7C split audit -> Phase7D-A building manifests -> Phase7D-B Seed-42 experts ->
Phase7D-C frozen routers -> Phase7D-D sealed held-out evaluation.

The final event-track evidence is single-initialization (Seed 42); random-seed
robustness was not established.
""",encoding="utf-8")

    summary={
        "stage":"G5C_finalize_release_tree_and_external_assets",
        "status":"PASS",
        "head_before_g5c":head,
        "protocol_sha256":EXPECTED_PROTOCOL_SHA256,
        "resolved_review_items":{
            "configs/data/crop.yaml":CROP_SHA,
            "configs/data/labels.yaml":LABELS_SHA
        },
        "final_disposition_counts":dict(sorted(counts.items())),
        "review_required_count":0,
        "external_asset_count":len(external),
        "gitignore_nested_data_rule_fixed":True,
        "scientific_file_contents_modified":False,
        "scientific_models_run":False,
        "files_deleted":False,
        "files_moved":False,
        "next_gate":"Stage allowlisted Git release files, package external assets, commit G5C, then run G6 final release audit."
    }
    sj=out/"g5c_finalize_release_tree_summary.json"
    sj.write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n",encoding="utf-8")

    print("=== G5C FINALIZE RELEASE TREE ===")
    print("HEAD before G5C:",head)
    print("Protocol SHA256:",EXPECTED_PROTOCOL_SHA256)
    print("Resolved crop.yaml: GIT_CANONICAL")
    print("Resolved labels.yaml: GIT_CANONICAL")
    print("Final disposition counts:")
    for k,v in sorted(counts.items()): print(f"  {k}: {v}")
    print("Review required: 0")
    print("External assets:",len(external))
    print(".gitignore nested data/ collision repaired: PASS")
    print("External/local/generated ignore policy: PASS")
    print("G5C FINALIZATION: PASS")
    print("FINAL DISPOSITION:",final_csv)
    print("EXTERNAL ASSET MANIFEST:",ext_json)
    print("SUMMARY:",sj)
    print("SUMMARY SHA256:",sha(sj))

if __name__=="__main__":
    main()
