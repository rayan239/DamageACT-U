#!/usr/bin/env python3
from pathlib import Path
import argparse, csv, hashlib, json, subprocess
from collections import Counter, defaultdict

EXPECTED_G5A_COMMIT="6bb5180"
EXPECTED_PROTOCOL_SHA256="03fd80ad7189a8ab7882347e26f4ed45aad0948073159036dfab0cc708017bb2"
EXPECTED_G5A_HASHES={
"audit/g5_repository_inventory/g5a_repository_inventory_summary.json":"d9152f74cb49b05f72cc6c6531671cf83c0ce877662e5d7869ee6036586a0538",
"audit/g5_repository_inventory/g5a_untracked_inventory.csv":"5a6900d0cf5a964bf07852ad01347b310d6d7e7327fc1d4e2b0805021e25f1df",
"audit/g5_repository_inventory/g5a_tracked_inventory.csv":"7186e37a34b676a21b7187cb3592ccc4cdcc7a946729580281bb0ab2d82876bc",
"audit/g5_repository_inventory/g5a_duplicate_hash_groups.json":"e840b49c5de18060e23a5a88c2616bf7f84362b3b46a08bc9c5f6d452ffdcafa",
}
MIB=1024*1024
RELEVANT_TOPS={"checkpoints","manifests","predictions","results","vendor","audit","scripts","dist","notebooks","configs","environment"}

def require(c,m):
    if not c: raise RuntimeError(m)

def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(4*MIB),b""): h.update(b)
    return h.hexdigest()

def git(repo,*args,raw=False):
    out=subprocess.check_output(["git",*args],cwd=repo)
    return out if raw else out.decode().strip()

def norm(p): return p.replace("\\","/")

def classify(path,size):
    p=norm(path)
    # External canonical assets: keep exact bytes/hashes, publish outside ordinary Git.
    if p.startswith("checkpoints/experts/"):
        return "EXTERNAL_CANONICAL_ASSET","Expert checkpoint; publish as verified release/Zenodo asset."
    if p.startswith("manifests/buildings/"):
        return "EXTERNAL_CANONICAL_ASSET","Large canonical building manifest; publish externally with SHA256."
    if p.startswith("predictions/"):
        return "EXTERNAL_CANONICAL_ASSET","Frozen prediction table; publish externally with SHA256."
    if p=="DamageACTU_G3_Parity_Execution_Bundle.bin":
        return "EXTERNAL_PROVENANCE_ASSET","Keep one canonical copy of the exact G3 execution carrier as a release provenance asset."

    # Redundant/extracted transport material. Preserve outside final working tree after manifest freeze.
    if p.startswith("dist/"):
        return "LOCAL_PROVENANCE_ONLY","Distribution/extracted bundle material; hashes retained, redundant with canonical paths or external carrier."

    # Small canonical Git artifacts.
    if p.startswith("checkpoints/routers/"):
        return "GIT_CANONICAL","Small frozen router artifact required for deterministic G4 reproduction."
    if p.startswith(("manifests/event_split/","manifests/router/","manifests/wrong_pre/")):
        return "GIT_CANONICAL","Small frozen canonical manifest."
    if p.startswith(("results/development/","results/heldout_events/")):
        return "GIT_CANONICAL","Frozen canonical result evidence."
    if p.startswith("vendor/frozen_phase7d_source/src/"):
        return "GIT_CANONICAL","Frozen historical source required for implementation fidelity."
    if p in {
        "audit/restoration_report_v2.json",
        "scripts/restore_canonical_artifacts_v2.py",
        "scripts/verify_restored_evidence_v2.py",
    }:
        return "GIT_CANONICAL","Canonical restoration/verification tooling or audit evidence."

    # Small provenance worth retaining in Git, but clearly historical.
    if p in {
        "audit/g3_clean_parity_carrier_96.csv",
        "audit/phase7dd_inference_extract.txt",
        "audit/phase7dd_source_resolver_extract.txt",
        "scripts/g3_checkpoint_parity_t4.synthetic96.pre_replay.py",
        "scripts/patch_g3_runner_for_canonical_replay.py",
    }:
        return "GIT_HISTORICAL_PROVENANCE","Small diagnostic/provenance artifact; retain but not primary reproduction path."

    if p=="results/reproduction_report.json":
        return "GENERATED_EXCLUDE","Recreatable/transitional report; final release uses frozen v2 reproduction reports."

    # Ignored caches/environments should never become release assets.
    lp=p.lower()
    if any(x in lp for x in ("__pycache__/","/.pytest_cache/","/.ipynb_checkpoints/")) or \
       lp.startswith((".venv","venv/","env/")) or lp.endswith((".pyc",".pyo",".tmp",".bak")):
        return "GENERATED_EXCLUDE","Environment/cache/editor artifact."

    return "REVIEW_REQUIRED","No predeclared safe disposition rule."

def record(repo,p,source):
    fp=repo/p
    require(fp.is_file(),f"Missing file during G5B: {p}")
    size=fp.stat().st_size
    d,r=classify(p,size)
    return {
        "path":norm(p),"source_inventory":source,"bytes":size,"mib":round(size/MIB,6),
        "sha256":sha(fp),"final_disposition":d,"reason":r
    }

def writecsv(path,rows):
    cols=["path","source_inventory","bytes","mib","sha256","final_disposition","reason"]
    with open(path,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(rows)

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--repo-root",type=Path,default=Path("."))
    repo=ap.parse_args().repo_root.resolve()
    require((repo/".git").exists(),"Not repo root.")
    proto=repo/"audit/g5b_release_disposition_protocol.json"
    require(proto.is_file(),"Missing G5B protocol.")
    require(sha(proto)==EXPECTED_PROTOCOL_SHA256,"G5B protocol SHA mismatch.")

    head=git(repo,"rev-parse","HEAD")
    rc=subprocess.run(["git","merge-base","--is-ancestor",EXPECTED_G5A_COMMIT,head],cwd=repo,
                      stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode
    require(rc==0,"Required G5A commit not ancestor of HEAD.")
    require(subprocess.run(["git","diff","--quiet"],cwd=repo).returncode==0,"Tracked unstaged changes exist.")
    require(subprocess.run(["git","diff","--cached","--quiet"],cwd=repo).returncode==0,"Staged changes exist.")

    for rel,h in EXPECTED_G5A_HASHES.items():
        p=repo/rel;require(p.is_file(),f"Missing frozen G5A input: {rel}")
        require(sha(p)==h,f"Frozen G5A input changed: {rel}")

    import pandas as pd
    g5a=pd.read_csv(repo/"audit/g5_repository_inventory/g5a_untracked_inventory.csv")
    base_paths=g5a["path"].astype(str).tolist()

    # Close G5A's --exclude-standard blind spot.
    ignored_all=[x.decode() for x in git(repo,"ls-files","--others","--ignored","--exclude-standard","-z",raw=True).split(b"\0") if x]
    ignored_top=Counter(norm(x).split("/",1)[0] for x in ignored_all)
    relevant_ignored=[]
    for x in ignored_all:
        n=norm(x); top=n.split("/",1)[0]
        if top in RELEVANT_TOPS:
            # Skip obvious caches even under release-relevant roots.
            low=n.lower()
            if any(k in low for k in ("__pycache__/","/.pytest_cache/","/.ipynb_checkpoints/")) or low.endswith((".pyc",".pyo")):
                continue
            relevant_ignored.append(n)
    relevant_ignored=sorted(set(relevant_ignored)-set(base_paths))

    rows=[record(repo,p,"G5A_UNTRACKED") for p in base_paths]
    rows += [record(repo,p,"IGNORED_GAP_SCAN") for p in relevant_ignored]
    rows=sorted(rows,key=lambda r:r["path"])

    counts=Counter(r["final_disposition"] for r in rows)
    reviews=[r for r in rows if r["final_disposition"]=="REVIEW_REQUIRED"]

    # Duplicate groups across all release-relevant rows.
    by=defaultdict(list)
    for r in rows: by[r["sha256"]].append(r["path"])
    dups=[{"sha256":h,"count":len(ps),"paths":sorted(ps)} for h,ps in by.items() if len(ps)>1]
    dups.sort(key=lambda x:(-x["count"],x["sha256"]))

    out=repo/"audit/g5_release_manifest";out.mkdir(parents=True,exist_ok=True)
    allcsv=out/"g5b_release_disposition.csv"
    writecsv(allcsv,rows)

    def subset(name,disps):
        rr=[r for r in rows if r["final_disposition"] in disps]
        payload={"dispositions":sorted(disps),"count":len(rr),"files":rr}
        (out/name).write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
        return rr

    gitfiles=subset("g5b_git_inclusion_manifest.json",{"GIT_CANONICAL","GIT_HISTORICAL_PROVENANCE"})
    extfiles=subset("g5b_external_asset_manifest.json",{"EXTERNAL_CANONICAL_ASSET","EXTERNAL_PROVENANCE_ASSET"})
    localfiles=subset("g5b_local_provenance_manifest.json",{"LOCAL_PROVENANCE_ONLY"})
    genfiles=subset("g5b_generated_exclude_manifest.json",{"GENERATED_EXCLUDE"})
    (out/"g5b_duplicate_hash_groups.json").write_text(json.dumps(dups,indent=2,sort_keys=True)+"\n",encoding="utf-8")

    summary={
        "stage":"G5B_release_manifest_and_ignored_gap_audit",
        "status":"PASS" if not reviews else "PASS_WITH_REVIEW_REQUIRED",
        "head":head,"protocol_sha256":EXPECTED_PROTOCOL_SHA256,
        "g5a_untracked_rows":len(base_paths),
        "ignored_all_file_count":len(ignored_all),
        "ignored_top_level_counts":dict(sorted(ignored_top.items())),
        "release_relevant_ignored_rows_added":len(relevant_ignored),
        "total_release_relevant_rows":len(rows),
        "disposition_counts":dict(sorted(counts.items())),
        "review_required_count":len(reviews),
        "review_required_files":reviews,
        "duplicate_hash_group_count":len(dups),
        "git_inclusion_count":len(gitfiles),
        "external_asset_count":len(extfiles),
        "local_provenance_only_count":len(localfiles),
        "generated_exclude_count":len(genfiles),
        "scientific_files_modified":False,"files_deleted":False,"files_moved":False,
        "gitignore_modified":False,"git_lfs_modified":False,
        "next_gate":"G5C apply frozen release disposition and build final reproducibility/documentation layer"
    }
    sj=out/"g5b_release_manifest_summary.json"
    sj.write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n",encoding="utf-8")

    print("=== G5B RELEASE MANIFEST + IGNORED GAP AUDIT ===")
    print("HEAD:",head)
    print("Protocol SHA256:",EXPECTED_PROTOCOL_SHA256)
    print("G5A rows:",len(base_paths))
    print("All ignored files seen:",len(ignored_all))
    print("Release-relevant ignored files added:",len(relevant_ignored))
    print("Total release-relevant rows:",len(rows))
    print("\nFinal disposition counts:")
    for k,v in sorted(counts.items()): print(f"  {k}: {v}")
    print("Duplicate SHA256 groups:",len(dups))
    print("Review-required files:",len(reviews))
    if relevant_ignored:
        print("\nNew release-relevant ignored files:")
        for r in [x for x in rows if x["source_inventory"]=="IGNORED_GAP_SCAN"]:
            print(f"  {r['mib']:9.3f} MiB  {r['final_disposition']:28s}  {r['path']}")
    if reviews:
        print("\nREVIEW REQUIRED:")
        for r in reviews: print(" ",r["path"])
    print("\nG5B STATUS:",summary["status"])
    print("SUMMARY:",sj)
    print("DISPOSITION CSV:",allcsv)
    print("SUMMARY SHA256:",sha(sj))

if __name__=="__main__": main()
