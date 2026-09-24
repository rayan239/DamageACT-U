#!/usr/bin/env python3
from pathlib import Path
import argparse, csv, hashlib, json, subprocess
from collections import defaultdict, Counter

EXPECTED_G4_COMMIT = "1f60787"
EXPECTED_PROTOCOL_SHA256 = "c6da54d84c361a86a4944326f0a193c8dbcf37a6eb77204de742eb8f15a22f31"
MIB = 1024 * 1024

def require(c, m):
    if not c:
        raise RuntimeError(m)

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(4*MIB), b""):
            h.update(chunk)
    return h.hexdigest()

def git(repo, *args):
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()

def suggestion(rel, size):
    p = rel.replace("\\","/")
    lp = p.lower()
    if p.startswith(("manifests/event_split/","manifests/buildings/","manifests/wrong_pre/",
                     "manifests/router/","results/development/","results/heldout_events/",
                     "vendor/frozen_phase7d_source/src/")):
        return "CANONICAL_VERSION_CANDIDATE", "Frozen Phase7C–7D evidence/source family."
    if p.startswith(("checkpoints/","predictions/")):
        return "RELEASE_LFS_OR_EXTERNAL_CANDIDATE", "Canonical model/data artifact; preserve identity and decide LFS vs release asset."
    if p in {
        "audit/g3_clean_parity_carrier_96.csv",
        "audit/phase7dd_inference_extract.txt",
        "audit/phase7dd_source_resolver_extract.txt",
        "scripts/g3_checkpoint_parity_t4.synthetic96.pre_replay.py",
        "scripts/patch_g3_runner_for_canonical_replay.py",
    } or p.startswith("dist/") or lp.endswith(".bin"):
        return "HISTORICAL_PROVENANCE_CANDIDATE", "Execution/debug/distribution provenance."
    if p in {
        "scripts/restore_canonical_artifacts_v2.py",
        "scripts/verify_restored_evidence_v2.py",
        "audit/restoration_report_v2.json",
    }:
        return "REVIEW_MANUALLY", "Important restoration/verification artifact; decide canonical tooling vs provenance."
    if p == "results/reproduction_report.json":
        return "GENERATED_OR_TEMP_CANDIDATE", "Generated report; decide versioned expected output vs regenerated release output."
    if any(x in lp for x in ("__pycache__/","/.pytest_cache/","/.ipynb_checkpoints/")) or lp.endswith((".pyc",".pyo",".tmp",".bak")):
        return "GENERATED_OR_TEMP_CANDIDATE", "Cache/editor/generated temporary file."
    if size >= 100*MIB:
        return "RELEASE_LFS_OR_EXTERNAL_CANDIDATE", "Unclassified file >=100 MiB."
    if size >= 50*MIB:
        return "REVIEW_MANUALLY", "Unclassified file >=50 MiB."
    return "REVIEW_MANUALLY", "No safe automatic disposition rule."

def rec(repo, rel, tracked):
    p = repo / rel
    require(p.is_file(), f"Missing file: {rel}")
    size = p.stat().st_size
    disp, reason = suggestion(rel, size)
    return {
        "path": rel.replace("\\","/"), "tracked": tracked, "bytes": size,
        "mib": round(size/MIB,6), "sha256": sha256_file(p),
        "extension": "".join(p.suffixes).lower(),
        "top_level": rel.replace("\\","/").split("/",1)[0],
        "suggested_disposition": "TRACKED_EXISTING" if tracked else disp,
        "suggestion_reason": "Already tracked in Git." if tracked else reason,
        "flag_ge_50_mib": size >= 50*MIB, "flag_ge_100_mib": size >= 100*MIB
    }

def write_csv(path, rows):
    fields = ["path","tracked","bytes","mib","sha256","extension","top_level",
              "suggested_disposition","suggestion_reason","flag_ge_50_mib","flag_ge_100_mib"]
    with open(path,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, default=Path("."))
    repo=ap.parse_args().repo_root.resolve()
    require((repo/".git").exists(), "Not a Git repository root.")

    protocol=repo/"audit/g5a_repository_inventory_protocol.json"
    require(protocol.is_file(), "Missing frozen G5A protocol.")
    require(sha256_file(protocol)==EXPECTED_PROTOCOL_SHA256, "G5A protocol SHA mismatch.")

    head=git(repo,"rev-parse","HEAD")
    anc=subprocess.run(["git","merge-base","--is-ancestor",EXPECTED_G4_COMMIT,head],cwd=repo).returncode
    require(anc==0, "Required successful G4 commit is not an ancestor of HEAD.")
    require(subprocess.run(["git","diff","--quiet"],cwd=repo).returncode==0, "Tracked unstaged changes exist.")
    require(subprocess.run(["git","diff","--cached","--quiet"],cwd=repo).returncode==0, "Staged changes exist.")

    untracked=[x for x in subprocess.check_output(
        ["git","ls-files","--others","--exclude-standard","-z"],cwd=repo
    ).decode().split("\0") if x]
    tracked=[x for x in subprocess.check_output(
        ["git","ls-files","-z"],cwd=repo
    ).decode().split("\0") if x]

    print("=== G5A REPOSITORY INVENTORY ===")
    print("HEAD:",head)
    print("Protocol SHA256:",EXPECTED_PROTOCOL_SHA256)
    print("Tracked files:",len(tracked))
    print("Untracked files:",len(untracked))

    ur=[rec(repo,p,False) for p in sorted(untracked)]
    tr=[rec(repo,p,True) for p in sorted(tracked)]

    out=repo/"audit/g5_repository_inventory"; out.mkdir(parents=True,exist_ok=True)
    ucsv=out/"g5a_untracked_inventory.csv"
    tcsv=out/"g5a_tracked_inventory.csv"
    djson=out/"g5a_duplicate_hash_groups.json"
    sjson=out/"g5a_repository_inventory_summary.json"
    write_csv(ucsv,ur); write_csv(tcsv,tr)

    by=defaultdict(list)
    for r in ur: by[r["sha256"]].append(r["path"])
    dups=[{"sha256":h,"paths":sorted(ps),"count":len(ps)} for h,ps in by.items() if len(ps)>1]
    dups.sort(key=lambda x:(-x["count"],x["sha256"]))
    djson.write_text(json.dumps(dups,indent=2,sort_keys=True)+"\n",encoding="utf-8")

    disp=Counter(r["suggested_disposition"] for r in ur)
    tops=Counter(r["top_level"] for r in ur)
    total=sum(r["bytes"] for r in ur)
    largest=sorted(ur,key=lambda r:(-r["bytes"],r["path"]))[:30]
    ge50=[r for r in ur if r["flag_ge_50_mib"]]
    ge100=[r for r in ur if r["flag_ge_100_mib"]]
    summary={
        "stage":"G5A_repository_inventory_and_classification","status":"PASS","head":head,
        "required_g4_success_commit":EXPECTED_G4_COMMIT,"protocol_sha256":EXPECTED_PROTOCOL_SHA256,
        "tracked_file_count":len(tr),"untracked_file_count":len(ur),
        "untracked_total_bytes":total,"untracked_total_mib":total/MIB,
        "untracked_top_level_counts":dict(sorted(tops.items())),
        "suggested_disposition_counts":dict(sorted(disp.items())),
        "files_ge_50_mib":[{"path":r["path"],"bytes":r["bytes"],"sha256":r["sha256"]} for r in ge50],
        "files_ge_100_mib":[{"path":r["path"],"bytes":r["bytes"],"sha256":r["sha256"]} for r in ge100],
        "largest_untracked_files":[{"path":r["path"],"mib":r["mib"],"sha256":r["sha256"],
                                    "suggested_disposition":r["suggested_disposition"]} for r in largest],
        "duplicate_sha256_group_count":len(dups),
        "scientific_files_modified":False,"files_deleted":False,"files_moved":False,
        "gitignore_modified":False,"git_lfs_modified":False,
        "next_gate":"G5A manual disposition review and release inclusion/exclusion manifest"
    }
    sjson.write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n",encoding="utf-8")

    print("\nDisposition suggestions:")
    for k,v in sorted(disp.items()): print(" ",k,":",v)
    print("\nTop-level untracked counts:")
    for k,v in sorted(tops.items()): print(" ",k,":",v)
    print("\nUntracked total MiB:",f"{total/MIB:.3f}")
    print("Files >=50 MiB:",len(ge50)); print("Files >=100 MiB:",len(ge100))
    print("Duplicate untracked SHA256 groups:",len(dups))
    print("\nLargest untracked files:")
    for r in largest[:15]:
        print(f"  {r['mib']:10.3f} MiB  {r['suggested_disposition']:35s}  {r['path']}")
    print("\nG5A INVENTORY: PASS")
    print("SUMMARY:",sjson)
    print("UNTRACKED INVENTORY:",ucsv)
    print("TRACKED INVENTORY:",tcsv)
    print("DUPLICATE HASH GROUPS:",djson)
    print("SUMMARY SHA256:",sha256_file(sjson))

if __name__=="__main__":
    main()
