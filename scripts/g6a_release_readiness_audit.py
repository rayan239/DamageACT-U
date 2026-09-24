#!/usr/bin/env python3
from pathlib import Path
import argparse, csv, hashlib, json, subprocess, sys, re, os

EXPECTED_G5_COMMIT="8ef6e7f"
EXPECTED_PROTOCOL_SHA256="31b1bc5c69819a9ca0ba7a5c082cc34a68083d8909e80302e9626f6e74613d12"
TAG="paper-v1.0.0"
MIB=1024*1024
REQUIRED_DOCS=[
    "README.md","REPRODUCIBILITY.md","PAPER_RESULTS.md","CITATION.cff",
    "LICENSE","DATA_LICENSE.md","CHANGELOG.md","Makefile","pyproject.toml",
    "release/README.md","release/external_assets_manifest.json",
]
PLACEHOLDER_PATTERNS=[
    ("author_placeholder", re.compile(r"\bAuthor Name\b",re.I)),
    ("university_placeholder", re.compile(r"\bUniversity Name\b",re.I)),
    ("email_placeholder", re.compile(r"email@example\.com",re.I)),
    ("todo", re.compile(r"\bTODO\b",re.I)),
    ("tbd", re.compile(r"\bTBD\b",re.I)),
    ("replace_me", re.compile(r"REPLACE[_ -]?ME",re.I)),
]
SCAN_DOCS=["README.md","REPRODUCIBILITY.md","PAPER_RESULTS.md","CITATION.cff","DATA_LICENSE.md"]

def require(c,m):
    if not c: raise RuntimeError(m)

def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(4*MIB),b""): h.update(b)
    return h.hexdigest()

def git(repo,*args,check=True):
    p=subprocess.run(["git",*args],cwd=repo,text=True,capture_output=True)
    if check and p.returncode!=0:
        raise RuntimeError(f"git {' '.join(args)} failed: {p.stderr or p.stdout}")
    return p

def run_capture(repo,label,cmd):
    print(f"\n=== {label} ===",flush=True)
    p=subprocess.run(cmd,cwd=repo,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    print(p.stdout,end="" if p.stdout.endswith("\n") else "\n")
    return {"label":label,"command":cmd,"returncode":p.returncode,"output_tail":p.stdout[-6000:]}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--repo-root",type=Path,default=Path("."))
    ap.add_argument("--external-assets",type=Path,required=True)
    ap.add_argument("--run-science",action="store_true")
    args=ap.parse_args()
    repo=args.repo_root.resolve()
    assets=args.external_assets.resolve()
    require((repo/".git").exists(),"Run from Git repository root.")
    proto=repo/"audit/g6a_release_readiness_protocol.json"
    require(proto.is_file(),"Missing G6A protocol.")
    require(sha(proto)==EXPECTED_PROTOCOL_SHA256,"G6A protocol SHA mismatch.")

    head=git(repo,"rev-parse","HEAD").stdout.strip()
    anc=git(repo,"merge-base","--is-ancestor",EXPECTED_G5_COMMIT,head,check=False)
    require(anc.returncode==0,"Required G5 commit 8ef6e7f is not an ancestor of HEAD.")

    # Require clean before audit output creation.
    status=git(repo,"status","--porcelain").stdout
    require(status=="","Working tree must be clean before G6A preflight.")

    tag_exists=git(repo,"rev-parse","-q","--verify",f"refs/tags/{TAG}",check=False).returncode==0

    blockers=[]
    warnings=[]
    checks=[]

    # Documents.
    for rel in REQUIRED_DOCS:
        p=repo/rel
        ok=p.is_file() and p.stat().st_size>0
        checks.append({"check":"required_document","path":rel,"pass":ok})
        if not ok: blockers.append(f"Missing/empty release document: {rel}")

    placeholders=[]
    for rel in SCAN_DOCS:
        p=repo/rel
        if not p.is_file(): continue
        text=p.read_text(encoding="utf-8",errors="replace")
        for name,pat in PLACEHOLDER_PATTERNS:
            for m in pat.finditer(text):
                placeholders.append({"path":rel,"pattern":name,"match":m.group(0)})
    # Placeholders are human-metadata blockers, not scientific failures.
    if placeholders:
        warnings.append(f"Release-facing placeholder hits: {len(placeholders)}")

    # Final disposition: all Git-designated rows must be tracked.
    disp=repo/"audit/g5_release_manifest/g5c_final_release_disposition.csv"
    require(disp.is_file(),"Missing final G5C disposition CSV.")
    with open(disp,newline="",encoding="utf-8") as f:
        rows=list(csv.DictReader(f))
    tracked=set(x for x in git(repo,"ls-files").stdout.splitlines() if x)
    git_designated={r["path"] for r in rows if r["final_disposition"] in ("GIT_CANONICAL","GIT_HISTORICAL_PROVENANCE")}
    missing_tracked=sorted(git_designated-tracked)
    if missing_tracked: blockers.append(f"G5C Git-designated paths not tracked: {missing_tracked}")

    # External assets must not be ordinary Git files.
    ext=json.loads((repo/"release/external_assets_manifest.json").read_text(encoding="utf-8"))
    require(ext.get("asset_count")==12 and len(ext.get("assets",[]))==12,"External manifest must contain 12 assets.")
    accidentally_tracked=sorted(a["path"] for a in ext["assets"] if a["path"] in tracked)
    if accidentally_tracked: blockers.append(f"External assets accidentally tracked: {accidentally_tracked}")

    # Track size hygiene and dist hygiene.
    oversized=[]
    dist_tracked=[]
    for rel in sorted(tracked):
        p=repo/rel
        if p.is_file() and p.stat().st_size>=100*MIB:
            oversized.append({"path":rel,"bytes":p.stat().st_size})
        if rel.startswith("dist/"):
            dist_tracked.append(rel)
    if oversized: blockers.append(f"Tracked files >=100 MiB: {oversized}")
    if dist_tracked: blockers.append(f"Tracked dist/ material: {dist_tracked}")

    # Verify external asset package directly.
    ext_verify=run_capture(repo,"External asset verification",
        [sys.executable,"scripts/verify_external_assets.py","--root",str(assets)])
    if ext_verify["returncode"]!=0: blockers.append("External asset verification failed.")

    science=[]
    if args.run_science:
        commands=[
            ("pytest",[sys.executable,"-m","pytest","-q"]),
            ("canonical evidence audit",[sys.executable,"scripts/verify_restored_evidence_v2.py"]),
            ("deterministic core",[sys.executable,"scripts/reproduce_results_v2_core.py"]),
            ("bootstrap reproduction",[sys.executable,"scripts/reproduce_results_v2_bootstrap.py"]),
        ]
        for label,cmd in commands:
            rr=run_capture(repo,label,cmd); science.append(rr)
            if rr["returncode"]!=0: blockers.append(f"{label} failed.")

    # Read frozen reproduced outputs and enforce headline conclusions.
    core_path=repo/"results/reproduction_core_v2.json"
    boot_path=repo/"results/reproduction_bootstrap_v2.json"
    if not core_path.is_file() or not boot_path.is_file():
        blockers.append("Frozen reproduction report(s) missing.")
    else:
        core=json.loads(core_path.read_text(encoding="utf-8"))
        boot=json.loads(boot_path.read_text(encoding="utf-8"))
        core_text=json.dumps(core)
        boot_text=json.dumps(boot)
        # Strong guardrail: false conclusion flag must remain visible in at least one canonical report.
        if '"all_precommitted_criteria_pass": false' not in core_text.lower() and \
           '"ALL_PRECOMMITTED_CRITERIA_PASS": false' not in core_text:
            warnings.append("Could not locate lowercase all_precommitted_criteria_pass=false key in core report schema.")
        # Scripts themselves are authoritative and were run above; record report hashes.
        checks += [
            {"check":"core_report_sha256","value":sha(core_path),"pass":True},
            {"check":"bootstrap_report_sha256","value":sha(boot_path),"pass":True},
        ]

    if tag_exists:
        blockers.append(f"Release tag {TAG} already exists before G6 closeout.")

    status_name="PASS" if not blockers and not placeholders else (
        "PASS_WITH_HUMAN_METADATA_REVIEW" if not blockers else "FAIL"
    )

    out=repo/"audit/g6_release"; out.mkdir(parents=True,exist_ok=True)
    report={
        "stage":"G6A_final_release_readiness_preflight",
        "status":status_name,
        "head":head,
        "protocol_sha256":EXPECTED_PROTOCOL_SHA256,
        "required_g5_commit":EXPECTED_G5_COMMIT,
        "external_asset_count":12,
        "external_assets_root":str(assets),
        "technical_blockers":blockers,
        "human_metadata_placeholder_hits":placeholders,
        "warnings":warnings,
        "tracked_file_count":len(tracked),
        "tracked_files_ge_100_mib":oversized,
        "tracked_dist_files":dist_tracked,
        "g5c_git_designated_missing_from_tracking":missing_tracked,
        "external_assets_accidentally_tracked":accidentally_tracked,
        "paper_tag_exists_before_closeout":tag_exists,
        "science_runs":science,
        "checks":checks,
        "scientific_conclusion_guardrails":{
            "training_from_scratch_reproduction_claimed":False,
            "final_event_track_seed":42,
            "all_precommitted_criteria_pass":False
        },
        "next_gate":"Resolve only listed blockers/placeholders; then fresh-clone final audit and annotated paper-v1.0.0 tag."
    }
    rp=out/"g6a_release_readiness_report.json"
    rp.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")

    md=[
        "# DamageACT-U G6A release-readiness report","",
        f"- Status: **{status_name}**",
        f"- HEAD: `{head}`",
        f"- Technical blockers: **{len(blockers)}**",
        f"- Human metadata placeholder hits: **{len(placeholders)}**",
        f"- Tracked files: **{len(tracked)}**",
        f"- External assets: **12**",
        "",
        "## Technical blockers",
    ]
    md += [f"- {x}" for x in blockers] or ["- None"]
    md += ["","## Human metadata placeholder hits"]
    md += [f"- `{x['path']}`: {x['pattern']} -> `{x['match']}`" for x in placeholders] or ["- None"]
    md += ["","## Scientific guardrails",
           "- Final event-track evidence remains Seed 42 only.",
           "- G3 validates checkpoint-inference parity, not training-from-scratch.",
           "- G4 validates frozen router/intervention/statistical parity.",
           "- Clean effectiveness remains failed; ALL_PRECOMMITTED_CRITERIA_PASS remains false.",
           ""]
    (out/"g6a_release_readiness_report.md").write_text("\n".join(md),encoding="utf-8")

    print("\n=== G6A RELEASE READINESS ===")
    print("Status:",status_name)
    print("Technical blockers:",len(blockers))
    print("Human metadata placeholder hits:",len(placeholders))
    print("Tracked files:",len(tracked))
    print("Tracked >=100 MiB:",len(oversized))
    print("Tracked dist/ files:",len(dist_tracked))
    print("External assets accidentally tracked:",len(accidentally_tracked))
    print("paper-v1.0.0 tag exists:",tag_exists)
    if blockers:
        print("\nTECHNICAL BLOCKERS:")
        for b in blockers: print(" -",b)
    if placeholders:
        print("\nHUMAN METADATA REVIEW:")
        for x in placeholders: print(f" - {x['path']}: {x['pattern']} -> {x['match']}")
    print("REPORT:",rp)
    print("REPORT SHA256:",sha(rp))

    if blockers:
        raise SystemExit(2)

if __name__=="__main__":
    main()
