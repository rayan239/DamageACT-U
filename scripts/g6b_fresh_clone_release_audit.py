#!/usr/bin/env python3
from pathlib import Path
import argparse, hashlib, json, subprocess, sys, shutil, platform, time

EXPECTED_G6A_PROTOCOL_COMMIT="7697509"
EXPECTED_G6A_REPORT_SHA256="bc68b15417a4b520276d2322f3cb2991b55e6cab10c8729ed6c2e8d82312c6d7"
EXPECTED_PROTOCOL_SHA256="0ea2b3c216c3b13e3cf09ff9ea8dc4dccdf1a26342b6774ad05cb87725220ec7"
TAG="paper-v1.0.0"
MIB=1024*1024

def require(c,m):
    if not c: raise RuntimeError(m)

def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(4*MIB),b""): h.update(b)
    return h.hexdigest()

def run(cmd,cwd,check=True,capture=True):
    p=subprocess.run(cmd,cwd=cwd,text=True,
                     stdout=subprocess.PIPE if capture else None,
                     stderr=subprocess.STDOUT if capture else None)
    if capture and p.stdout:
        print(p.stdout,end="" if p.stdout.endswith("\n") else "\n")
    if check and p.returncode!=0:
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(map(str,cmd))}")
    return p

def git(repo,*args,check=True):
    return run(["git",*args],repo,check=check,capture=True)

def main():
    ap=argparse.ArgumentParser(description="DamageACT-U G6B exact fresh-clone release audit.")
    ap.add_argument("--source-repo",type=Path,default=Path("."))
    ap.add_argument("--external-assets",type=Path,required=True)
    ap.add_argument("--clone-dir",type=Path,required=True)
    ap.add_argument("--attestation-dir",type=Path,required=True)
    args=ap.parse_args()

    source=args.source_repo.resolve()
    assets=args.external_assets.resolve()
    clone=args.clone_dir.resolve()
    attest=args.attestation_dir.resolve()

    require((source/".git").exists(),"Source repository is not a Git repository.")
    proto=source/"audit/g6b_fresh_clone_release_protocol.json"
    require(proto.is_file(),"Missing frozen G6B protocol.")
    require(sha(proto)==EXPECTED_PROTOCOL_SHA256,"G6B protocol SHA mismatch.")

    head=git(source,"rev-parse","HEAD").stdout.strip()
    anc=git(source,"merge-base","--is-ancestor",EXPECTED_G6A_PROTOCOL_COMMIT,head,check=False)
    require(anc.returncode==0,"Required G6A protocol commit is not an ancestor of candidate HEAD.")
    require(git(source,"status","--porcelain").stdout=="","Source repository must be clean.")

    g6a=source/"audit/g6_release/g6a_release_readiness_report.json"
    require(g6a.is_file(),"G6A PASS report is not tracked/present.")
    require(sha(g6a)==EXPECTED_G6A_REPORT_SHA256,"G6A report SHA256 mismatch.")
    g6a_data=json.loads(g6a.read_text(encoding="utf-8"))
    require(g6a_data.get("status")=="PASS","G6A report is not PASS.")
    require(not g6a_data.get("technical_blockers"),"G6A report contains technical blockers.")
    require(len(g6a_data.get("human_metadata_placeholder_hits",[]))==0,
            "G6A report contains metadata placeholder hits.")

    tag_pre=git(source,"rev-parse","-q","--verify",f"refs/tags/{TAG}",check=False)
    require(tag_pre.returncode!=0,f"{TAG} already exists before G6B.")

    require(assets.is_dir(),"External-assets root does not exist.")
    require(not clone.exists(),
            f"Clone directory already exists: {clone}. Refusing to delete/overwrite it.")
    require(not attest.exists(),
            f"Attestation directory already exists: {attest}. Refusing to delete/overwrite it.")
    attest.mkdir(parents=True)

    started=time.time()
    print("=== G6B FRESH-CLONE RELEASE AUDIT ===")
    print("Candidate HEAD:",head)
    print("Source repository:",source)
    print("Clone directory:",clone)
    print("External assets:",assets)
    print("Protocol SHA256:",EXPECTED_PROTOCOL_SHA256)

    print("\n[1/8] Cloning exact candidate with no hardlinks...")
    run(["git","clone","--no-hardlinks",str(source),str(clone)],source.parent)
    git(clone,"checkout","--detach",head)
    clone_head=git(clone,"rev-parse","HEAD").stdout.strip()
    require(clone_head==head,"Fresh clone HEAD mismatch.")
    require(git(clone,"status","--porcelain").stdout=="","Fresh clone is dirty before asset install.")

    print("\n[2/8] Verifying external assets are absent from Git clone...")
    manifest=json.loads((clone/"release/external_assets_manifest.json").read_text(encoding="utf-8"))
    require(manifest.get("asset_count")==12 and len(manifest.get("assets",[]))==12,
            "Fresh-clone external manifest does not contain exactly 12 assets.")
    preexisting=[a["path"] for a in manifest["assets"] if (clone/a["path"]).exists()]
    require(not preexisting,f"External assets unexpectedly present in fresh Git clone: {preexisting}")
    print("Fresh clone contains 0/12 external assets before installation: PASS")

    print("\n[3/8] Installing frozen external assets...")
    run([sys.executable,"scripts/install_external_assets.py",
         "--assets-root",str(assets),"--repo-root",str(clone)],clone)

    print("\n[4/8] Verifying installed external assets...")
    run([sys.executable,"scripts/verify_external_assets.py","--root",str(clone)],clone)

    print("\n[5/8] Running public frozen-result reproduction entry point...")
    repro=run([sys.executable,"scripts/reproduce_release.py"],clone)

    print("\n[6/8] Running Git object/database integrity check...")
    git(clone,"fsck","--full")

    print("\n[7/8] Verifying exact tracked-tree cleanliness after reproduction...")
    post_status=git(clone,"status","--porcelain","--untracked-files=all").stdout
    require(post_status=="",f"Fresh clone became dirty after reproduction:\n{post_status}")
    print("Fresh clone remains Git-clean: PASS")

    print("\n[8/8] Comparing key frozen report identities...")
    key_paths=[
        "audit/g6_release/g6a_release_readiness_report.json",
        "audit/g4_router_parity/g4b_router_intervention_parity_report.json",
        "results/reproduction_core_v2.json",
        "results/reproduction_bootstrap_v2.json",
        "release/external_assets_manifest.json",
    ]
    comparisons=[]
    for rel in key_paths:
        sp=source/rel; cp=clone/rel
        require(sp.is_file() and cp.is_file(),f"Missing key report: {rel}")
        shs,shc=sha(sp),sha(cp)
        require(shs==shc,f"Key report identity changed in fresh clone: {rel}")
        comparisons.append({"path":rel,"sha256":shs})
        print(rel,shs,"PASS")

    core=json.loads((clone/"results/reproduction_core_v2.json").read_text(encoding="utf-8"))
    boot=json.loads((clone/"results/reproduction_bootstrap_v2.json").read_text(encoding="utf-8"))

    elapsed=time.time()-started
    report={
        "stage":"G6B_fresh_clone_final_release",
        "status":"PASS",
        "candidate_commit":head,
        "protocol_sha256":EXPECTED_PROTOCOL_SHA256,
        "g6a_report_sha256":EXPECTED_G6A_REPORT_SHA256,
        "tag_expected":TAG,
        "tag_existed_before_audit":False,
        "clone_no_hardlinks":True,
        "fresh_clone_external_assets_preexisting":0,
        "external_assets_verified":12,
        "public_reproduction_returncode":repro.returncode,
        "git_fsck_pass":True,
        "fresh_clone_clean_after_reproduction":True,
        "key_report_identity_checks":comparisons,
        "python_executable":sys.executable,
        "python_version":platform.python_version(),
        "elapsed_seconds":elapsed,
        "scientific_guardrails":{
            "final_seed":42,
            "training_from_scratch_reproduction_claimed":False,
            "clean_effectiveness_pass":False,
            "all_precommitted_criteria_pass":False
        },
        "next_action":f"Create annotated local tag {TAG} on candidate_commit using g6b_create_release_tag.py."
    }
    rp=attest/"g6b_fresh_clone_release_attestation.json"
    rp.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    md=[
        "# DamageACT-U G6B fresh-clone release attestation","",
        "- Status: **PASS**",
        f"- Candidate commit: `{head}`",
        "- External assets verified: **12/12**",
        "- Public frozen-result reproduction: **PASS**",
        "- Git fsck: **PASS**",
        "- Fresh clone clean after reproduction: **PASS**",
        "- Training-from-scratch reproduction: **not claimed**",
        "- Final event-track execution: **Seed 42 only**",
        "- Clean-effectiveness criterion: **FAIL (preserved)**",
        "- `ALL_PRECOMMITTED_CRITERIA_PASS`: **False (preserved)**",
        ""
    ]
    (attest/"g6b_fresh_clone_release_attestation.md").write_text("\n".join(md),encoding="utf-8")
    (attest/"SHA256SUMS.txt").write_text(
        f"{sha(rp)}  g6b_fresh_clone_release_attestation.json\n"
        f"{sha(attest/'g6b_fresh_clone_release_attestation.md')}  g6b_fresh_clone_release_attestation.md\n",
        encoding="utf-8"
    )

    print("\nG6B FRESH-CLONE RELEASE AUDIT: PASS")
    print("Candidate commit:",head)
    print("ATTESTATION:",rp)
    print("ATTESTATION SHA256:",sha(rp))
    print("Elapsed seconds:",f"{elapsed:.1f}")

if __name__=="__main__":
    main()
