#!/usr/bin/env python3
from pathlib import Path
import argparse, json, subprocess, hashlib

TAG="paper-v1.0.0"
EXPECTED_PROTOCOL_SHA256="0ea2b3c216c3b13e3cf09ff9ea8dc4dccdf1a26342b6774ad05cb87725220ec7"
MIB=1024*1024

def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(4*MIB),b""): h.update(b)
    return h.hexdigest()

def run(cmd,cwd,check=True):
    p=subprocess.run(cmd,cwd=cwd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    if p.stdout: print(p.stdout,end="" if p.stdout.endswith("\n") else "\n")
    if check and p.returncode!=0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    return p

ap=argparse.ArgumentParser(description="Create DamageACT-U paper-v1.0.0 tag only after a PASS G6B attestation.")
ap.add_argument("--attestation",type=Path,required=True)
ap.add_argument("--repo-root",type=Path,default=Path("."))
args=ap.parse_args()
repo=args.repo_root.resolve()
att=args.attestation.resolve()

if not (repo/".git").exists(): raise RuntimeError("Not a Git repository.")
proto=repo/"audit/g6b_fresh_clone_release_protocol.json"
if not proto.is_file() or sha(proto)!=EXPECTED_PROTOCOL_SHA256:
    raise RuntimeError("Frozen G6B protocol identity mismatch.")
if not att.is_file(): raise RuntimeError("Attestation file missing.")

data=json.loads(att.read_text(encoding="utf-8"))
if data.get("status")!="PASS": raise RuntimeError("G6B attestation is not PASS.")
head=run(["git","rev-parse","HEAD"],repo).stdout.strip()
if data.get("candidate_commit")!=head:
    raise RuntimeError(f"Attested commit {data.get('candidate_commit')} != current HEAD {head}.")
if run(["git","status","--porcelain"],repo).stdout!="":
    raise RuntimeError("Repository is not clean; refusing to tag.")
if run(["git","rev-parse","-q","--verify",f"refs/tags/{TAG}"],repo,check=False).returncode==0:
    raise RuntimeError(f"Tag {TAG} already exists.")

message=(
    "DamageACT-U paper release v1.0.0\n\n"
    "Fresh-clone reproduction: PASS\n"
    "External assets: 12/12 SHA256-verified\n"
    "Final event-track evidence: Seed 42 only\n"
    "Training-from-scratch reproduction: not claimed\n"
    "Clean-effectiveness criterion: FAIL (preserved)\n"
    "ALL_PRECOMMITTED_CRITERIA_PASS: False (preserved)\n"
    f"G6B attestation SHA256: {sha(att)}"
)
run(["git","tag","-a",TAG,"-m",message],repo)
tag_commit=run(["git","rev-list","-n","1",TAG],repo).stdout.strip()
if tag_commit!=head:
    raise RuntimeError("Created tag does not resolve to current HEAD.")

print("FINAL LOCAL RELEASE TAG: PASS")
print("Tag:",TAG)
print("Commit:",head)
print("Attestation SHA256:",sha(att))
print("No push was performed.")
