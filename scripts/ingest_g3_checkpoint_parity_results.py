#!/usr/bin/env python3
"""
DamageACT-U G3 checkpoint-parity evidence ingestion.

Purpose:
- ingest ONLY the already-successful frozen G3 T4 parity evidence;
- verify exact uploaded result identities and scientific status;
- copy small audit artifacts into the repository;
- preserve the executed notebook used for the successful run;
- write a deterministic closeout manifest.

This script does NOT train, tune, regenerate TEST predictions, change tolerances,
or modify the canonical Phase7C/7D evidence.
"""

from pathlib import Path
import argparse
import csv
import hashlib
import io
import json
import shutil
import zipfile

EXPECTED_RESULTS_ZIP_SHA256 = "9a83b451cb810ac9b0d4a8f750b5d22b7d7a75950706ca7598146270ad72a312"
EXPECTED_RESULTS_ZIP_BYTES = 28469
EXPECTED_EXECUTED_NOTEBOOK_SHA256 = "d0b270bb545008979f71bac10cdd470b20d220964f7e0285b3e41830215c3205"

EXPECTED_RUNNER_SHA256 = "cda8f058123ecbe9f63e65ded32fc8ba8161654e07a0e769980f3ecb9fdc0ea9"
EXPECTED_FILES = {
    "g3_regenerated_vs_frozen_96.csv": "13f649ae137dde9cf32decfe4933366f7ba667c6390bedd5e48306f7c6d8ef9b",
    "g3_environment.json": "680183efa30b8c76b44af4893bf638ab0b4a7c7ff09294686ddfbd9c16a2d598",
    "g3_checkpoint_parity_report.json": "b043a27728f6b4aa018ec317056aa0a2a4df8b3ef6643073c1d4e3ed36238be8",
}

EXPECTED_ENV = {
    "torch": "2.10.0+cu128",
    "torchvision": "0.25.0+cu128",
    "cuda": "12.8",
    "gpu": "Tesla T4",
    "pillow": "11.3.0",
}

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def require(cond, message):
    if not cond:
        raise RuntimeError(message)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-zip", required=True, type=Path)
    ap.add_argument("--executed-notebook", required=True, type=Path)
    ap.add_argument("--repo-root", type=Path, default=Path("."))
    args = ap.parse_args()

    repo = args.repo_root.resolve()
    results_zip = args.results_zip.resolve()
    executed_nb = args.executed_notebook.resolve()

    require((repo / ".git").exists(), f"Not a Git repository root: {repo}")
    require(results_zip.is_file(), f"Missing results ZIP: {results_zip}")
    require(executed_nb.is_file(), f"Missing executed notebook: {executed_nb}")

    print("=== G3 CLOSEOUT: VERIFY EXTERNAL EVIDENCE ===")
    require(results_zip.stat().st_size == EXPECTED_RESULTS_ZIP_BYTES,
            f"Results ZIP size mismatch: {results_zip.stat().st_size}")
    got_zip_sha = sha256_file(results_zip)
    require(got_zip_sha == EXPECTED_RESULTS_ZIP_SHA256,
            f"Results ZIP SHA mismatch: {got_zip_sha}")
    print("PASS: result ZIP byte identity")

    got_nb_sha = sha256_file(executed_nb)
    require(got_nb_sha == EXPECTED_EXECUTED_NOTEBOOK_SHA256,
            f"Executed notebook SHA mismatch: {got_nb_sha}")
    print("PASS: executed notebook byte identity")

    with zipfile.ZipFile(results_zip, "r") as z:
        require(z.testzip() is None, "Results ZIP CRC failure")
        names = z.namelist()
        require(sorted(names) == sorted(EXPECTED_FILES),
                f"Unexpected result ZIP members: {names}")

        payloads = {name: z.read(name) for name in names}

    for name, expected_sha in EXPECTED_FILES.items():
        got = sha256_bytes(payloads[name])
        require(got == expected_sha, f"{name} SHA mismatch: {got}")
        print("PASS:", name, got)

    env = json.loads(payloads["g3_environment.json"])
    report = json.loads(payloads["g3_checkpoint_parity_report.json"])

    require(env["torch"] == EXPECTED_ENV["torch"], "Torch version mismatch")
    require(env["torchvision"] == EXPECTED_ENV["torchvision"], "Torchvision version mismatch")
    require(env["cuda"] == EXPECTED_ENV["cuda"], "CUDA version mismatch")
    require(env["gpu"] == EXPECTED_ENV["gpu"], "GPU mismatch")
    require(env["pillow"] == EXPECTED_ENV["pillow"], "Pillow version mismatch")
    require(str(env["python"]).startswith("3.12.13 "), "Python version mismatch")
    print("PASS: successful T4 environment identity")

    require(report.get("stage") == "G3_checkpoint_inference_parity", "Wrong stage")
    require(report.get("status") == "PASS", "G3 report status is not PASS")
    require(report.get("strict_parity_pass") is True, "strict_parity_pass is not true")
    require(report.get("runner_sha256") == EXPECTED_RUNNER_SHA256, "Frozen runner SHA mismatch")

    counts = report["counts"]
    require(counts == {
        "manifest_rows": 115349,
        "selector_rows": 96,
        "anchors": 23,
        "selected_scenes": 180,
    }, f"Unexpected G3 counts: {counts}")

    rule = report["predeclared_parity_rule"]
    require(rule == {
        "atol": 1e-5,
        "rtol": 1e-5,
        "exact_argmax_required": True,
    }, f"Frozen parity rule changed: {rule}")

    cat = report["categorical"]
    require(cat["post_matches"] == cat["post_total"] == 96, "POST is not 96/96")
    require(cat["siamese_matches"] == cat["siamese_total"] == 96, "Siamese is not 96/96")
    require(cat["anchor_post_matches"] == 23, "Anchor POST mismatch")
    require(cat["anchor_siamese_matches"] == 23, "Anchor Siamese mismatch")
    require(cat["all_predictions_match"] is True, "Prediction parity flag false")

    for section_name in ("numerical", "anchor_numerical"):
        section = report[section_name]
        require(set(section) == {
            "post_logits", "post_probabilities",
            "siamese_logits", "siamese_probabilities",
        }, f"Unexpected {section_name} keys")
        for name, rec in section.items():
            require(rec["allclose"] is True, f"{section_name}/{name} allclose false")
            for k in ("max_absolute_difference", "mean_absolute_difference", "median_absolute_difference"):
                require(float(rec[k]) >= 0.0, f"Negative numerical difference at {section_name}/{name}/{k}")

    replay = report["replay_context"]
    require(replay["original_batches_replayed"] == 78, "Historical batch count changed")
    require(replay["canonical_replay_rows"] == 7445, "Canonical replay row count changed")
    require(replay["canonical_replay_scenes"] == 180, "Canonical replay scene count changed")
    require(replay["full_96_sample_batches"] == 77, "Full batch count changed")
    require(replay["partial_batches"] == 1, "Partial batch count changed")
    require(replay["final_original_batch_size"] == 53, "Final batch size changed")
    require(replay["targets_in_final_partial_batch"] == 1, "Final partial target count changed")

    csv_text = payloads["g3_regenerated_vs_frozen_96.csv"].decode("utf-8")
    rows = list(csv.DictReader(io.StringIO(csv_text)))
    require(len(rows) == 96, f"Comparison CSV should have 96 rows, found {len(rows)}")
    require(len({r["building_id"] for r in rows}) == 96, "Comparison building IDs are not unique")
    require(all(r["new_pred_post"] == r["frozen_pred_post"] for r in rows),
            "At least one row has POST argmax mismatch")
    require(all(r["new_pred_pair"] == r["frozen_pred_pair"] for r in rows),
            "At least one row has Siamese argmax mismatch")
    print("PASS: 96-row regenerated-vs-frozen table identity")

    audit_dir = repo / "audit" / "g3_checkpoint_parity"
    nb_dir = repo / "notebooks" / "historical" / "execution_fixes"
    audit_dir.mkdir(parents=True, exist_ok=True)
    nb_dir.mkdir(parents=True, exist_ok=True)

    # Refuse silent overwrite with different bytes.
    for name, data in payloads.items():
        dst = audit_dir / name
        if dst.exists():
            require(dst.read_bytes() == data, f"Refusing to overwrite changed file: {dst}")
        else:
            dst.write_bytes(data)

    canonical_nb_name = "DamageACTU_G3_T4_Parity_FINAL_DEVIL_AUDITED_EXECUTED.ipynb"
    nb_dst = nb_dir / canonical_nb_name
    if nb_dst.exists():
        require(sha256_file(nb_dst) == EXPECTED_EXECUTED_NOTEBOOK_SHA256,
                f"Refusing to overwrite changed notebook: {nb_dst}")
    else:
        shutil.copy2(executed_nb, nb_dst)

    manifest = {
        "stage": "G3_checkpoint_inference_parity_closeout",
        "status": "FROZEN_SUCCESSFUL_EVIDENCE",
        "scientific_scope": (
            "Checkpoint-inference parity only. No training-from-scratch reproduction "
            "is claimed by G3."
        ),
        "source_result_zip": {
            "sha256": EXPECTED_RESULTS_ZIP_SHA256,
            "bytes": EXPECTED_RESULTS_ZIP_BYTES,
        },
        "executed_notebook": {
            "repository_path": f"notebooks/historical/execution_fixes/{canonical_nb_name}",
            "sha256": EXPECTED_EXECUTED_NOTEBOOK_SHA256,
        },
        "payloads": EXPECTED_FILES,
        "frozen_runner_sha256": EXPECTED_RUNNER_SHA256,
        "environment": env,
        "parity": {
            "post": "96/96",
            "siamese": "96/96",
            "anchors_post": "23/23",
            "anchors_siamese": "23/23",
            "strict_parity_pass": True,
            "atol": 1e-5,
            "rtol": 1e-5,
            "exact_argmax_required": True,
        },
        "replay": {
            "historical_batches": 78,
            "canonical_rows": 7445,
            "canonical_scenes": 180,
            "full_batches": 77,
            "final_partial_batch_size": 53,
        },
        "limitations": report["limitations"],
        "next_gate": (
            "G4 router/intervention parity: validate wrong-PRE construction, "
            "26-feature routing schema, and frozen router behavior without retraining."
        ),
        "prohibitions": [
            "Do not alter frozen G3 tolerances.",
            "Do not replace or regenerate canonical Phase7C/7D evidence in this closeout.",
            "Do not infer training-from-scratch reproducibility from checkpoint parity.",
            "Do not claim random-seed robustness; final event-track evidence is Seed 42 only.",
            "Do not tune against held-out TEST after this point.",
        ],
    }

    manifest_path = audit_dir / "g3_final_closeout_manifest.json"
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if manifest_path.exists():
        require(manifest_path.read_bytes() == manifest_bytes,
                f"Existing closeout manifest differs: {manifest_path}")
    else:
        manifest_path.write_bytes(manifest_bytes)

    print("\n=== G3 CLOSEOUT: PASS ===")
    print("Audit directory:", audit_dir)
    print("Executed notebook:", nb_dst)
    print("Closeout manifest SHA256:", sha256_file(manifest_path))
    print("\nNext gate: G4 router/intervention parity.")
    print("Do not begin G4 until the verification commands and Git commit below pass.")

if __name__ == "__main__":
    main()
