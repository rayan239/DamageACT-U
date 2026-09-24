from pathlib import Path
import hashlib
import shutil

RUNNER = Path(r"scripts\g3_checkpoint_parity_t4.py")
BACKUP = Path(r"scripts\g3_checkpoint_parity_t4.synthetic96.pre_replay.py")

OLD_SHA = "6358fea71cf4d28c1cb70a02ac4b416d0f49d1e9f5b1c3f41a9abdf8998b89c1"
REPLAY_LOCK_SHA = "78f6a5eda2aed875f7b65f08879c7ce85987766c9283e10f6ad7fc6a7547bef0"

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def replace_section(text, start_marker, end_marker, new_block):
    a = text.find(start_marker)
    if a < 0:
        raise RuntimeError(f"START MARKER NOT FOUND:\n{start_marker}")
    b = text.find(end_marker, a)
    if b < 0:
        raise RuntimeError(f"END MARKER NOT FOUND:\n{end_marker}")
    if text.find(start_marker, a + 1) >= 0:
        raise RuntimeError(f"START MARKER IS NOT UNIQUE:\n{start_marker}")
    return text[:a] + new_block.rstrip() + "\n\n" + text[b:]


if not RUNNER.is_file():
    raise FileNotFoundError(RUNNER)

got = sha(RUNNER)
print("CURRENT RUNNER SHA256 =", got)

if got != OLD_SHA:
    raise RuntimeError(
        "REFUSING TO PATCH: current runner is not the validated "
        f"synthetic-96 version.\nexpected={OLD_SHA}\ngot={got}"
    )

shutil.copy2(RUNNER, BACKUP)
print("BACKUP =", BACKUP)
print("BACKUP SHA256 =", sha(BACKUP))


text = RUNNER.read_text(encoding="utf-8-sig")


# ------------------------------------------------------------
# A. Add frozen canonical replay lock to required bundle hashes
# ------------------------------------------------------------

old = '''    "audit/g3_parity_execution_protocol.json":
        "112ade4ac863dd7bc7df03a6c2544be6b3757ca0034f864556738851b2d2409c",
'''

new = '''    "audit/g3_parity_execution_protocol.json":
        "112ade4ac863dd7bc7df03a6c2544be6b3757ca0034f864556738851b2d2409c",
    "audit/g3_canonical_batch_replay_lock.json":
        "78f6a5eda2aed875f7b65f08879c7ce85987766c9283e10f6ad7fc6a7547bef0",
'''

if text.count(old) != 1:
    raise RuntimeError("Could not uniquely add canonical replay-lock hash.")

text = text.replace(old, new, 1)


# ------------------------------------------------------------
# B. SHA256SUMS may now contain more than the old 11 entries
# ------------------------------------------------------------

old = '''if len(sums) != 11:
    raise RuntimeError(
        f"Expected 11 SHA256SUMS entries; found {len(sums)}"
    )
'''

new = '''if not isinstance(sums, dict) or len(sums) < 12:
    raise RuntimeError(
        f"Expected at least 12 SHA256SUMS entries; found {len(sums)}"
    )
'''

if text.count(old) != 1:
    raise RuntimeError("Could not uniquely patch SHA256SUMS cardinality gate.")

text = text.replace(old, new, 1)

text = text.replace(
    'print("PASS: all 11 frozen input hashes")',
    'print("PASS: frozen input/source/checkpoint/replay-lock hashes")',
    1,
)


# ------------------------------------------------------------
# C. Require the later frozen runner manifest at execution time
# ------------------------------------------------------------

old = '''print("PASS: internal SHA256SUMS")
'''

new = '''print("PASS: internal SHA256SUMS")

runner_manifest_path = BUNDLE / "audit/g3_parity_runner_manifest.json"

if not runner_manifest_path.is_file():
    raise FileNotFoundError(runner_manifest_path)

runner_manifest = json.loads(
    runner_manifest_path.read_text(encoding="utf-8")
)

if runner_manifest.get("status") != "PRE_INFERENCE_RUNNER_FROZEN_VALIDATED":
    raise RuntimeError(
        "Runner manifest does not have the validated frozen status."
    )

runtime_runner_sha = sha256_file(Path(__file__).resolve())

if runner_manifest.get("runner_sha256") != runtime_runner_sha:
    raise RuntimeError(
        "Runtime runner SHA does not match frozen runner manifest. "
        f"runtime={runtime_runner_sha} "
        f"manifest={runner_manifest.get('runner_sha256')}"
    )

if runner_manifest.get("contains_regenerated_outputs") is not False:
    raise RuntimeError(
        "Runner manifest unexpectedly indicates regenerated outputs."
    )

if runner_manifest.get("runner_has_training_code") is not False:
    raise RuntimeError(
        "Runner manifest training-code flag is not False."
    )

if runner_manifest.get("runner_has_tuning_code") is not False:
    raise RuntimeError(
        "Runner manifest tuning-code flag is not False."
    )

print("PASS: runner self-hash agrees with frozen runner manifest")
'''

if text.count(old) != 1:
    raise RuntimeError("Could not uniquely add runner-manifest self-check.")

text = text.replace(old, new, 1)


# ------------------------------------------------------------
# D. Replace old 77-target-scene staging with canonical replay
# ------------------------------------------------------------

start5 = '''# ============================================================
# 5. Stage exactly the 77 selected scenes
# ============================================================
'''

end5 = '''# ============================================================
# 6. Frozen source imports
# ============================================================
'''

new5 = r'''# ============================================================
# 5. Derive canonical historical batches and stage 180 scenes
# ============================================================

replay_lock_path = (
    BUNDLE /
    "audit/g3_canonical_batch_replay_lock.json"
)

replay_lock = json.loads(
    replay_lock_path.read_text(encoding="utf-8")
)

if replay_lock.get("status") != "PRE_INFERENCE_CANONICAL_BATCH_REPLAY_FROZEN":
    raise RuntimeError(
        "Canonical batch replay lock has unexpected status."
    )

N = len(manifest)

if N != 115349:
    raise RuntimeError(
        f"Canonical TEST manifest rows changed: {N}"
    )

target_rows = selector.manifest_row.astype(int).tolist()

batch_ids = sorted(
    set(
        row // BATCH_SIZE
        for row in target_rows
    )
)

replay_rows = []

for batch_id in batch_ids:

    start = batch_id * BATCH_SIZE

    stop = min(
        (batch_id + 1) * BATCH_SIZE,
        N,
    )

    replay_rows.extend(
        range(start, stop)
    )


def hash_integer_sequence(values):

    payload = (
        "\n".join(
            map(str, values)
        )
        + "\n"
    ).encode("utf-8")

    return hashlib.sha256(
        payload
    ).hexdigest()


if len(batch_ids) != 78:
    raise RuntimeError(
        f"Expected 78 original replay batches; found {len(batch_ids)}"
    )

if len(replay_rows) != 7445:
    raise RuntimeError(
        f"Expected 7445 canonical replay rows; found {len(replay_rows)}"
    )

if len(set(replay_rows)) != 7445:
    raise RuntimeError(
        "Canonical replay rows are not unique."
    )

if not set(target_rows).issubset(
    set(replay_rows)
):
    raise RuntimeError(
        "Canonical replay rows do not contain all frozen targets."
    )


batch_sha = hash_integer_sequence(
    batch_ids
)

row_sha = hash_integer_sequence(
    replay_rows
)

if (
    batch_sha
    !=
    "129c4f7d58e61097a93d57ccdeeb480cdd1e639312d72a0d1270ab5dcb310b9b"
):
    raise RuntimeError(
        "Canonical replay batch-ID SHA mismatch."
    )

if (
    row_sha
    !=
    "22d9894b540e9d3f8f239f94749fba82fe1c56ec553e6029bf784194e9c14196"
):
    raise RuntimeError(
        "Canonical replay row-index SHA mismatch."
    )


lock_identity = replay_lock["identity"]
lock_counts = replay_lock["counts"]

if lock_identity["selected_batch_ids"] != batch_ids:
    raise RuntimeError(
        "Freshly derived replay batch IDs differ from frozen replay lock."
    )

if (
    lock_identity["selected_batch_ids_sha256"]
    != batch_sha
):
    raise RuntimeError(
        "Replay-lock batch SHA mismatch."
    )

if (
    lock_identity["canonical_replay_row_indices_sha256"]
    != row_sha
):
    raise RuntimeError(
        "Replay-lock row SHA mismatch."
    )

if lock_counts["frozen_targets"] != 96:
    raise RuntimeError(
        "Replay-lock target count changed."
    )

if lock_counts["original_batches_replayed"] != 78:
    raise RuntimeError(
        "Replay-lock batch count changed."
    )

if lock_counts["canonical_replay_rows"] != 7445:
    raise RuntimeError(
        "Replay-lock row count changed."
    )

if lock_counts["unique_replay_scenes"] != 180:
    raise RuntimeError(
        "Replay-lock scene count changed."
    )

if lock_counts["final_original_batch_size"] != 53:
    raise RuntimeError(
        "Replay-lock final batch size changed."
    )


replay_df = (
    manifest
    .iloc[replay_rows]
    .reset_index(drop=True)
)


scene_check = (
    replay_df
    .groupby("scene_id")
    .agg(
        n_subset=("source_subset", "nunique"),
        n_pre=("pre_image", "nunique"),
        n_post=("post_image", "nunique"),
    )
)

if not (
    (scene_check.n_subset == 1)
    &
    (scene_check.n_pre == 1)
    &
    (scene_check.n_post == 1)
).all():
    raise RuntimeError(
        "Inconsistent canonical replay scene metadata."
    )


scenes = (
    replay_df[
        [
            "scene_id",
            "source_subset",
            "pre_image",
            "post_image",
        ]
    ]
    .drop_duplicates()
    .reset_index(drop=True)
)

if len(scenes) != 180:
    raise RuntimeError(
        f"Expected 180 canonical replay scenes; found {len(scenes)}"
    )

if replay_df.disaster.astype(str).nunique() != 6:
    raise RuntimeError(
        "Canonical replay population no longer contains all six events."
    )


if STAGE.exists():
    shutil.rmtree(
        STAGE
    )

(STAGE / "images").mkdir(
    parents=True
)

(STAGE / "labels").mkdir(
    parents=True
)


created = set()

for r in scenes.itertuples(index=False):

    src = subset_dir(
        XBD_ROOT,
        r.source_subset,
    )

    stem = str(
        r.scene_id
    )

    required = [
        (
            src / str(r.pre_image),
            STAGE / str(r.pre_image),
        ),
        (
            src / str(r.post_image),
            STAGE / str(r.post_image),
        ),
        (
            src / "labels" /
            f"{stem}_pre_disaster.json",
            STAGE / "labels" /
            f"{stem}_pre_disaster.json",
        ),
        (
            src / "labels" /
            f"{stem}_post_disaster.json",
            STAGE / "labels" /
            f"{stem}_post_disaster.json",
        ),
    ]

    for source, destination in required:

        if not source.is_file():
            raise FileNotFoundError(
                source
            )

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if destination not in created:

            os.symlink(
                source,
                destination,
            )

            created.add(
                destination
            )


if len(created) != 180 * 4:
    raise RuntimeError(
        f"Unexpected staged artifact count: {len(created)}"
    )


print("PASS: canonical batch replay lock verified")
print("PASS: 78 historical batches -> 7445 canonical rows")
print("PASS: 180 canonical replay scenes staged")
'''

text = replace_section(
    text,
    start5,
    end5,
    new5,
)


# ------------------------------------------------------------
# E. Replace 96-target Subset with 7445-row replay Subset
# ------------------------------------------------------------

start7 = '''# ============================================================
# 7. Canonical dataset + 96-row Subset
# ============================================================
'''

end7 = '''# ============================================================
# 8. Frozen expert checkpoints
# ============================================================
'''

new7 = r'''# ============================================================
# 7. Canonical dataset + exact historical-batch replay Subset
# ============================================================

full_ds = PairedXBDDataset(
    manifest_path=manifest_path,
    dataset_root=STAGE,
    crop_config=CROP,
    training=False,
    augment_geometry=False,
    return_metadata=True,
)

replay_ds = Subset(
    full_ds,
    replay_rows,
)

loader = DataLoader(
    replay_ds,
    batch_size=96,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
    drop_last=False,
    persistent_workers=False,
)

if len(loader) != 78:
    raise RuntimeError(
        f"Expected 78 historical replay batches; got {len(loader)}"
    )
'''

text = replace_section(
    text,
    start7,
    end7,
    new7,
)


# ------------------------------------------------------------
# F. Replace synthetic one-batch inference with exact replay
# ------------------------------------------------------------

start9 = '''# ============================================================
# 9. One canonical-size FP32 inference batch
# ============================================================
'''

end9 = '''# ============================================================
# 10. Align sealed Phase7D-D outputs
# ============================================================
'''

new9 = r'''# ============================================================
# 9. Replay exact original FP32 batch contexts
# ============================================================

target_ids = set(
    selector.building_id.astype(str)
)

expected_target_order = (
    selector
    .building_id
    .astype(str)
    .tolist()
)

target_row_by_id = dict(
    zip(
        expected_target_order,
        selector.manifest_row.astype(int).tolist(),
    )
)

rows = []

observed_batch_ids = []
observed_batch_sizes = []


with torch.inference_mode():

    for replay_position, batch in enumerate(loader):

        original_batch_id = batch_ids[
            replay_position
        ]

        original_start = (
            original_batch_id
            *
            BATCH_SIZE
        )

        original_stop = min(
            (original_batch_id + 1)
            *
            BATCH_SIZE,
            len(manifest),
        )

        expected_ids = (
            manifest
            .iloc[
                original_start:
                original_stop
            ]
            .building_id
            .astype(str)
            .tolist()
        )

        got_ids = [
            str(x)
            for x in batch["building_id"]
        ]

        if got_ids != expected_ids:
            raise RuntimeError(
                "Historical batch membership/order mismatch "
                f"for original batch {original_batch_id}"
            )

        observed_batch_ids.append(
            original_batch_id
        )

        observed_batch_sizes.append(
            len(got_ids)
        )


        pre = batch["pre"].to(
            DEVICE,
            non_blocking=True,
        )

        post = batch["post"].to(
            DEVICE,
            non_blocking=True,
        )


        # Canonical final Phase7D-D semantics:
        # direct FP32 forwards, no autocast.
        lp = post_model(
            pre,
            post,
        )

        lt = pair_model(
            pre,
            post,
        )


        pp = torch.softmax(
            lp,
            dim=1,
        ).cpu().numpy()

        pt = torch.softmax(
            lt,
            dim=1,
        ).cpu().numpy()

        lpn = lp.cpu().numpy()
        ltn = lt.cpu().numpy()


        for i, bid in enumerate(
            got_ids
        ):

            if bid not in target_ids:
                continue

            manifest_row = (
                original_start
                +
                i
            )

            if (
                target_row_by_id[bid]
                != manifest_row
            ):
                raise RuntimeError(
                    "Frozen target manifest-row identity mismatch "
                    f"for {bid}"
                )

            rec = {
                "manifest_row":
                    manifest_row,

                "original_batch_id":
                    original_batch_id,

                "position_in_original_batch":
                    i,

                "original_batch_size":
                    len(got_ids),

                "building_id":
                    bid,

                "scene_id":
                    str(batch["scene_id"][i]),

                "disaster":
                    str(batch["disaster"][i]),

                "true_label":
                    int(
                        batch["label"][i].item()
                    ),

                "new_pred_post":
                    int(pp[i].argmax()),

                "new_pred_pair":
                    int(pt[i].argmax()),
            }

            for k in range(4):

                rec[f"new_p{k}_post"] = (
                    float(pp[i, k])
                )

                rec[f"new_p{k}_pair"] = (
                    float(pt[i, k])
                )

                rec[f"new_logit{k}_post"] = (
                    float(lpn[i, k])
                )

                rec[f"new_logit{k}_pair"] = (
                    float(ltn[i, k])
                )

            rows.append(
                rec
            )


if observed_batch_ids != batch_ids:
    raise RuntimeError(
        "Observed historical batch-ID sequence changed."
    )

if (
    observed_batch_sizes[:-1]
    !=
    [96] * 77
):
    raise RuntimeError(
        "One or more of the first 77 replay batches "
        "does not contain exactly 96 rows."
    )

if observed_batch_sizes[-1] != 53:
    raise RuntimeError(
        "Historical final batch was not reproduced at size 53."
    )


new = (
    pd.DataFrame(rows)
    .sort_values(
        "manifest_row",
        kind="mergesort",
    )
    .reset_index(drop=True)
)

if len(new) != 96:
    raise RuntimeError(
        f"Expected 96 frozen target outputs; got {len(new)}"
    )

if not new.building_id.is_unique:
    raise RuntimeError(
        "Regenerated target building IDs are not unique."
    )

if (
    new.building_id.astype(str).tolist()
    !=
    expected_target_order
):
    raise RuntimeError(
        "Regenerated target ordering changed."
    )

print("PASS: exact historical batch membership/order replayed")
print("PASS: 77 x 96 batches + final 53-sample batch preserved")
print("PASS: regenerated exactly the frozen 96 target outputs")
'''

text = replace_section(
    text,
    start9,
    end9,
    new9,
)


# ------------------------------------------------------------
# G. Add replay context to final report
# ------------------------------------------------------------

marker = '''combined.to_csv(
    OUT /
    "g3_regenerated_vs_frozen_96.csv",
    index=False,
)
'''

injection = '''report["frozen_commits"]["canonical_batch_replay_design"] = "847da83"

report["replay_context"] = {
    "original_batches_replayed": len(batch_ids),
    "canonical_replay_rows": len(replay_rows),
    "canonical_replay_scenes": len(scenes),
    "selected_batch_ids_sha256": batch_sha,
    "canonical_replay_row_indices_sha256": row_sha,
    "observed_batch_sizes": observed_batch_sizes,
    "full_96_sample_batches": int(
        sum(x == 96 for x in observed_batch_sizes)
    ),
    "partial_batches": int(
        sum(x < 96 for x in observed_batch_sizes)
    ),
    "final_original_batch_size": int(
        observed_batch_sizes[-1]
    ),
    "targets_in_final_partial_batch": int(
        (new.original_batch_size == 53).sum()
    ),
}

'''

if text.count(marker) != 1:
    raise RuntimeError(
        "Could not uniquely find report-output insertion point."
    )

text = text.replace(
    marker,
    injection + marker,
    1,
)


# ------------------------------------------------------------
# H. Final static invariants before writing
# ------------------------------------------------------------

required_after = [
    "g3_canonical_batch_replay_lock.json",
    "78 historical batches",
    "7445 canonical rows",
    "180 canonical replay scenes",
    "observed_batch_sizes[-1] != 53",
    "exact historical batch membership/order replayed",
    "targets_in_final_partial_batch",
    "runner self-hash agrees with frozen runner manifest",
]

missing = [
    x for x in required_after
    if x not in text
]

if missing:
    raise RuntimeError(
        f"PATCH RESULT MISSING REQUIRED TOKENS: {missing}"
    )

forbidden_after = [
    "# 5. Stage exactly the 77 selected scenes",
    "# 7. Canonical dataset + 96-row Subset",
    "# 9. One canonical-size FP32 inference batch",
    "if len(loader) != 1:",
]

still_present = [
    x for x in forbidden_after
    if x in text
]

if still_present:
    raise RuntimeError(
        f"OBSOLETE SYNTHETIC-BATCH CODE STILL PRESENT: {still_present}"
    )


RUNNER.write_text(
    text,
    encoding="utf-8",
    newline="\n",
)

print()
print("PATCH COMPLETE")
print("NEW BYTES =", RUNNER.stat().st_size)
print(
    "NEW LINES =",
    len(
        RUNNER.read_text(
            encoding="utf-8"
        ).splitlines()
    ),
)
print("NEW SHA256 =", sha(RUNNER))
print("BACKUP REMAINS =", BACKUP)