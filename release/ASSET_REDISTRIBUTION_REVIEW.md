# External-asset redistribution review

Status: **HOLD FOR LICENSE CLEARANCE**

The frozen release manifest contains 12 external assets. They are technically verified and sufficient for the public fresh-clone reproduction workflow, but this does not automatically establish the legal right to redistribute every derived file.

## Conservative classification

### Potentially publishable after final review

- expert model checkpoints:
  - `checkpoints/experts/post_seed42_best_state_dict.pt`
  - `checkpoints/experts/siamese_seed42_best_state_dict.pt`

These are trained-model artifacts rather than raw xBD imagery/labels. They should still be checked against the applicable upstream data and pretrained-weight terms before public archival release.

### Hold until xBD-derived-data redistribution is confirmed

- all building manifests
- all development prediction tables
- all held-out prediction tables

Reason: these files contain xBD-derived identifiers, labels, geometry/manifest information and/or ground-truth-linked outputs. Even though they are not raw imagery, they may reproduce or encode source annotations.

### Hold as a package

- `DamageACTU_G3_Parity_Execution_Bundle.bin`

Reason: it packages a mixture of checkpoints, manifests, prediction evidence and frozen execution material, so it inherits the most restrictive unresolved redistribution status of its contents.

## Current release rule

Do not upload the 12-asset package to a public GitHub Release or Zenodo record until the redistribution question is resolved.

The source repository remains public because raw xBD imagery is not included. Users can obtain xBD independently and use the code/manifests workflow subject to the upstream terms.

## Technical identities

The authoritative paths, sizes and SHA256 values remain in:

- `release/external_assets_manifest.json`
- `release/SHA256SUMS.external.txt`

Licensing review must not change those frozen scientific identities.
