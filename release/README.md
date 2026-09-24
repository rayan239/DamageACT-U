# DamageACT-U external release assets

The Git repository intentionally keeps large frozen assets outside ordinary Git.

Expected external assets: **12**

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
