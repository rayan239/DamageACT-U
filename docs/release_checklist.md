# Public release checklist


- Fill author names and paper citation.
- Confirm dataset redistribution rules.
- Restore exact canonical source snapshot and large artifacts.
- Run pytest -q.
- Run fast statistical reproduction and verify the paper-number report.
- Generate audit/artifact_manifest.csv and SHA256SUMS.
- Confirm no hard-coded Kaggle username or private path remains.
- Confirm single-seed limitation is visible in README and paper.
- Confirm failed clean primary endpoint is visible in PAPER_RESULTS.md.
- Tag the exact Git commit as paper-v1.0.0.
- Archive the tag/release on Zenodo and cite the DOI.
