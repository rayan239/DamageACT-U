.PHONY: verify test reproduce-results artifact-manifest


verify:
        python scripts/00_verify_environment.py


test:
        pytest -q


reproduce-results:
        python scripts/reproduce_results.py


artifact-manifest:
        python scripts/build_artifact_manifest.py
