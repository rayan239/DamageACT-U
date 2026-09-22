# Execution-only amendments


1. Monolithic Phase7D-B run exceeded Kaggle 12-hour limit.
   Fix: segmented/resumable execution.
   Scientific hyperparameters changed: no.


2. Kaggle sometimes expanded CSV.GZ into CSV.
   Fix: distinguish original compressed-file identity from exact decompressed logical identity.
   Scientific decisions changed: no.


3. Source files appeared under different Kaggle mount layouts.
   Fix: robust source resolver plus exact source SHA verification.
   Scientific decisions changed: no.


4. Raw gzip SHA and logical CSV SHA were initially conflated in a downstream resolver.
   Fix: explicit raw/logical/semantic identity terminology.
   Scientific decisions changed: no.


5. CSV parsing could change redundant patch_id rendering while canonical pair_key retained original identity.
   Fix: recover canonical patch_id from already-verified pair_key and re-run structural checks.
   Scientific decisions changed: no.


These records are provenance, not alternate paper protocols.
