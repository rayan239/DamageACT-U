# Canonical artifacts required for exact paper reproduction


Do not synthesize missing canonical checkpoints or predictions.


## Phase7C
phase7d_frozen_unique_patch_roles.csv.gz
raw SHA-256: 7a396d0c61c6a78151a58b2c62dd016786ab1c4f9dd85f05eed5bde430ee4f05
phase7d_build_lock.json
results_index.json


## Phase7D-A
event_train_buildings.csv.gz
raw SHA-256: c82b2743f857a11ed7b390a06b4b2b24e6bebdbd7e925585221c66ba5a60e9d4


event_val_buildings.csv.gz
raw SHA-256: 44adcd20edcff627ef36ae9fbb7d259f0ca404ee86f343cea2313c6d3fefc35c


event_val_hard_wrongpre_donor_map.csv.gz
raw SHA-256: 4bfb7d73e5cec58a70977ce719cb32f33e15b6e8f06edf917392f218801905a6


## Frozen original source snapshot
build_manifest.py  4337d8ec54d2fdaa5361e4c30e7544e16f0edf306f517dcff612b7fe12d7807a
dataset.py         165a71c4ebf6adea14f20f735bc487e883c5448e5ccfaf4fe8b74e49a3a77044
crop_utils.py      8c34139a0f96eaf8246dd228cf27c83386d6c1ca049426c25d28666576589593
damage_models.py   9f9ee3382902f339309ddf03a203b882d42e8674de7bf8ee3d18b14863478f2d


Place these exact four files under vendor/frozen_phase7d_source using the original relative layout.


## Phase7D-B Seed42
Final POST checkpoint
Final Siamese checkpoint
event_val_post_only_predictions.csv.gz
event_val_siamese_predictions.csv.gz
event_val_siamese_wrongpre_predictions.csv.gz
training_history.csv
results_index.json


## Phase7D-C
Final neural router
Final logistic router
Final HGB router
Final static-alpha JSON
Diagnostic metrics/bootstrap/utility evidence
Final results index


Frozen choices:
neural selected epoch: 1
static alpha: 0.45


## Phase7D-D
Final held-out clean predictions
Final held-out wrong-PRE predictions
Pooled metrics
Per-event metrics
Per-class metrics
Scene bootstrap
Utility bootstrap
Final results index
