import json
from pathlib import Path


def main():
    choices = {"neural_selected_epoch":1, "static_alpha":0.45}
    print(json.dumps(choices, indent=2))
    raise SystemExit(
        "The canonical paper release must restore the already-frozen final Phase7D-C routers. "
        "Refitting is a replication experiment, not required for exact paper-number reproduction."
    )


if __name__ == "__main__":
    main()
