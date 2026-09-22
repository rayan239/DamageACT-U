from pathlib import Path
import json
from damageactu.utils.reproducibility import capture_environment
from damageactu.utils.hashing import raw_file_sha256


def main():
    env = capture_environment()
    print(json.dumps(env, indent=2))
    print("PASS: environment capture completed.")
    p = Path.home()/".cache/torch/hub/checkpoints/resnet18-f37072fd.pth"
    if p.is_file():
        got = raw_file_sha256(p)
        exp = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
        if got != exp:
            raise RuntimeError(f"ResNet18 weight hash mismatch: {got}")
        print("PASS: pretrained ResNet18 identity.")
    else:
        print("NOTE: pretrained ResNet18 file is not cached locally; hash check skipped.")


if __name__ == "__main__":
    main()
