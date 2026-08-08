"""Download pinned OMZ 2023.0 FP32 IR files and verify SHA-384."""

import hashlib
from pathlib import Path
from urllib.request import urlopen

BASE = "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/human-pose-estimation-0001/FP32/"
FILES = {
    "human-pose-estimation-0001.xml": "64e24ceca496aba4fb818edbea7c3a1405bc42c3166eac1b4be4f6cc89623c80cb9a1fd4c32d245d423653aa588e1a0b",
    "human-pose-estimation-0001.bin": "64ef06d261ae522409172a396d927f7345525dbde011e3e2bff487f73e7b1d8cf622091feba9c23c7d4625c0aab0a3e5",
}


def main():
    out = Path("models/human-pose-estimation-0001")
    out.mkdir(parents=True, exist_ok=True)
    for name, expected in FILES.items():
        data = urlopen(BASE + name, timeout=60).read()
        actual = hashlib.sha384(data).hexdigest()
        if actual != expected:
            raise SystemExit(f"SHA-384 mismatch for {name}: {actual}")
        (out / name).write_bytes(data)


if __name__ == "__main__":
    main()
