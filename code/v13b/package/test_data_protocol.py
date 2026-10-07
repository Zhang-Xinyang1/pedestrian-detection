"""Validate unchanged data and sampler protocol."""
import hashlib
import json
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent

def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

record = json.loads((PACKAGE / "DATA_PROTOCOL_SHA256.json").read_text(encoding="utf8"))
for name, expected in record["files"].items():
    if sha(PACKAGE / name) != expected:
        raise ValueError("Data protocol source changed: " + name)
print("XMODAL_DATA_PROTOCOL_OK", record)
