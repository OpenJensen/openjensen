import json
import os
import pathlib
import subprocess

root = pathlib.Path.home() / "smolvla-benchmark"
os.chdir(root)
code = """import hashlib,json,numpy as np
rows={}
for initial in range(2):
 h=hashlib.sha256()
 for chunk in range(128):
  noise=np.random.default_rng(42+1000*initial+chunk).standard_normal((1,50,32)).astype('float32')
  h.update(noise.tobytes())
 rows[str(initial)]=h.hexdigest()
print(json.dumps({'numpy':np.__version__,'noise_sha256':rows}))
"""
rows = {}
for engine in ("native", "vllm", "trtllm"):
    rows[engine] = json.loads(
        subprocess.check_output(
            [str(root / (".venv-" + engine) / "bin/python"), "-c", code], text=True
        )
    )
assert (
    rows["native"]["noise_sha256"] == rows["vllm"]["noise_sha256"] == rows["trtllm"]["noise_sha256"]
)
result = {
    "status": "passed",
    "initial_states": [0, 1],
    "chunks_per_initial_state": 128,
    "seed_rule": "42 + 1000 * initial_state + chunk_index",
    "shape": [1, 50, 32],
    "dtype": "float32",
    "environments": rows,
}
(root / "comparison/quality-noise-parity.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
