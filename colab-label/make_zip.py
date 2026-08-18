# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0
"""Pack the Colab labeling bundle: judge.py + run_labels.py + bundle/*.jsonl -> router_label_bundle.zip"""
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
out = HERE / "router_label_bundle.zip"
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for name in ("judge.py", "run_labels.py"):
        z.write(HERE / name, f"colab-label/{name}")
    for p in sorted((HERE / "bundle").glob("*")):
        z.write(p, f"colab-label/bundle/{p.name}")
print(f"wrote {out} ({out.stat().st_size/1e6:.1f} MB)")
