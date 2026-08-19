# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0
"""QEMU arm64 runtime check on a Windows/x86_64 host, replicating the official per-tier limits
(2 CPU, 2 GiB, no swap, pids 32, read-only root, /tmp 256m, no network, uid 65532).
`tools/check_runtime.py` needs fcntl (POSIX); this is the Windows stand-in — the timings are
QEMU-emulated and only indicative, pass/fail is decided on native arm64.

    python tools/qemu_check.py --image ossp-router:e43 --input data/combined/inputs.json --out build/qemu/hit
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tiers", default="fast,balanced,premium")
    ap.add_argument("--repeat", type=int, default=1)
    args = ap.parse_args()
    results = []
    for tier in args.tiers.split(","):
        for rep in range(args.repeat):
            out = args.out / tier
            if out.exists():
                for p in out.iterdir():
                    p.unlink()
            out.mkdir(parents=True, exist_ok=True)
            cmd = [
                "docker", "run", "--rm", "--pull", "never", "--log-driver", "none", "--stop-signal", "SIGTERM",
                "--no-healthcheck", "--platform", "linux/arm64", "--network", "none", "--ipc", "none",
                "--cgroupns", "private", "--ulimit", "core=0:0", "--read-only", "--user", "65532:65532",
                "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--cpus", "2", "--memory", "2g",
                "--memory-swap", "2g", "--pids-limit", "32", "--tmpfs", "/tmp:rw,noexec,nosuid,size=256m",
                "--mount", f"type=bind,src={args.input.resolve()},dst=/challenge/input/inputs.json,readonly",
                "--mount", f"type=bind,src={out.resolve()},dst=/challenge/output",
                args.image, "--input", "/challenge/input/inputs.json", "--tier", tier,
                "--output", "/challenge/output/submission.json",
            ]
            t0 = time.perf_counter()
            r = subprocess.run(cmd, capture_output=True, text=True)
            wall = time.perf_counter() - t0
            sub = out / "submission.json"
            size = sub.stat().st_size if sub.exists() else None
            n = None
            if size:
                try:
                    n = len(json.loads(sub.read_text(encoding="utf-8")).get("selections", []))
                except Exception:
                    n = "?"
            results.append({"tier": tier, "rep": rep, "rc": r.returncode, "wall_s": round(wall, 1), "size": size, "n": n})
            print(f"{tier} rep{rep}: rc={r.returncode} wall={wall:.1f}s size={size} selections={n}"
                  + (f" | stderr: {r.stderr.strip()[-200:]}" if r.returncode else ""), flush=True)
    (args.out / "qemu_check.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    return 0 if all(x["rc"] == 0 for x in results) else 1


if __name__ == "__main__":
    sys.exit(main())
