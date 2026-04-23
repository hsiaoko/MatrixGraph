#!/usr/bin/env python3
"""
Collect graph features from a MatrixGraph CSR directory (meta.yaml + graphs/0.bin).
Calls apps: wcc_exec, diameter_exec, skew_exec (via subprocess).

Dependencies: PyYAML; NumPy recommended for large graphs (optional fallback to statistics).

Example:
  ./tools/python/graph_features.py -g /data/mygraph/ -o features.yaml
  ./tools/python/graph_features.py -g /data/mygraph/   # prints YAML to stdout
"""

from __future__ import annotations

import argparse
import os
import re
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_bin_dir() -> Path:
    return _repo_root() / "bin"


def load_subgraph_meta(graph_root: Path) -> Dict[str, Any]:
    import yaml

    meta_path = graph_root / "meta.yaml"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing meta.yaml: {meta_path}")
    with open(meta_path, "r", encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    gm = doc.get("GraphMetadata") or doc
    subs = gm.get("subgraphs") or []
    if not subs:
        raise ValueError("GraphMetadata.subgraphs is empty")
    sg = subs[0]
    return {
        "num_vertices": int(sg["num_vertices"]),
        "num_incoming_edges": int(sg["num_incoming_edges"]),
        "num_outgoing_edges": int(sg["num_outgoing_edges"]),
        "max_vid": int(sg["max_vid"]),
        "min_vid": int(sg["min_vid"]),
    }


def read_total_degrees(graph_root: Path, sg: Dict[str, Any]) -> Tuple[list, int]:
    """Return (list of total degree per local vertex, num_outgoing_edges for |E| report)."""
    n = sg["num_vertices"]
    bin_path = graph_root / "graphs" / "0.bin"
    if not bin_path.is_file():
        raise FileNotFoundError(f"Missing CSR bin: {bin_path}")

    # Layout matches ImmutableCSR::Write (immutable_csr.cu)
    # globalid[n], indegree[n], outdegree[n], ...
    with open(bin_path, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(0)
        need = 4 * n * 3  # skip globalid + indegree + outdegree
        if size < need:
            raise ValueError(f"CSR bin too small: {size} < {need}")
        f.seek(4 * n)  # skip globalid
        raw_in = f.read(4 * n)
        raw_out = f.read(4 * n)
        if len(raw_in) != 4 * n or len(raw_out) != 4 * n:
            raise ValueError("Short read for degree arrays")

    indeg = list(struct.unpack(f"<{n}I", raw_in))
    outdeg = list(struct.unpack(f"<{n}I", raw_out))
    total = [float(indeg[i] + outdeg[i]) for i in range(n)]
    return total, int(sg["num_outgoing_edges"])


def degree_stats(total_deg: list) -> Dict[str, float]:
    try:
        import numpy as np

        a = np.asarray(total_deg, dtype=np.float64)
        return {
            "avg": float(a.mean()),
            "max": float(a.max()),
            "min": float(a.min()),
            "std": float(a.std(ddof=0)),
        }
    except ImportError:
        import statistics

        return {
            "avg": float(statistics.mean(total_deg)),
            "max": float(max(total_deg)),
            "min": float(min(total_deg)),
            "std": float(statistics.pstdev(total_deg)) if len(total_deg) > 1 else 0.0,
        }


def run_app(
    bin_path: Path,
    args: list,
    cwd: Optional[Path],
    timeout: Optional[float],
) -> str:
    env = os.environ.copy()
    # Avoid locale surprises in parsed numbers
    env.setdefault("LC_ALL", "C")
    r = subprocess.run(
        [str(bin_path)] + args,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )
    out = (r.stdout or "") + "\n" + (r.stderr or "")
    if r.returncode != 0:
        raise RuntimeError(
            f"Command failed ({r.returncode}): {bin_path} {' '.join(args)}\n{out}"
        )
    return out


def parse_wcc_components(text: str) -> int:
    m = re.search(
        r"\[WCC\]\s*num_weakly_connected_components:\s*(\d+)", text
    )
    if not m:
        raise ValueError("Could not parse WCC component count from output")
    return int(m.group(1))


def parse_diameter(text: str) -> int:
    last: Optional[int] = None
    for line in text.splitlines():
        if "[Diameter] undirected diameter" in line:
            m = re.search(r":\s*(\d+)\s*$", line.strip())
            if m:
                last = int(m.group(1))
    if last is None:
        raise ValueError("Could not parse diameter from diameter_exec output")
    return last


def parse_skew_ratio(text: str) -> float:
    last: Optional[float] = None
    for line in text.splitlines():
        if "[Skew] skew(G)" in line and "d_bar" in line:
            m = re.search(r"=\s*([0-9.eE+-]+)\s*$", line.strip())
            if m:
                last = float(m.group(1))
    if last is None:
        raise ValueError("Could not parse skew from skew_exec output")
    return last


def build_features(
    graph_root: Path,
    bin_dir: Path,
    diameter_samples: int,
    skew_samples: int,
    seed: int,
    timeout: Optional[float],
) -> Dict[str, Any]:
    graph_root = graph_root.resolve()
    sg = load_subgraph_meta(graph_root)
    total_deg, num_edges_report = read_total_degrees(graph_root, sg)
    dstats = degree_stats(total_deg)

    gstr = str(graph_root) + os.sep

    wcc_bin = bin_dir / "wcc_exec"
    dia_bin = bin_dir / "diameter_exec"
    skew_bin = bin_dir / "skew_exec"
    for p in (wcc_bin, dia_bin, skew_bin):
        if not p.is_file():
            raise FileNotFoundError(f"Missing executable: {p} (build apps first)")

    wcc_out = run_app(wcc_bin, ["-g", gstr], cwd=None, timeout=timeout)
    ncomp = parse_wcc_components(wcc_out)

    dia_args = ["-g", gstr, "-diameter_samples", str(diameter_samples), "-diameter_seed", str(seed)]
    dia_out = run_app(dia_bin, dia_args, cwd=None, timeout=timeout)
    diam = parse_diameter(dia_out)

    skew_args = ["-g", gstr, "-skew_samples", str(skew_samples), "-skew_seed", str(seed)]
    skew_out = run_app(skew_bin, skew_args, cwd=None, timeout=timeout)
    skew_val = parse_skew_ratio(skew_out)

    return {
        "graph_features": {
            "basic": {
                "num_edges": num_edges_report,
                "num_vertices": sg["num_vertices"],
                "num_components": ncomp,
            },
            "degree": {
                "avg": dstats["avg"],
                "max": dstats["max"],
                "min": dstats["min"],
                "skew": skew_val,
                "std": dstats["std"],
            },
            "diameter": diam,
        }
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Export CSR graph features to YAML")
    parser.add_argument(
        "-g",
        "--graph",
        required=True,
        help="CSR graph directory (meta.yaml + graphs/0.bin + label/0.bin)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="",
        help="Write YAML to this file (default: stdout)",
    )
    parser.add_argument(
        "--bin-dir",
        type=Path,
        default=None,
        help=f"Directory containing *_exec (default: {_default_bin_dir()})",
    )
    parser.add_argument(
        "--diameter-samples",
        type=int,
        default=50,
        help="Forwarded to diameter_exec -diameter_samples (0 = exact, slow)",
    )
    parser.add_argument(
        "--skew-samples",
        type=int,
        default=50,
        help="Forwarded to skew_exec -skew_samples (0 = exact d_hat, slow)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for diameter/skew sampling",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Per-app subprocess timeout in seconds (optional)",
    )
    args = parser.parse_args()

    bin_dir = args.bin_dir if args.bin_dir is not None else _default_bin_dir()

    try:
        data = build_features(
            Path(args.graph),
            bin_dir,
            diameter_samples=args.diameter_samples,
            skew_samples=args.skew_samples,
            seed=args.seed,
            timeout=args.timeout,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    import yaml

    yml = yaml.dump(
        data,
        default_flow_style=False,
        sort_keys=False,
        allow_unicode=True,
    )
    if args.output:
        Path(args.output).write_text(yml, encoding="utf-8")
    else:
        sys.stdout.write(yml)
    return 0


if __name__ == "__main__":
    sys.exit(main())
