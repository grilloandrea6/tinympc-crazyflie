#!/usr/bin/env python3
import argparse
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


POINT_RE = re.compile(
    r"BENCH point: H=(?P<H>\d+) iter=(?P<iter>\d+) "
    r"us\[min/avg/max\]=(?P<min>\d+)/(?P<avg>\d+)/(?P<max>\d+) "
    r"miss=(?P<miss>\d+)/(?P<total>\d+)"
)

STATUS_RE = re.compile(
    r"BENCH status: solved=(?P<solved>\d+) max_iter=(?P<max_iter>\d+) "
    r"noncvx=(?P<noncvx>\d+) other=(?P<other>\d+)"
)


@dataclass
class Point:
    H: int
    it: int
    min_us: int
    avg_us: int
    max_us: int
    miss: int
    total: int
    solved: int = 0
    max_iter: int = 0
    noncvx: int = 0
    other: int = 0


@dataclass
class Group:
    H: int
    points: List[Point] = field(default_factory=list)

    def fit(self):
        xs = [p.it for p in self.points]
        ys = [p.avg_us for p in self.points]
        n = len(xs)
        if n < 2:
            return float("nan"), float("nan"), float("nan")
        sx = sum(xs)
        sy = sum(ys)
        sxx = sum(x * x for x in xs)
        sxy = sum(x * y for x, y in zip(xs, ys))
        den = n * sxx - sx * sx
        if den == 0:
            return float("nan"), float("nan"), float("nan")
        k = (n * sxy - sx * sy) / den
        t0 = (sy - k * sx) / n
        yhat = [t0 + k * x for x in xs]
        ss_res = sum((y - yh) ** 2 for y, yh in zip(ys, yhat))
        ybar = sy / n
        ss_tot = sum((y - ybar) ** 2 for y in ys)
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else float("nan")
        return t0, k, r2


def parse(text: str) -> Dict[int, Group]:
    groups: Dict[int, Group] = {}
    last_point: Optional[Point] = None
    for line in text.splitlines():
        pm = POINT_RE.search(line)
        if pm:
            p = Point(
                H=int(pm.group("H")),
                it=int(pm.group("iter")),
                min_us=int(pm.group("min")),
                avg_us=int(pm.group("avg")),
                max_us=int(pm.group("max")),
                miss=int(pm.group("miss")),
                total=int(pm.group("total")),
            )
            groups.setdefault(p.H, Group(H=p.H)).points.append(p)
            last_point = p
            continue
        sm = STATUS_RE.search(line)
        if sm and last_point is not None:
            last_point.solved = int(sm.group("solved"))
            last_point.max_iter = int(sm.group("max_iter"))
            last_point.noncvx = int(sm.group("noncvx"))
            last_point.other = int(sm.group("other"))
    for g in groups.values():
        g.points.sort(key=lambda p: p.it)
    return groups


def pct(a: float) -> str:
    return f"{100.0*a:.2f}%"


def main() -> int:
    ap = argparse.ArgumentParser(description="Analyze TinyMPC benchmark logs.")
    ap.add_argument("logfile", type=Path, help="Path to text log file")
    ap.add_argument("--budget-us", type=float, default=2000.0, help="Deadline budget in microseconds")
    args = ap.parse_args()

    text = args.logfile.read_text()
    groups = parse(text)
    if not groups:
        print("No benchmark points found.")
        return 1

    print("Per-H fits (avg_us ~= T0 + k_iter*iter)")
    print("H | points | T0_us | k_iter_us/iter | k_step_iter_us/(iter*step) | R^2")
    for H in sorted(groups):
        g = groups[H]
        t0, k, r2 = g.fit()
        k_step = k / H if H > 0 else float("nan")
        print(f"{H:>2} | {len(g.points):>6} | {t0:>6.1f} | {k:>15.1f} | {k_step:>27.3f} | {r2:>4.4f}")

    print("\nPer-point summary")
    print("H,iter,avg_us,min_us,max_us,miss_rate,solved,max_iter,noncvx,other")
    for H in sorted(groups):
        for p in groups[H].points:
            miss_rate = p.miss / p.total if p.total else float("nan")
            print(
                f"{p.H},{p.it},{p.avg_us},{p.min_us},{p.max_us},{miss_rate:.4f},"
                f"{p.solved},{p.max_iter},{p.noncvx},{p.other}"
            )

    hs = sorted(groups)
    if len(hs) >= 2:
        print("\nCross-H consistency check (k_step_iter)")
        vals = []
        for H in hs:
            _, k, _ = groups[H].fit()
            vals.append((H, k / H))
        mean = sum(v for _, v in vals) / len(vals)
        max_dev = max(abs(v - mean) / mean for _, v in vals) if mean else float("nan")
        for H, v in vals:
            print(f"H={H}: {v:.3f} us/(iter*step)")
        print(f"mean={mean:.3f}, max_rel_dev={pct(max_dev)}")

    print("\nBudget check")
    print(f"budget_us={args.budget_us:.1f}")
    for H in sorted(groups):
        g = groups[H]
        feas = [p.it for p in g.points if p.avg_us <= args.budget_us]
        msg = ",".join(map(str, feas)) if feas else "none"
        print(f"H={H}: avg-feasible iters <= budget: {msg}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

