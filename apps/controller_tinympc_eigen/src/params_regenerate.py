#!/usr/bin/env python3
import argparse
import re
import sys
from collections import OrderedDict

import numpy as np


def parse_dump(text):
    pattern = re.compile(r"(\w+)\s*<<\s*(.*?);", re.DOTALL)
    matrices = OrderedDict()

    for name, body in pattern.findall(text):
        rows = []
        for raw_line in body.strip().splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.endswith(","):
                line = line[:-1]

            vals = []
            for x in line.split(","):
                x = x.strip()
                if not x:
                    continue
                vals.append(float(x.replace("f", "")))
            rows.append(vals)

        if not rows:
            raise ValueError(f"Matrix {name} has no rows")

        ncols = len(rows[0])
        for i, row in enumerate(rows):
            if len(row) != ncols:
                raise ValueError(
                    f"Matrix {name} has inconsistent row length at row {i}: "
                    f"expected {ncols}, got {len(row)}"
                )

        matrices[name] = np.array(rows, dtype=float)

    return matrices


def format_matrix(name, M, decimals=6):
    lines = [f"{name} << "]
    for i, row in enumerate(M):
        row_str = ",".join(f"{x:.{decimals}f}f" for x in row)
        lines.append(row_str + (";" if i == len(M) - 1 else ","))
    return "\n".join(lines)


def compute_dlqr(A, B, Q, R, n_steps=500):
    P = Q.copy()
    for _ in range(n_steps):
        K = np.linalg.solve(
            R + B.T @ P @ B + 1e-8 * np.eye(B.shape[1]),
            B.T @ P @ A,
        )
        P = Q + A.T @ P @ (A - B @ K)
    return P


def compute_params500_compat(A, B, Q, R, rho, dlqr_steps=500, riccati_steps=5000, tol=1e-10):
    """
    Dedicated params_500hz-compatible regeneration pipeline:
      1) P_lqr = DLQR(A,B,Q,R)
      2) K = (R + rho*I + B'PB)^(-1) B'PA
         P = Q + A'P(A-BK)
      3) Quu_inv = (R + rho*I + B'PB)^(-1)
      4) AmBKt = (A - BK)^T
      5) coeff_d2p = K'*(R + rho*I) - AmBKt*P*B
    """
    nu = B.shape[1]
    R_rho = R + rho * np.eye(nu)

    Pinf = compute_dlqr(A, B, Q, R, n_steps=dlqr_steps)
    Kinf = np.zeros((nu, A.shape[0]), dtype=float)

    for _ in range(riccati_steps):
        K_prev = Kinf.copy()
        Kinf = np.linalg.solve(R_rho + B.T @ Pinf @ B, B.T @ Pinf @ A)
        Pinf = Q + A.T @ Pinf @ (A - B @ Kinf)
        if np.linalg.norm(Kinf - K_prev, 2) < tol:
            break

    Quu_inv = np.linalg.inv(R_rho + B.T @ Pinf @ B)
    AmBKt = (A - B @ Kinf).T
    coeff_d2p = Kinf.T @ R_rho - AmBKt @ Pinf @ B

    return {
        "Kinf": Kinf,
        "Pinf": Pinf,
        "Quu_inv": Quu_inv,
        "AmBKt": AmBKt,
        "coeff_d2p": coeff_d2p,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate params_500hz derived matrices (single dedicated method)."
    )
    parser.add_argument("input", help="Input dump file")
    parser.add_argument("output", help="Output dump file")
    parser.add_argument("--rho", type=float, default=250.0, help="ADMM rho (default: 250.0)")
    parser.add_argument("--dlqr-steps", type=int, default=500, help="DLQR iterations (default: 500)")
    parser.add_argument("--riccati-steps", type=int, default=5000, help="Riccati iterations (default: 5000)")
    parser.add_argument("--tol", type=float, default=1e-10, help="Riccati convergence tol (default: 1e-10)")
    parser.add_argument("--decimals", type=int, default=6, help="Output decimals (default: 6)")
    args = parser.parse_args()

    with open(args.input, "r") as f:
        mats = parse_dump(f.read())

    required = ["A", "B", "Q", "R"]
    for name in required:
        if name not in mats:
            raise ValueError(f"Required matrix '{name}' not found in input dump")

    A = mats["A"]
    B = mats["B"]
    Q = mats["Q"]
    R = mats["R"]

    nx = A.shape[0]
    if A.shape[1] != nx:
        raise ValueError(f"A must be square, got {A.shape}")
    if B.shape[0] != nx:
        raise ValueError(f"B rows must match A size, got A={A.shape}, B={B.shape}")
    if Q.shape != (nx, nx):
        raise ValueError(f"Q must be {nx}x{nx}, got {Q.shape}")
    nu = B.shape[1]
    if R.shape != (nu, nu):
        raise ValueError(f"R must be {nu}x{nu}, got {R.shape}")

    out = compute_params500_compat(
        A=A,
        B=B,
        Q=Q,
        R=R,
        rho=args.rho,
        dlqr_steps=args.dlqr_steps,
        riccati_steps=args.riccati_steps,
        tol=args.tol,
    )

    mats["Kinf"] = out["Kinf"]
    mats["Pinf"] = out["Pinf"]
    mats["Quu_inv"] = out["Quu_inv"]
    mats["AmBKt"] = out["AmBKt"]
    mats["coeff_d2p"] = out["coeff_d2p"]

    with open(args.output, "w") as f:
        first = True
        for name, M in mats.items():
            if not first:
                f.write("\n\n")
            f.write(format_matrix(name, M, decimals=args.decimals))
            first = False

    print(f"Wrote recomputed dump to: {args.output}")
    print(f"rho = {args.rho}, dlqr_steps = {args.dlqr_steps}, riccati_steps = {args.riccati_steps}, tol = {args.tol}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
