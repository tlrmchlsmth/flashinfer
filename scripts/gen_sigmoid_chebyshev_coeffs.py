#!/usr/bin/env python3
"""Generate Chebyshev polynomial coefficients for sigmoid approximation.

Outputs a C++ SigmoidCoeffs specialization ready to paste into
quantization_utils.cuh or trtllm_fused_moe_dev_kernel.cu.

Usage:
    # Default: order 7, R=4
    python gen_sigmoid_chebyshev_coeffs.py

    # Custom config
    python gen_sigmoid_chebyshev_coeffs.py --order 9 --range 4

    # Sweep ranges to compare accuracy
    python gen_sigmoid_chebyshev_coeffs.py --sweep
"""

import argparse
import math


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def chebyshev_nodes(n):
    return [math.cos((2 * k + 1) * math.pi / (2 * n)) for k in range(n)]


# Chebyshev T_n(t) polynomial coefficients for odd orders
T_COEFFS = {
    1: [0, 1],
    3: [0, -3, 0, 4],
    5: [0, 5, 0, -20, 0, 16],
    7: [0, -7, 0, 56, 0, -112, 0, 64],
    9: [0, 9, 0, -120, 0, 432, 0, -576, 0, 256],
    11: [0, 11, 0, -220, 0, 1232, 0, -2816, 0, 2816, 0, 1024],
}


def compute_coefficients(order, R, N=4000):
    nodes = chebyshev_nodes(N)
    vals = [sigmoid(R * t) - 0.5 for t in nodes]

    cheb = {}
    for n in range(order + 1):
        cn = (2.0 / N) * sum(
            v * math.cos(n * math.acos(t)) for v, t in zip(vals, nodes)
        )
        cheb[n] = cn

    poly = [0.0] * (order + 1)
    for n in range(1, order + 1, 2):
        Tn = T_COEFFS.get(n)
        if Tn is None:
            raise ValueError(f"T_{n} coefficients not available (add to T_COEFFS)")
        for j in range(len(Tn)):
            poly[j] += cheb[n] * Tn[j]

    std_coeffs = {k: poly[k] / (R**k) for k in range(len(poly)) if k % 2 == 1 and abs(poly[k]) > 1e-20}
    return std_coeffs


def eval_sigmoid_approx(x, coeffs, R):
    """Evaluate sigmoid approximation: polynomial on [-R, R], exact outside."""
    if x < -R or x > R:
        return sigmoid(x)
    t = x * x
    horner = 0.0
    for k in sorted(coeffs.keys(), reverse=True):
        horner = horner * t + coeffs[k]
    return 0.5 + x * horner


def measure_error(coeffs, R, n_points=200000):
    max_silu_abs = 0.0
    max_silu_rel = 0.0

    for i in range(n_points):
        x = -10.0 + 20.0 * i / (n_points - 1)
        exact = x * sigmoid(x)
        approx = x * eval_sigmoid_approx(x, coeffs, R)

        err = abs(approx - exact)
        max_silu_abs = max(max_silu_abs, err)
        if abs(exact) > 0.01:
            max_silu_rel = max(max_silu_rel, err / abs(exact))

    return max_silu_abs, max_silu_rel


def emit_cpp(order, R, coeffs):
    lines = [f"template <>", f"struct SigmoidCoeffs<{order}, {R}> {{"]
    for k in sorted(coeffs.keys()):
        lines.append(f"  static constexpr float c{k} = {coeffs[k]:+.12e}f;")
    lines.append("};")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate Chebyshev sigmoid coefficients")
    parser.add_argument("--order", type=int, default=7, choices=[3, 5, 7, 9, 11])
    parser.add_argument("--range", type=int, default=4, dest="R")
    parser.add_argument("--sweep", action="store_true", help="Sweep R and order to compare")
    args = parser.parse_args()

    if args.sweep:
        print(f"{'Order':>6} {'R':>3} {'silu_abs':>10} {'silu_rel':>10} {'FP8 ok?':>8}")
        print("-" * 45)
        for R in [3, 4, 5, 6]:
            for order in [3, 5, 7, 9]:
                coeffs = compute_coefficients(order, R)
                abs_err, rel_err = measure_error(coeffs, R)
                ok = "YES" if rel_err < 0.06 else ("~" if rel_err < 0.12 else "no")
                print(f"{order:>6} {R:>3} {abs_err:>10.6f} {rel_err:>9.3%} {ok:>8}")
            print()
    else:
        coeffs = compute_coefficients(args.order, args.R)
        abs_err, rel_err = measure_error(coeffs, args.R)

        print(f"// Order {args.order}, R={args.R}")
        print(f"// Max silu absolute error: {abs_err:.6f}")
        print(f"// Max silu relative error: {rel_err:.3%}")
        print()
        print(emit_cpp(args.order, args.R, coeffs))


if __name__ == "__main__":
    main()
