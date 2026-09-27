"""Throughput benchmark + roofline-style chart.

Compares, per matrix size, TinySolverDx backends (SIMD tile / scalar, all threads and
one thread), the naive C++ baseline, batched ``numpy.linalg.cholesky`` and a Python
loop over ``numpy`` / ``scipy`` calls. Ceilings come from measured machine probes.

    python -m tinysolverdx.bench --out docs/report [--quick] [--precision fp32]
"""
import argparse
import csv
import os
import pathlib
import time

import numpy as np
import scipy.linalg as sla

import tinysolverdx as tsdx
from tinysolverdx import matrices

SIZES = [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]


def best_time(fn, reps=5, min_total=0.05):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
        if sum(ts) > 2.0:
            break
    return min(ts)


def flops(kind, n):
    return n ** 3 / 3 if kind == "chol" else 2 * n ** 3 / 3


def bench_kind(kind, n, dtype, prec, batch, loop_batch, threads):
    rng = np.random.default_rng(n)
    gen = matrices.spd if kind == "chol" else matrices.general
    A = gen(rng, batch, n, 10.0).astype(dtype)
    Al = A[:loop_batch]
    f = tsdx.cholesky if kind == "chol" else tsdx.lu
    res = {}

    def tsdx_run(backend, nt):
        tsdx.set_num_threads(nt)
        t = best_time(lambda: f(A, precision=prec, backend=backend))
        return flops(kind, n) * batch / t * 1e-9

    res["tsdx simd (all threads)"] = tsdx_run("simd", threads)
    res["tsdx scalar (all threads)"] = tsdx_run("scalar", threads)
    res["tsdx simd (1 thread)"] = tsdx_run("simd", 1)
    res["naive C++ (1 thread)"] = tsdx_run("naive", 1)
    tsdx.set_num_threads(0)

    if kind == "chol":
        t = best_time(lambda: np.linalg.cholesky(A))
        res["numpy batched"] = flops(kind, n) * batch / t * 1e-9
        loop = lambda: [np.linalg.cholesky(a) for a in Al]
    else:
        loop = lambda: [sla.lu_factor(a, check_finite=False) for a in Al]
    t = best_time(loop, reps=3)
    res["python loop (numpy/scipy)"] = flops(kind, n) * loop_batch / t * 1e-9
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.environ.get("TSDX_OUT_DIR", "docs/report"))
    ap.add_argument("--precision", default="fp32", choices=["fp32", "fp64"])
    ap.add_argument("--threads", type=int, default=0, help="0 = all available")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args(argv)
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dtype = np.float32 if a.precision == "fp32" else np.float64
    if a.threads:
        tsdx.set_num_threads(a.threads)
    threads = tsdx.num_threads()
    sizes = [2, 8, 16, 32, 64] if a.quick else SIZES

    peak = tsdx.measure_peak_gflops(a.precision)
    bw = tsdx.measure_bandwidth_gbs()
    print(f"threads={threads} openmp={tsdx.has_openmp()} peak={peak:.1f} GFLOP/s bandwidth={bw:.1f} GB/s")

    rows = []
    for kind in ("chol", "lu"):
        for n in sizes:
            batch = max(2048, (1 << 21) // (n * n))
            batch = min(batch, 20000 if not a.quick else 4000)
            loop_batch = min(batch, 1500 if not a.quick else 400)
            for name, g in bench_kind(kind, n, dtype, a.precision, batch, loop_batch, threads).items():
                rows.append(dict(kind=kind, n=n, precision=a.precision, series=name, gflops=g, batch=batch, threads=threads))
            r = {x["series"]: x["gflops"] for x in rows if x["kind"] == kind and x["n"] == n}
            print(f"{kind:4s} n={n:2d} simd={r['tsdx simd (all threads)']:8.2f} naive={r['naive C++ (1 thread)']:7.2f} "
                  f"loop={r['python loop (numpy/scipy)']:6.2f} GFLOP/s")
    with open(out / f"bench_{a.precision}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    plot(rows, out, a.precision, peak, bw, threads)
    print(f"wrote {out}/roofline_{a.precision}.png")


def plot(rows, out, prec, peak, bw, threads):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker

    style = {"tsdx simd (all threads)": ("#0072b2", "-", "o"), "tsdx scalar (all threads)": ("#56b4e9", "-", "^"),
             "tsdx simd (1 thread)": ("#009e73", "--", "s"), "naive C++ (1 thread)": ("#d55e00", "--", "v"),
             "numpy batched": ("#999999", ":", "D"), "python loop (numpy/scipy)": ("#cc79a7", "-", "x")}
    item = 4 if prec == "fp32" else 8
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, kind, title in zip(axes, ("chol", "lu"), ("Batched Cholesky", "Batched LU (partial pivoting)")):
        ns = sorted({r["n"] for r in rows if r["kind"] == kind})
        for name, (c, ls, mk) in style.items():
            ys = [next((r["gflops"] for r in rows if r["kind"] == kind and r["n"] == n and r["series"] == name), np.nan) for n in ns]
            if not np.all(np.isnan(ys)):
                ax.loglog(ns, ys, ls, color=c, marker=mk, label=name)
        # roofline for a factorization that streams n^2 elements in and out of memory
        nn = np.geomspace(min(ns), max(ns), 100)
        ai = np.array([flops(kind, n) for n in nn]) / (2 * nn ** 2 * item)
        ax.loglog(nn, np.minimum(peak, bw * ai), "k-", lw=1.2, alpha=0.7, label="roofline (measured peak & DRAM BW)")
        ax.axhline(peak, color="k", ls=":", lw=0.8)
        ax.set(title=f"{title} ({prec}, {threads} threads)", xlabel="matrix size n", xticks=ns, xticklabels=[str(n) for n in ns])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.grid(alpha=0.25, which="both")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("GFLOP/s (higher is better)")
    axes[1].legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / f"roofline_{prec}.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
