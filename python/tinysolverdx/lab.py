"""Numerical error lab: backward/forward error vs condition number and size.

Every precision is exercised on the *same* rounded input matrices, factored with
the library kernels (emulated per-operation rounding for fp16/bf16) and judged
against long-double ground truth (see ``reference_solve`` & friends).

    python -m tinysolverdx.lab --out docs/report [--quick]
"""
import argparse
import csv
import os
import pathlib

import numpy as np

import tinysolverdx as tsdx
from tinysolverdx import matrices

PRECISIONS = ["fp16", "bf16", "fp32", "fp64"]
COLORS = {"fp16": "#d55e00", "bf16": "#cc79a7", "fp32": "#0072b2", "fp64": "#009e73"}
KINDS = {  # kind -> (label, matrix generator, factor/solve)
    "chol": "Cholesky", "lu": "LU (partial pivoting)",
}


def _med(v, ok):
    v = v[ok]
    return float(np.median(v)) if v.size else np.nan


def _max(v, ok):
    v = v[ok]
    return float(np.max(v)) if v.size else np.nan


def run_point(rng, prec, kappa, n, trials):
    """Returns one row per kind for the given (precision, kappa, n)."""
    rows = []
    b = rng.standard_normal((trials, n))
    for kind, gen in (("chol", matrices.spd), ("lu", matrices.general)):
        A = tsdx.round_to(gen(rng, trials, n, kappa), prec)  # the matrix actually factored
        cond = np.linalg.cond(A)
        if kind == "chol":
            L, info = tsdx.cholesky(A, precision=prec)
            bw = tsdx.chol_backward_error(A, L.astype(np.float64))
            x, _ = tsdx.posv(A, b, precision=prec)
        else:
            LU, piv, info = tsdx.lu(A, precision=prec)
            bw = tsdx.lu_backward_error(A, LU.astype(np.float64), piv)
            x, _ = tsdx.gesv(A, b, precision=prec)
        ok = (info == 0) & np.isfinite(bw)
        x = x.astype(np.float64)
        okx = ok & np.isfinite(x).all(axis=1)
        x0 = np.where(okx[:, None], x, 0.0)
        fwd = tsdx.solve_forward_error(A, x0, b)
        sbw = tsdx.solve_backward_error(A, x0, b)
        rows.append(dict(precision=prec, kind=kind, n=n, kappa=kappa, cond_actual=_med(cond, np.isfinite(cond)),
                         fail_frac=float(1 - ok.mean()), backward_med=_med(bw, ok), backward_max=_max(bw, ok),
                         solve_backward_med=_med(sbw, okx), forward_med=_med(fwd, okx), forward_max=_max(fwd, okx)))
    return rows


def sweep(seed, trials, n_fixed, kappas, sizes, size_kappa):
    rng = np.random.default_rng(seed)
    cond_rows, size_rows = [], []
    for prec in PRECISIONS:
        for k in kappas:
            cond_rows += run_point(rng, prec, k, n_fixed, trials)
        for n in sizes:
            size_rows += run_point(rng, prec, size_kappa, n, trials)
    return cond_rows, size_rows


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _series(rows, prec, kind, key, x="kappa"):
    r = [q for q in rows if q["precision"] == prec and q["kind"] == kind]
    xs = np.array([q[x] for q in r], float)
    ys = np.array([q[key] for q in r], float)
    return xs, ys


def make_plots(cond_rows, size_rows, out, n_fixed, size_kappa):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
                         "axes.spines.right": False, "font.size": 10})
    kinds = {"chol": ("Cholesky (SPD)", "-", "o"), "lu": ("LU, partial pivoting (general)", "--", "s")}

    for prec in PRECISIONS:
        u = tsdx.unit_roundoff(prec)
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
        for kind, (lab, ls, mk) in kinds.items():
            xs, bw = _series(cond_rows, prec, kind, "backward_med")
            ax[0].loglog(xs, bw, ls, marker=mk, color=COLORS[prec], label=lab)
            xs, fw = _series(cond_rows, prec, kind, "forward_med")
            ax[1].loglog(xs, fw, ls, marker=mk, color=COLORS[prec], label=lab)
            xs, ff = _series(cond_rows, prec, kind, "fail_frac")
            ax[2].semilogx(xs, ff, ls, marker=mk, color=COLORS[prec], label=lab)
        ks = np.array(sorted({r["kappa"] for r in cond_rows}))
        ax[0].axhline(u, color="gray", ls=":", label=f"unit roundoff u={u:.1e}")
        ax[1].loglog(ks, np.minimum(ks * u, 1.0), color="gray", ls=":", label=r"$\kappa\,u$")
        ax[0].set(title=f"{prec}: backward error ||A-LL'||/||A||", xlabel=r"condition number $\kappa_2$", ylabel="median over trials")
        ax[1].set(title=f"{prec}: solve forward error", xlabel=r"condition number $\kappa_2$", ylabel="||x-x_ref||/||x_ref||")
        ax[2].set(title=f"{prec}: factorization failure rate", xlabel=r"condition number $\kappa_2$", ylabel="fraction with info>0", ylim=(-0.02, 1.02))
        for a in ax:
            a.legend(fontsize=8)
        fig.suptitle(f"n={n_fixed}, long-double reference, inputs rounded to {prec}", y=1.02)
        fig.tight_layout()
        fig.savefig(out / f"error_vs_cond_{prec}.png", dpi=130, bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True)
    for a, kind in zip(ax, ("chol", "lu")):
        for prec in PRECISIONS:
            xs, bw = _series(cond_rows, prec, kind, "backward_med")
            a.loglog(xs, bw, "-o", ms=4, color=COLORS[prec], label=prec)
            a.axhline(tsdx.unit_roundoff(prec), color=COLORS[prec], ls=":", lw=0.8)
        a.set(title=f"{kinds[kind][0]}: backward error (n={n_fixed})", xlabel=r"condition number $\kappa_2$")
        a.legend()
    ax[0].set_ylabel("median normwise backward error")
    fig.tight_layout()
    fig.savefig(out / "backward_error_vs_cond.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True)
    for a, kind in zip(ax, ("chol", "lu")):
        for prec in PRECISIONS:
            xs, bw = _series(size_rows, prec, kind, "backward_med", x="n")
            a.loglog(xs, bw, "-o", ms=4, color=COLORS[prec], label=prec)
        a.set(title=f"{kinds[kind][0]}: backward error vs size (kappa={size_kappa:g})", xlabel="matrix size n")
        a.legend()
    ax[0].set_ylabel("median normwise backward error")
    fig.tight_layout()
    fig.savefig(out / "backward_error_vs_n.png", dpi=130)
    plt.close(fig)


def fmt(v):
    return "-" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.2e}"


def write_report(path, cond_rows, size_rows, n_fixed, size_kappa, trials, seed):
    L = ["# TinySolverDx numerical error report", "",
         f"Generated by `python -m tinysolverdx.lab` (seed={seed}, {trials} matrices per point, n={n_fixed} for the "
         "condition sweep). Matrices are built with a prescribed 2-norm condition number, rounded to the target "
         "precision, factored by the library (fp16/bf16 emulate per-operation rounding on the CPU) and compared "
         "with a long-double reference (`reference_solve`, residuals in long double).", "",
         "Metrics: backward error `||A-LL^T||_F/||A||_F` (Cholesky) and `||PA-LU||_F/||A||_F` (LU); solve forward error "
         "`||x-x_ref||_2/||x_ref||_2`; failure rate = fraction of matrices with `info>0` (non-positive pivot / zero pivot). "
         "Medians are over successful factorizations.", "",
         "![backward error vs kappa](backward_error_vs_cond.png)", "",
         "![backward error vs n](backward_error_vs_n.png)", ""]
    for prec in PRECISIONS:
        L += [f"## {prec} (u = {tsdx.unit_roundoff(prec):.2e})", "", f"![{prec}](error_vs_cond_{prec}.png)", "",
              "| kappa | chol backward | chol forward | chol fail | LU backward | LU forward | LU fail |", "|---|---|---|---|---|---|---|"]
        for k in sorted({r["kappa"] for r in cond_rows}):
            c = next(r for r in cond_rows if r["precision"] == prec and r["kind"] == "chol" and r["kappa"] == k)
            g = next(r for r in cond_rows if r["precision"] == prec and r["kind"] == "lu" and r["kappa"] == k)
            L.append(f"| {k:.0e} | {fmt(c['backward_med'])} | {fmt(c['forward_med'])} | {c['fail_frac']:.2f} | "
                     f"{fmt(g['backward_med'])} | {fmt(g['forward_med'])} | {g['fail_frac']:.2f} |")
        L.append("")
    L += ["## Reading the results", "",
          "* Backward error stays within a small multiple of the unit roundoff `u` (growing mildly with n) and does *not grow with "
          "kappa* for as long as the factorization succeeds: both algorithms are backward stable.",
          "* Forward error tracks `kappa*u`; once `kappa*u` approaches 1 the solution loses all digits "
          "(and Cholesky starts failing with `info>0`, since rounding makes the matrix numerically indefinite).",
          "* fp16 has a very narrow range: tiny entries of ill-conditioned inputs underflow/flush to subnormals, "
          "so its failure rate rises well before `kappa = 1/u`.",
          f"* Size sweep (kappa={size_kappa:g}): backward error grows roughly like `sqrt(n)`-`n` times `u`, as expected from standard bounds.", ""]
    path.write_text("\n".join(L))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.environ.get("TSDX_OUT_DIR", "docs/report"))
    ap.add_argument("--seed", type=int, default=int(os.environ.get("TSDX_SEED", "0")))
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--n", type=int, default=16, help="matrix size for the condition sweep")
    ap.add_argument("--quick", action="store_true", help="fewer trials/points (CI smoke run)")
    a = ap.parse_args(argv)
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    trials = 20 if a.quick else a.trials
    kappas = [1e1, 1e2, 1e4, 1e6, 1e8, 1e10, 1e12] if a.quick else [10.0 ** e for e in range(1, 13)]
    sizes = [2, 8, 32, 64] if a.quick else [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]
    size_kappa = 10.0
    cond_rows, size_rows = sweep(a.seed, trials, a.n, kappas, sizes, size_kappa)
    write_csv(out / "error_vs_cond.csv", cond_rows)
    write_csv(out / "error_vs_n.csv", size_rows)
    make_plots(cond_rows, size_rows, out, a.n, size_kappa)
    write_report(out / "REPORT.md", cond_rows, size_rows, a.n, size_kappa, trials, a.seed)
    print(f"wrote report to {out}/")


if __name__ == "__main__":
    main()
