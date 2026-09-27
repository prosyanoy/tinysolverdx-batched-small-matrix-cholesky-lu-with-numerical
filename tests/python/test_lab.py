import csv

import numpy as np

from tinysolverdx import bench, lab


def test_lab_quick_run_produces_consistent_report(tmp_path):
    lab.main(["--out", str(tmp_path), "--quick", "--trials", "10"])
    for name in ["REPORT.md", "backward_error_vs_cond.png", "backward_error_vs_n.png",
                 "error_vs_cond_fp16.png", "error_vs_cond_fp64.png", "error_vs_cond.csv"]:
        assert (tmp_path / name).exists()
    rows = list(csv.DictReader(open(tmp_path / "error_vs_cond.csv")))
    for r in rows:
        if r["precision"] == "fp64" and r["backward_med"] != "nan":
            assert float(r["backward_med"]) < 1e-14          # stable, regardless of kappa
    fp32 = [r for r in rows if r["precision"] == "fp32" and r["kind"] == "lu"]
    fwd = [float(r["forward_med"]) for r in fp32]
    assert fwd[-1] > 1e3 * fwd[0]                              # forward error grows with kappa


def test_bench_quick_run(tmp_path):
    bench.main(["--out", str(tmp_path), "--quick"])
    assert (tmp_path / "roofline_fp32.png").exists()
    rows = list(csv.DictReader(open(tmp_path / "bench_fp32.csv")))
    assert {"tsdx simd (all threads)", "naive C++ (1 thread)", "python loop (numpy/scipy)"} <= {r["series"] for r in rows}
    assert all(np.isfinite(float(r["gflops"])) and float(r["gflops"]) > 0 for r in rows)
