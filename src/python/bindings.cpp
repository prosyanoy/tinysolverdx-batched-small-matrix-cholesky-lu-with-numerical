#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <stdexcept>
#include <vector>
#include <string>

#include "tinysolverdx/tinysolverdx.hpp"

namespace py = pybind11;
using namespace tsdx;

namespace {

Precision parse_precision(const std::string& s) {
  if (s == "fp32") return Precision::FP32;
  if (s == "fp64") return Precision::FP64;
  if (s == "fp16") return Precision::FP16;
  if (s == "bf16") return Precision::BF16;
  throw std::invalid_argument("precision must be one of fp32, fp64, fp16, bf16");
}
Backend parse_backend(const std::string& s) {
  if (s == "auto") return Backend::Auto;
  if (s == "naive") return Backend::Naive;
  if (s == "scalar") return Backend::Scalar;
  if (s == "simd") return Backend::Simd;
  if (s == "cuda") return Backend::Cuda;
  throw std::invalid_argument("backend must be one of auto, naive, scalar, simd, cuda");
}

template <class S> using Arr = py::array_t<S, py::array::c_style>;

template <class S>
void check_square(const Arr<S>& a) {
  if (a.ndim() != 3 || a.shape(1) != a.shape(2)) throw std::invalid_argument("expected array of shape (batch, n, n)");
}

template <class S>
py::tuple py_cholesky(Arr<S> A, const std::string& prec, const std::string& be) {
  check_square<S>(A);
  const int n = static_cast<int>(A.shape(1));
  const py::ssize_t batch = A.shape(0);
  Arr<S> out(A.request().shape);
  std::copy(A.data(), A.data() + A.size(), out.mutable_data());
  py::array_t<int> info(std::vector<py::ssize_t>{batch});
  const Precision p = parse_precision(prec);
  const Backend b = parse_backend(be);
  {
    py::gil_scoped_release rel;
    cholesky<S>(p, out.mutable_data(), n, batch, info.mutable_data(), b);
  }
  return py::make_tuple(out, info);
}

template <class S>
py::tuple py_lu(Arr<S> A, const std::string& prec, const std::string& be) {
  check_square<S>(A);
  const int n = static_cast<int>(A.shape(1));
  const py::ssize_t batch = A.shape(0);
  Arr<S> out(A.request().shape);
  std::copy(A.data(), A.data() + A.size(), out.mutable_data());
  py::array_t<int> piv({batch, static_cast<py::ssize_t>(n)});
  py::array_t<int> info(std::vector<py::ssize_t>{batch});
  const Precision p = parse_precision(prec);
  const Backend b = parse_backend(be);
  {
    py::gil_scoped_release rel;
    lu<S>(p, out.mutable_data(), n, batch, piv.mutable_data(), info.mutable_data(), b);
  }
  return py::make_tuple(out, piv, info);
}

template <class S>
Arr<S> py_cholesky_solve(Arr<S> L, Arr<S> B, const std::string& prec) {
  check_square<S>(L);
  if (B.ndim() != 3 || B.shape(0) != L.shape(0) || B.shape(1) != L.shape(1)) throw std::invalid_argument("B must be (batch, n, nrhs)");
  Arr<S> out(B.request().shape);
  std::copy(B.data(), B.data() + B.size(), out.mutable_data());
  const Precision p = parse_precision(prec);
  const int n = static_cast<int>(L.shape(1)), nrhs = static_cast<int>(B.shape(2));
  const py::ssize_t batch = L.shape(0);
  {
    py::gil_scoped_release rel;
    cholesky_solve<S>(p, L.data(), n, batch, out.mutable_data(), nrhs);
  }
  return out;
}

template <class S>
Arr<S> py_lu_solve(Arr<S> LU, py::array_t<int, py::array::c_style> piv, Arr<S> B, const std::string& prec) {
  check_square<S>(LU);
  if (B.ndim() != 3 || B.shape(0) != LU.shape(0) || B.shape(1) != LU.shape(1)) throw std::invalid_argument("B must be (batch, n, nrhs)");
  if (piv.ndim() != 2 || piv.shape(0) != LU.shape(0) || piv.shape(1) != LU.shape(1)) throw std::invalid_argument("piv must be (batch, n)");
  Arr<S> out(B.request().shape);
  std::copy(B.data(), B.data() + B.size(), out.mutable_data());
  const Precision p = parse_precision(prec);
  const int n = static_cast<int>(LU.shape(1)), nrhs = static_cast<int>(B.shape(2));
  const py::ssize_t batch = LU.shape(0);
  {
    py::gil_scoped_release rel;
    lu_solve<S>(p, LU.data(), piv.data(), n, batch, out.mutable_data(), nrhs);
  }
  return out;
}

template <class S>
Arr<S> py_round(Arr<S> x, const std::string& prec) {
  Arr<S> out(x.request().shape);
  std::copy(x.data(), x.data() + x.size(), out.mutable_data());
  round_to<S>(parse_precision(prec), out.mutable_data(), out.size());
  return out;
}

template <class S>
void bind_dtype(py::module_& m, const std::string& sfx) {
  m.def(("cholesky_" + sfx).c_str(), &py_cholesky<S>, py::arg("A"), py::arg("precision") = "fp32", py::arg("backend") = "auto");
  m.def(("lu_" + sfx).c_str(), &py_lu<S>, py::arg("A"), py::arg("precision") = "fp32", py::arg("backend") = "auto");
  m.def(("cholesky_solve_" + sfx).c_str(), &py_cholesky_solve<S>, py::arg("L"), py::arg("B"), py::arg("precision") = "fp32");
  m.def(("lu_solve_" + sfx).c_str(), &py_lu_solve<S>, py::arg("LU"), py::arg("piv"), py::arg("B"), py::arg("precision") = "fp32");
  m.def(("round_to_" + sfx).c_str(), &py_round<S>, py::arg("x"), py::arg("precision"));
}

using D = py::array_t<double, py::array::c_style>;

void check_vec(const D& A, const D& v, const char* what) {
  if (A.ndim() != 3 || A.shape(1) != A.shape(2)) throw std::invalid_argument("A must be (batch, n, n)");
  if (v.ndim() != 2 || v.shape(0) != A.shape(0) || v.shape(1) != A.shape(1))
    throw std::invalid_argument(std::string(what) + " must be (batch, n)");
}

}  // namespace

PYBIND11_MODULE(_core, m) {
  m.doc() = "TinySolverDx native core: batched small-matrix Cholesky/LU";
  bind_dtype<float>(m, "f32");
  bind_dtype<double>(m, "f64");

  m.def("has_cuda", &has_cuda);
  m.def("has_openmp", &has_openmp);
  m.def("num_threads", &num_threads);
  m.def("set_num_threads", &set_num_threads);
  m.def("unit_roundoff", [](const std::string& p) { return unit_roundoff(parse_precision(p)); });

  // ---- error metrics (float64 in, long-double arithmetic inside) ----
  m.def("chol_backward_error", [](D A, D L) {
    if (A.ndim() != 3 || A.shape(1) != A.shape(2) || L.ndim() != 3 || L.size() != A.size()) throw std::invalid_argument("shape mismatch");
    py::array_t<double> out(std::vector<py::ssize_t>{A.shape(0)});
    chol_backward_error(A.data(), L.data(), (int)A.shape(1), A.shape(0), out.mutable_data());
    return out;
  });
  m.def("lu_backward_error", [](D A, D LU, py::array_t<int, py::array::c_style> piv) {
    if (A.ndim() != 3 || A.shape(1) != A.shape(2) || LU.size() != A.size() || piv.size() != A.shape(0) * A.shape(1)) throw std::invalid_argument("shape mismatch");
    py::array_t<double> out(std::vector<py::ssize_t>{A.shape(0)});
    lu_backward_error(A.data(), LU.data(), piv.data(), (int)A.shape(1), A.shape(0), out.mutable_data());
    return out;
  });
  m.def("solve_backward_error", [](D A, D x, D b) {
    check_vec(A, x, "x"); check_vec(A, b, "b");
    py::array_t<double> out(std::vector<py::ssize_t>{A.shape(0)});
    solve_backward_error(A.data(), x.data(), b.data(), (int)A.shape(1), A.shape(0), out.mutable_data());
    return out;
  });
  m.def("solve_forward_error", [](D A, D x, D b) {
    check_vec(A, x, "x"); check_vec(A, b, "b");
    py::array_t<double> out(std::vector<py::ssize_t>{A.shape(0)});
    solve_forward_error(A.data(), x.data(), b.data(), (int)A.shape(1), A.shape(0), out.mutable_data());
    return out;
  });
  m.def("reference_solve", [](D A, D b) {
    check_vec(A, b, "b");
    D x(b.request().shape);
    reference_solve(A.data(), b.data(), (int)A.shape(1), A.shape(0), x.mutable_data());
    return x;
  });

  m.def("measure_peak_gflops", [](const std::string& p) { return measure_peak_gflops(parse_precision(p)); });
  m.def("measure_bandwidth_gbs", &measure_bandwidth_gbs);
}
