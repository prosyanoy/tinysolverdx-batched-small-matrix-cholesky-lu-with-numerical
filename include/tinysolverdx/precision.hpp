// Precision descriptors and CPU emulation of fp16 / bfloat16 arithmetic.
//
// Emulated types store a float that is always exactly representable in the
// target format; every arithmetic operation rounds its result (round to
// nearest, ties to even), which matches hardware that computes in the narrow
// format without a wider accumulator.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <type_traits>

#if defined(__CUDACC__)
#define TSDX_HD __host__ __device__
#else
#define TSDX_HD
#endif

#if defined(__GNUC__) || defined(__clang__)
#define TSDX_INLINE inline __attribute__((always_inline))
#define TSDX_NOINLINE __attribute__((noinline))
#define TSDX_RESTRICT __restrict__
#else
#define TSDX_INLINE inline
#define TSDX_NOINLINE
#define TSDX_RESTRICT
#endif

namespace tsdx {

enum class Precision { FP32, FP64, FP16, BF16 };
enum class Backend { Auto, Naive, Scalar, Simd, Cuda };

namespace detail {

inline std::uint32_t f2u(float x) { std::uint32_t u; std::memcpy(&u, &x, 4); return u; }
inline float u2f(std::uint32_t u) { float x; std::memcpy(&x, &u, 4); return x; }

// Round a float to the nearest IEEE binary16 value (RNE, subnormals, overflow -> inf).
inline float round_fp16(float x) {
  const std::uint32_t u = f2u(x);
  const std::uint32_t sign = u & 0x80000000u;
  const std::uint32_t a = u & 0x7fffffffu;
  if (a >= 0x7f800000u) return x;                       // inf / NaN
  const float ax = u2f(a);
  if (ax >= 65520.0f) return u2f(sign | 0x7f800000u);   // rounds past max finite 65504
  if (ax < 6.103515625e-05f) {                          // fp16 subnormal range: quantum 2^-24
    const float q = 5.9604644775390625e-08f;
    return u2f(sign | f2u(std::nearbyint(ax / q) * q));
  }
  std::uint32_t r = a + 0xfffu + ((a >> 13) & 1u);      // drop 13 mantissa bits, RNE
  r &= ~0x1fffu;
  return u2f(sign | r);
}

// Round a float to the nearest bfloat16 value (RNE).
inline float round_bf16(float x) {
  const std::uint32_t u = f2u(x);
  const std::uint32_t a = u & 0x7fffffffu;
  if (a > 0x7f800000u) return x;                        // NaN
  std::uint32_t r = a + 0x7fffu + ((a >> 16) & 1u);
  r &= ~0xffffu;
  return u2f((u & 0x80000000u) | r);
}

}  // namespace detail

enum class EmuFormat { Half, BFloat16 };

template <EmuFormat F>
struct Emu {
  float v = 0.0f;

  static float round(float x) {
    return F == EmuFormat::Half ? detail::round_fp16(x) : detail::round_bf16(x);
  }
  Emu() = default;
  explicit Emu(float x) : v(round(x)) {}
  explicit Emu(double x) : v(round(static_cast<float>(x))) {}
  explicit Emu(int x) : v(round(static_cast<float>(x))) {}
  float to_float() const { return v; }

  friend Emu operator+(Emu a, Emu b) { return Emu(a.v + b.v); }
  friend Emu operator-(Emu a, Emu b) { return Emu(a.v - b.v); }
  friend Emu operator*(Emu a, Emu b) { return Emu(a.v * b.v); }
  friend Emu operator/(Emu a, Emu b) { return Emu(a.v / b.v); }
  friend Emu operator-(Emu a) { Emu r; r.v = -a.v; return r; }
  Emu& operator+=(Emu b) { *this = *this + b; return *this; }
  Emu& operator-=(Emu b) { *this = *this - b; return *this; }
  Emu& operator*=(Emu b) { *this = *this * b; return *this; }
  Emu& operator/=(Emu b) { *this = *this / b; return *this; }
  friend bool operator<(Emu a, Emu b) { return a.v < b.v; }
  friend bool operator>(Emu a, Emu b) { return a.v > b.v; }
  friend bool operator==(Emu a, Emu b) { return a.v == b.v; }
  friend bool operator!=(Emu a, Emu b) { return a.v != b.v; }
};

using Half = Emu<EmuFormat::Half>;
using BFloat16 = Emu<EmuFormat::BFloat16>;

// Type-generic helpers used by the kernels (usable in device code for float/double).
template <class T> TSDX_HD TSDX_INLINE T t_abs(T x) { return x < T(0) ? -x : x; }
template <class T> TSDX_HD TSDX_INLINE T t_sqrt(T x) { return std::sqrt(x); }
template <EmuFormat F> inline Emu<F> t_sqrt(Emu<F> x) { return Emu<F>(std::sqrt(x.v)); }

template <class T> struct PrecisionOf;
template <> struct PrecisionOf<float> { static constexpr Precision value = Precision::FP32; };
template <> struct PrecisionOf<double> { static constexpr Precision value = Precision::FP64; };
template <> struct PrecisionOf<Half> { static constexpr Precision value = Precision::FP16; };
template <> struct PrecisionOf<BFloat16> { static constexpr Precision value = Precision::BF16; };

template <class T> struct is_native : std::false_type {};
template <> struct is_native<float> : std::true_type {};
template <> struct is_native<double> : std::true_type {};

// Unit roundoff u = 2^-p for each format.
inline double unit_roundoff(Precision p) {
  switch (p) {
    case Precision::FP32: return 5.960464477539063e-08;
    case Precision::FP64: return 1.1102230246251565e-16;
    case Precision::FP16: return 4.8828125e-04;
    case Precision::BF16: return 3.90625e-03;
  }
  return 0.0;
}

}  // namespace tsdx
