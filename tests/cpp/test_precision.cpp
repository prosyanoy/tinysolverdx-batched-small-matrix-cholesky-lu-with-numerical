#include <gtest/gtest.h>

#include <cmath>
#include <limits>

#include "tinysolverdx/tinysolverdx.hpp"

using namespace tsdx;

TEST(Fp16, RoundsToNearestEven) {
  EXPECT_EQ(detail::round_fp16(1.0f), 1.0f);
  EXPECT_EQ(detail::round_fp16(1.0f + 1.0f / 2048.0f), 1.0f);               // tie -> even (1.0)
  EXPECT_EQ(detail::round_fp16(1.0f + 3.0f / 2048.0f), 1.0f + 4.0f / 2048.0f);  // tie -> even (up)
  EXPECT_EQ(detail::round_fp16(65504.0f), 65504.0f);
  EXPECT_TRUE(std::isinf(detail::round_fp16(65520.0f)));
  EXPECT_EQ(detail::round_fp16(6.0e-8f), 5.9604644775390625e-08f);          // smallest subnormal
  EXPECT_EQ(detail::round_fp16(1.0e-9f), 0.0f);
  EXPECT_EQ(detail::round_fp16(-0.1f), -0.0999755859375f);
  EXPECT_TRUE(std::isnan(detail::round_fp16(std::numeric_limits<float>::quiet_NaN())));
}

TEST(Bf16, RoundsToNearestEven) {
  EXPECT_EQ(detail::round_bf16(1.0f), 1.0f);
  EXPECT_EQ(detail::round_bf16(1.0f + 1.0f / 256.0f), 1.0f);  // tie -> even
  EXPECT_EQ(detail::round_bf16(1.0f + 3.0f / 256.0f), 1.0f + 4.0f / 256.0f);
  EXPECT_EQ(detail::round_bf16(3.14159265f), 3.140625f);
  EXPECT_TRUE(std::isinf(detail::round_bf16(std::numeric_limits<float>::max())));
}

TEST(Emu, ArithmeticRoundsEachOp) {
  Half a(1.0f), b(1.0f / 4096.0f);
  EXPECT_EQ((a + b).v, 1.0f);  // below half-ulp of 1.0 in fp16
  BFloat16 c(1.0f), d(1.0f / 512.0f);
  EXPECT_EQ((c + d).v, 1.0f);
  EXPECT_EQ(Half(0.5f) * Half(0.5f), Half(0.25f));
}

TEST(Precision, UnitRoundoff) {
  EXPECT_DOUBLE_EQ(unit_roundoff(Precision::FP32), std::ldexp(1.0, -24));
  EXPECT_DOUBLE_EQ(unit_roundoff(Precision::FP16), std::ldexp(1.0, -11));
  EXPECT_DOUBLE_EQ(unit_roundoff(Precision::BF16), std::ldexp(1.0, -8));
}
