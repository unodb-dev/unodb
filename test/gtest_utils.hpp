// Copyright 2021-2026 UnoDB contributors
#ifndef UNODB_DETAIL_GTEST_UTILS_HPP
#define UNODB_DETAIL_GTEST_UTILS_HPP

/// \file
/// Google Test wrapper macros.
///
/// \ingroup test-internals
///
/// These macros wrap Google Test functionality while suppressing various
/// compiler and static analysis warnings. Use these macros instead of direct
/// Google Test macros in UnoDB tests.

// Should be the first include
#include "global.hpp"

#include <gtest/gtest.h>

/// \addtogroup test-internals
/// \{

/// \name Google Test wrapper macros
/// \{

/// Wrapper for Google Test `TYPED_TEST_SUITE` macro.
// Because Google thinks
// error: must specify at least one argument for '...' parameter of variadic
// macro
//       [-Werror,-Wgnu-zero-variadic-macro-arguments]
// TYPED_TEST_CASE(ARTCorrectnessTest, ARTTypes);
//                                             ^
// is not a bug: https://github.com/google/googletest/issues/2271
#define UNODB_TYPED_TEST_SUITE(Suite, Types)                                \
  UNODB_DETAIL_DISABLE_CLANG_WARNING("-Wgnu-zero-variadic-macro-arguments") \
  TYPED_TEST_SUITE(Suite, Types);                                           \
  UNODB_DETAIL_RESTORE_CLANG_WARNINGS()

/// Wrapper for Google Test `ASSERT_DEATH` macro.
#define UNODB_ASSERT_DEATH(statement, regex)                       \
  do {                                                             \
    UNODB_DETAIL_DISABLE_CLANG_WARNING("-Wused-but-marked-unused") \
    UNODB_DETAIL_DISABLE_CLANG_WARNING("-Wcovered-switch-default") \
    UNODB_DETAIL_DISABLE_MSVC_WARNING(6326)                        \
    UNODB_DETAIL_DISABLE_MSVC_WARNING(26818)                       \
    ASSERT_DEATH(statement, regex);                                \
    UNODB_DETAIL_RESTORE_MSVC_WARNINGS()                           \
    UNODB_DETAIL_RESTORE_MSVC_WARNINGS()                           \
    UNODB_DETAIL_RESTORE_CLANG_WARNINGS()                          \
    UNODB_DETAIL_RESTORE_CLANG_WARNINGS()                          \
  } while (0)

/// Wrapper for Google Test `TEST` macro.
#define UNODB_TEST(Suite, Test)            \
  UNODB_DETAIL_DISABLE_MSVC_WARNING(26409) \
  UNODB_DETAIL_DISABLE_MSVC_WARNING(26426) \
  UNODB_DETAIL_DISABLE_MSVC_WARNING(26440) \
  UNODB_DETAIL_DISABLE_MSVC_WARNING(26455) \
  TEST(Suite, Test)                        \
  UNODB_DETAIL_RESTORE_MSVC_WARNINGS()     \
  UNODB_DETAIL_RESTORE_MSVC_WARNINGS()     \
  UNODB_DETAIL_RESTORE_MSVC_WARNINGS()     \
  UNODB_DETAIL_RESTORE_MSVC_WARNINGS()

/// Wrapper for Google Test `TEST_F` macro.
#define UNODB_TEST_F(Suite, Test)          \
  UNODB_DETAIL_DISABLE_MSVC_WARNING(26409) \
  UNODB_DETAIL_DISABLE_MSVC_WARNING(26426) \
  UNODB_DETAIL_DISABLE_MSVC_WARNING(26455) \
  TEST_F(Suite, Test)                      \
  UNODB_DETAIL_RESTORE_MSVC_WARNINGS()     \
  UNODB_DETAIL_RESTORE_MSVC_WARNINGS()     \
  UNODB_DETAIL_RESTORE_MSVC_WARNINGS()

/// Wrapper for Google Test `TYPED_TEST` macro.
#define UNODB_TYPED_TEST(Suite, Test)      \
  UNODB_DETAIL_DISABLE_MSVC_WARNING(26426) \
  UNODB_DETAIL_DISABLE_MSVC_WARNING(26440) \
  TYPED_TEST(Suite, Test)                  \
  UNODB_DETAIL_RESTORE_MSVC_WARNINGS()     \
  UNODB_DETAIL_RESTORE_MSVC_WARNINGS()

/// Wrapper for Google Test `ASSERT_EQ` macro.
#define UNODB_ASSERT_EQ(x, y)                \
  do {                                       \
    UNODB_DETAIL_DISABLE_MSVC_WARNING(6326)  \
    UNODB_DETAIL_DISABLE_MSVC_WARNING(26818) \
    ASSERT_EQ((x), (y));                     \
    UNODB_DETAIL_RESTORE_MSVC_WARNINGS()     \
    UNODB_DETAIL_RESTORE_MSVC_WARNINGS()     \
  } while (0)

/// Wrapper for Google Test `ASSERT_NEAR` macro.
#define UNODB_ASSERT_NEAR(x, y, e)           \
  do {                                       \
    UNODB_DETAIL_DISABLE_MSVC_WARNING(6326)  \
    UNODB_DETAIL_DISABLE_MSVC_WARNING(26818) \
    ASSERT_NEAR((x), (y), (e));              \
    UNODB_DETAIL_RESTORE_MSVC_WARNINGS()     \
    UNODB_DETAIL_RESTORE_MSVC_WARNINGS()     \
  } while (0)

/// Wrapper for Google Test `ASSERT_FALSE` macro.
#define UNODB_ASSERT_FALSE(cond)             \
  do {                                       \
    UNODB_DETAIL_DISABLE_MSVC_WARNING(6326)  \
    UNODB_DETAIL_DISABLE_MSVC_WARNING(26818) \
    ASSERT_FALSE(cond);                      \
    UNODB_DETAIL_RESTORE_MSVC_WARNINGS()     \
    UNODB_DETAIL_RESTORE_MSVC_WARNINGS()     \
  } while (0)

/// Wrapper for Google Test `ASSERT_GT` macro.
#define UNODB_ASSERT_GT(val1, val2)          \
  do {                                       \
    UNODB_DETAIL_DISABLE_MSVC_WARNING(6326)  \
    UNODB_DETAIL_DISABLE_MSVC_WARNING(26818) \
    ASSERT_GT((val1), (val2));               \
    UNODB_DETAIL_RESTORE_MSVC_WARNINGS()     \
    UNODB_DETAIL_RESTORE_MSVC_WARNINGS()     \
  } while (0)

/// Wrapper for Google Test `ASSERT_LE` macro.
#define UNODB_ASSERT_LE(val1, val2)          \
  do {                                       \
    UNODB_DETAIL_DISABLE_MSVC_WARNING(6326)  \
    UNODB_DETAIL_DISABLE_MSVC_WARNING(26818) \
    ASSERT_LE((val1), (val2));               \
    UNODB_DETAIL_RESTORE_MSVC_WARNINGS()     \
    UNODB_DETAIL_RESTORE_MSVC_WARNINGS()     \
  } while (0)

/// Wrapper for Google Test `ASSERT_LT` macro.
#define UNODB_ASSERT_LT(val1, val2)          \
  do {                                       \
    UNODB_DETAIL_DISABLE_MSVC_WARNING(6326)  \
    UNODB_DETAIL_DISABLE_MSVC_WARNING(26818) \
    ASSERT_LT((val1), (val2));               \
    UNODB_DETAIL_RESTORE_MSVC_WARNINGS()     \
    UNODB_DETAIL_RESTORE_MSVC_WARNINGS()     \
  } while (0)

/// Wrapper for Google Test `ASSERT_THAT` macro.
#define UNODB_ASSERT_THAT(value, matcher)    \
  do {                                       \
    UNODB_DETAIL_DISABLE_MSVC_WARNING(6326)  \
    UNODB_DETAIL_DISABLE_MSVC_WARNING(26818) \
    ASSERT_THAT((value), (matcher));         \
    UNODB_DETAIL_RESTORE_MSVC_WARNINGS()     \
    UNODB_DETAIL_RESTORE_MSVC_WARNINGS()     \
  } while (0)

/// Wrapper for Google Test `ASSERT_THROW` macro.
#define UNODB_ASSERT_THROW(statement, expected_exception) \
  do {                                                    \
    UNODB_DETAIL_DISABLE_MSVC_WARNING(6326)               \
    UNODB_DETAIL_DISABLE_MSVC_WARNING(26818)              \
    ASSERT_THROW(statement, expected_exception);          \
    UNODB_DETAIL_RESTORE_MSVC_WARNINGS()                  \
    UNODB_DETAIL_RESTORE_MSVC_WARNINGS()                  \
  } while (0)

/// Wrapper for Google Test `ASSERT_NO_THROW` macro.
#define UNODB_ASSERT_NO_THROW(statement)     \
  do {                                       \
    UNODB_DETAIL_DISABLE_MSVC_WARNING(6326)  \
    UNODB_DETAIL_DISABLE_MSVC_WARNING(26818) \
    ASSERT_NO_THROW(statement);              \
    UNODB_DETAIL_RESTORE_MSVC_WARNINGS()     \
    UNODB_DETAIL_RESTORE_MSVC_WARNINGS()     \
  } while (0)

/// Wrapper for Google Test `ASSERT_TRUE` macro.
#define UNODB_ASSERT_TRUE(cond)              \
  do {                                       \
    UNODB_DETAIL_DISABLE_MSVC_WARNING(6326)  \
    UNODB_DETAIL_DISABLE_MSVC_WARNING(26818) \
    ASSERT_TRUE(cond);                       \
    UNODB_DETAIL_RESTORE_MSVC_WARNINGS()     \
    UNODB_DETAIL_RESTORE_MSVC_WARNINGS()     \
  } while (0)

/// Wrapper for Google Test `EXPECT_EQ` macro.
// Do not wrap in a block to support streaming to EXPECT_EQ. Happens to be OK
// because the warning macros are not statements.
#define UNODB_EXPECT_EQ(x, y)              \
  UNODB_DETAIL_DISABLE_MSVC_WARNING(6326)  \
  UNODB_DETAIL_DISABLE_MSVC_WARNING(26818) \
  EXPECT_EQ((x), (y))                      \
  UNODB_DETAIL_RESTORE_MSVC_WARNINGS()     \
  UNODB_DETAIL_RESTORE_MSVC_WARNINGS()

/// Wrapper for Google Test `EXPECT_NE` macro.
#define UNODB_EXPECT_NE(x, y)                \
  do {                                       \
    UNODB_DETAIL_DISABLE_MSVC_WARNING(6326)  \
    UNODB_DETAIL_DISABLE_MSVC_WARNING(26818) \
    EXPECT_NE((x), (y));                     \
    UNODB_DETAIL_RESTORE_MSVC_WARNINGS()     \
    UNODB_DETAIL_RESTORE_MSVC_WARNINGS()     \
  } while (0)

/// Wrapper for Google Test `EXPECT_GT` macro.
#define UNODB_EXPECT_GT(x, y)                \
  do {                                       \
    UNODB_DETAIL_DISABLE_MSVC_WARNING(6326)  \
    UNODB_DETAIL_DISABLE_MSVC_WARNING(26818) \
    EXPECT_GT((x), (y));                     \
    UNODB_DETAIL_RESTORE_MSVC_WARNINGS()     \
    UNODB_DETAIL_RESTORE_MSVC_WARNINGS()     \
  } while (0)

/// Wrapper for Google Test `EXPECT_LT` macro.
#define UNODB_EXPECT_LT(x, y)                \
  do {                                       \
    UNODB_DETAIL_DISABLE_MSVC_WARNING(6326)  \
    UNODB_DETAIL_DISABLE_MSVC_WARNING(26818) \
    EXPECT_LT((x), (y));                     \
    UNODB_DETAIL_RESTORE_MSVC_WARNINGS()     \
    UNODB_DETAIL_RESTORE_MSVC_WARNINGS()     \
  } while (0)

/// Wrapper for Google Test `EXPECT_TRUE` macro.
// Do not wrap in a block to support streaming to EXPECT_TRUE. Happens to be OK
// because the warning macros are not statements.
#define UNODB_EXPECT_TRUE(cond)            \
  UNODB_DETAIL_DISABLE_MSVC_WARNING(6326)  \
  UNODB_DETAIL_DISABLE_MSVC_WARNING(26818) \
  EXPECT_TRUE(cond)                        \
  UNODB_DETAIL_RESTORE_MSVC_WARNINGS()     \
  UNODB_DETAIL_RESTORE_MSVC_WARNINGS()

/// Wrapper for Google Test `EXPECT_FALSE` macro.
#define UNODB_EXPECT_FALSE(cond)             \
  do {                                       \
    UNODB_DETAIL_DISABLE_MSVC_WARNING(6326)  \
    UNODB_DETAIL_DISABLE_MSVC_WARNING(26818) \
    EXPECT_FALSE(cond);                      \
    UNODB_DETAIL_RESTORE_MSVC_WARNINGS()     \
    UNODB_DETAIL_RESTORE_MSVC_WARNINGS()     \
  } while (0)

/// \}

/// \}

#endif  // UNODB_DETAIL_GTEST_UTILS_HPP
