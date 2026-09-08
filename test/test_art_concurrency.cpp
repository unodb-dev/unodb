// Copyright 2021-2026 UnoDB contributors

// Should be the first include
#include "global.hpp"

// IWYU pragma: no_include <string>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <random>
#include <span>
#include <thread>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include "art_common.hpp"
#include "art_test_data.hpp"
#include "assert.hpp"
#include "db_test_utils.hpp"
#include "gtest_utils.hpp"
#include "olc_art.hpp"
#include "qsbr.hpp"
#include "qsbr_test_utils.hpp"

#ifndef NDEBUG
#include <optional>
#include <type_traits>
#include <utility>

#include <gtest/gtest-spi.h>

#include "art_internal_impl.hpp"
#include "node_type.hpp"
#include "sync.hpp"
#include "sync_point_test_utils.hpp"
#include "test_tuple_heap.hpp"
#include "thread_sync.hpp"
using unodb::test::release_after_missed_park;
using unodb::test::run_parked_interleaving;
using unodb::test::sync_point_guard;
#endif

// MSVC C26815 false positive on every `copy_key(make_*(enc, ...), buf)` call
// below, including inside `make_chain_key`: the analyzer sees the `key_view`
// returned by the `make_*` builder die at the end of the full expression and
// concludes the result dangles.  It does not model that `copy_key` copies those
// bytes into `buf` and returns a view over `buf`, severing the dependency on
// its `kv` argument - which is exactly why `kv` is left unannotated while `buf`
// carries UNODB_DETAIL_LIFETIMEBOUND.
UNODB_DETAIL_DISABLE_MSVC_WARNING(26815)

namespace {

/// One-byte values 1, 2, ..., \a N, one per key of a test, so that a lookup
/// that reaches another key's keyless leaf finds the wrong value.
template <std::size_t N>
[[nodiscard]] constexpr std::array<std::array<std::byte, 1>, N>
distinct_values() noexcept {
  static_assert(N <= 255);
  std::array<std::array<std::byte, 1>, N> vals{};
  for (std::size_t i = 0; i < N; ++i)
    vals[i][0] = static_cast<std::byte>(i + 1);
  return vals;
}

[[nodiscard]] constexpr bool odd(std::uint64_t x) noexcept {
  return static_cast<bool>(x % 2);
}

using unodb::test_data::copy_key;
using unodb::test_data::decode;
using unodb::test_data::make_key;
using unodb::test_data::make_short_key;
using unodb::test_data::test_values;

template <class Db>
class ARTConcurrencyTest : public ::testing::Test {
 public:
  UNODB_DETAIL_DISABLE_MSVC_WARNING(26447)
  ~ARTConcurrencyTest() noexcept override {
    if constexpr (unodb::test::is_olc_db<Db>) {
      unodb::this_thread().quiescent();
      unodb::test::expect_idle_qsbr();
    }
  }
  UNODB_DETAIL_RESTORE_MSVC_WARNINGS()

 protected:
  // NOLINTNEXTLINE(bugprone-exception-escape)
  ARTConcurrencyTest() noexcept {
    if constexpr (unodb::test::is_olc_db<Db>) unodb::test::expect_idle_qsbr();
  }

  // TestFn is void(unodb::test::tree_verifier<Db> *verifier, std::size_t
  // thread_i, std::size_t ops_per_thread)
  template <std::size_t ThreadCount, std::size_t OpsPerThread, typename TestFn>
  void parallel_test(TestFn test_function) {
    if constexpr (unodb::test::is_olc_db<Db>) unodb::this_thread().qsbr_pause();

    std::array<unodb::test::thread<Db>, ThreadCount> threads;
    for (decltype(ThreadCount) i = 0; i < ThreadCount; ++i) {
      threads[i] =
          unodb::test::thread<Db>{test_function, &verifier, i, OpsPerThread};
    }
    for (auto& t : threads) {
      t.join();
    }

    if constexpr (unodb::test::is_olc_db<Db>)
      unodb::this_thread().qsbr_resume();
  }

  template <unsigned PreinsertLimit, std::size_t ThreadCount,
            std::size_t OpsPerThread>
  void key_range_op_test() {
    verifier.insert_key_range(0, PreinsertLimit, true);

    parallel_test<ThreadCount, OpsPerThread>(key_range_op_thread);
  }

  static void parallel_insert_thread(unodb::test::tree_verifier<Db>* verifier,
                                     std::size_t thread_i,
                                     std::size_t ops_per_thread) {
    verifier->insert_preinserted_key_range(thread_i * ops_per_thread,
                                           ops_per_thread);
  }

  static void parallel_remove_thread(unodb::test::tree_verifier<Db>* verifier,
                                     std::size_t thread_i,
                                     std::size_t ops_per_thread) {
    const auto start_key = thread_i * ops_per_thread;
    for (decltype(ops_per_thread) i = 0; i < ops_per_thread; ++i) {
      verifier->remove(start_key + i, true);
    }
  }

  // test helper for scan() verification.
  static void do_scan_verification(unodb::test::tree_verifier<Db>* verifier,
                                   std::uint64_t key) {
    const bool fwd = odd(key);  // select scan direction
    const auto k0 = (key > 100) ? (key - 100) : key;
    const auto k1 = key + 100;
    uint64_t n = 0;
    uint64_t sum = 0;
    std::uint64_t prior{};

    UNODB_DETAIL_DISABLE_MSVC_WARNING(26440)
    auto fn = [&n, &sum, &fwd, &k0, &k1,
               &prior](const unodb::visitor<typename Db::iterator>& v) {
      n++;
      const auto& akey = decode(v.get_key());  // actual visited key.
      sum += akey;
      const auto expected =  // Note: same value formula as insert().
          test_values[akey % test_values.size()];
      const auto actual = v.get_value();
      // LCOV_EXCL_START
      UNODB_EXPECT_TRUE(std::ranges::equal(actual, expected));
      std::ignore = v.get_value();
      if (fwd) {  // [k0,k1) -- k0 is from_key, k1 is to_key
        UNODB_EXPECT_TRUE(akey >= k0 && akey < k1)
            << "fwd=" << fwd << ", key=" << akey << ", k0=" << k0
            << ", k1=" << k1;
      } else {  // (k1,k0]
        UNODB_EXPECT_TRUE(akey > k0 && akey <= k1)
            << "fwd=" << fwd << ", key=" << akey << ", k0=" << k0
            << ", k1=" << k1;
      }
      if (n > 1) {
        UNODB_EXPECT_TRUE(fwd ? (akey > prior) : (akey < prior))
            << "fwd=" << fwd << ", prior=" << prior << ", key=" << akey
            << ", k0=" << k0 << ", k1=" << k1;
      }
      // LCOV_EXCL_STOP
      prior = akey;
      return false;
    };
    UNODB_DETAIL_RESTORE_MSVC_WARNINGS()

    if (fwd) {
      verifier->scan_range(k0, k1, fn);
    } else {
      verifier->scan_range(k1, k0, fn);
    }
    if constexpr (unodb::test::is_olc_db<Db>) {
      unodb::this_thread().quiescent();
    }
  }

  static void key_range_op_thread(unodb::test::tree_verifier<Db>* verifier,
                                  std::size_t thread_i,
                                  std::size_t ops_per_thread) {
    constexpr auto ntasks = 4;  // Note: 4 to enable scan tests.
    std::uint64_t key = thread_i / ntasks * ntasks;
    for (decltype(ops_per_thread) i = 0; i < ops_per_thread; ++i) {
      switch (thread_i % ntasks) {
        case 0: /* insert (same value formula as insert_key_range!) */
          verifier->try_insert(key, test_values[key % test_values.size()]);
          break;
        case 1: /* remove */
          verifier->try_remove(key);
          break;
        case 2: /* get */
          verifier->try_get(key);
          break;
        case 3: /* scan */
          do_scan_verification(verifier, key);
          break;
          // LCOV_EXCL_START
        default:
          UNODB_DETAIL_CANNOT_HAPPEN();
          // LCOV_EXCL_STOP
      }
      key++;
    }
  }

  static void random_op_thread(unodb::test::tree_verifier<Db>* verifier,
                               std::size_t thread_i,
                               std::size_t ops_per_thread) {
    std::random_device rd;
    std::mt19937 gen{rd()};
    std::geometric_distribution<std::uint64_t> key_generator{0.5};
    constexpr auto ntasks = 4;  // Note: 4 to enable scan tests.
    for (decltype(ops_per_thread) i = 0; i < ops_per_thread; ++i) {
      const auto key{key_generator(gen)};
      switch (thread_i % ntasks) {
        case 0: /* insert (same value formula as insert_key_range!) */
          verifier->try_insert(key, test_values[key % test_values.size()]);
          break;
        case 1: /* remove */
          verifier->try_remove(key);
          break;
        case 2: /* get */
          verifier->try_get(key);
          break;
        case 3: /* scan */
          do_scan_verification(verifier, key);
          break;
          // LCOV_EXCL_START
        default:
          UNODB_DETAIL_CANNOT_HAPPEN();
          // LCOV_EXCL_STOP
      }
    }
  }

  unodb::test::tree_verifier<Db> verifier{true};

 public:
  ARTConcurrencyTest(const ARTConcurrencyTest<Db>&) = delete;
  ARTConcurrencyTest(ARTConcurrencyTest<Db>&&) = delete;
  ARTConcurrencyTest<Db>& operator=(const ARTConcurrencyTest<Db>&) = delete;
  ARTConcurrencyTest<Db>& operator=(ARTConcurrencyTest<Db>&&) = delete;
};

// TODO(thompsonbry) variable length keys - enable key_view variants (#8)
// once some critical bug fixes are resolved in master.
using ConcurrentARTTypes =
    ::testing::Types<unodb::test::u64_mutex_db, unodb::test::u64_olc_db
                     // unodb::test::key_view_mutex_db,
                     // unodb::test::key_view_olc_db
                     >;

UNODB_TYPED_TEST_SUITE(ARTConcurrencyTest, ConcurrentARTTypes)

UNODB_TYPED_TEST(ARTConcurrencyTest, ParallelInsertOneTree) {
  constexpr auto thread_count = 4;
  constexpr auto total_keys = 1024;
  constexpr auto ops_per_thread = total_keys / thread_count;

  this->verifier.preinsert_key_range_to_verifier_only(0, total_keys);
  this->template parallel_test<thread_count, ops_per_thread>(
      TestFixture::parallel_insert_thread);
  this->verifier.check_present_values();
}

UNODB_TYPED_TEST(ARTConcurrencyTest, ParallelTearDownOneTree) {
  constexpr auto thread_count = 8;
  constexpr auto total_keys = 2048;
  constexpr auto ops_per_thread = total_keys / thread_count;

  this->verifier.insert_key_range(0, total_keys);
  this->template parallel_test<thread_count, ops_per_thread>(
      TestFixture::parallel_remove_thread);
  this->verifier.assert_empty();
}

UNODB_TYPED_TEST(ARTConcurrencyTest, Node4ParallelOps) {
  this->template key_range_op_test<3, 8, 6>();
}

UNODB_TYPED_TEST(ARTConcurrencyTest, Node16ParallelOps) {
  this->template key_range_op_test<10, 8, 12>();
}

UNODB_TYPED_TEST(ARTConcurrencyTest, Node48ParallelOps) {
  this->template key_range_op_test<32, 8, 32>();
}

UNODB_TYPED_TEST(ARTConcurrencyTest, Node256ParallelOps) {
  this->template key_range_op_test<152, 8, 208>();
}

UNODB_TYPED_TEST(ARTConcurrencyTest, ParallelRandomInsertDeleteGetScan) {
  constexpr auto thread_count = 4;
  constexpr auto initial_keys = 128;
  constexpr auto ops_per_thread = 500;

  this->verifier.insert_key_range(0, initial_keys, true);
  this->template parallel_test<thread_count, ops_per_thread>(
      TestFixture::random_op_thread);
}

UNODB_TYPED_TEST(ARTConcurrencyTest,
                 DISABLED_MediumParallelRandomInsertDeleteGetScan) {
  constexpr auto thread_count = 4 * 3;
  constexpr auto initial_keys = 2048;
  constexpr auto ops_per_thread = 10'000;

  this->verifier.insert_key_range(0, initial_keys, true);
  this->template parallel_test<thread_count, ops_per_thread>(
      TestFixture::random_op_thread);
}

// A more challenging test using a smaller key range and the same
// number of threads and operations per thread.  The goal of this test
// is to try an increase coverage of the N256 case.
UNODB_TYPED_TEST(ARTConcurrencyTest,
                 DISABLED_ParallelRandomInsertDeleteGetScan2) {
  constexpr auto thread_count = 4 * 3;
  constexpr auto initial_keys = 152;
  constexpr auto ops_per_thread = 100'000;

  this->verifier.insert_key_range(0, initial_keys, true);
  this->template parallel_test<thread_count, ops_per_thread>(
      TestFixture::random_op_thread);
}

// A more challenging test using an even smaller key range and the
// same number of threads and operations per thread.  The goal of this
// test is to try an increase coverage of the N48 case.
UNODB_TYPED_TEST(ARTConcurrencyTest,
                 DISABLED_ParallelRandomInsertDeleteGetScan3) {
  constexpr auto thread_count = 4 * 3;
  constexpr auto initial_keys = 32;
  constexpr auto ops_per_thread = 100'000;

  this->verifier.insert_key_range(0, initial_keys, true);
  this->template parallel_test<thread_count, ops_per_thread>(
      TestFixture::random_op_thread);
}

// Optionally enable this for more confidence in debug builds. Set the
// thread_count for your machine.  Fewer keys, more threads, and more
// operations per thread is more challenging.
//
// LCOV_EXCL_START
UNODB_TYPED_TEST(ARTConcurrencyTest,
                 DISABLED_ParallelRandomInsertDeleteGetScanStressTest) {
  constexpr auto thread_count = 48;
  constexpr auto initial_keys = 152;
  constexpr auto ops_per_thread = 10'000'000;

  this->verifier.insert_key_range(0, initial_keys, true);
  this->template parallel_test<thread_count, ops_per_thread>(
      TestFixture::random_op_thread);
}
// LCOV_EXCL_STOP

// ===================================================================
// Chain concurrency tests for key_view types.
//
// These tests exercise the OLC chain insert (single-child inode4
// creation + restart) and two-pass remove (stack-based upward walk)
// under concurrent contention.  The existing tests above only use u64
// keys which cannot trigger chains.
//
// Key encoding: encode(uint8_t tag).encode(uint64_t v) = 9 bytes.
// Keys with the same tag share 8 prefix bytes, triggering chain
// creation.  Different tags diverge at byte 0 (no chain).
// ===================================================================

/// Helper: encode a 9-byte chain-triggering key into a caller-owned buffer.
/// The returned key_view is valid for the lifetime of \a buf.
[[nodiscard]] inline unodb::key_view make_chain_key(
    unodb::key_encoder& enc, std::uint8_t tag, std::uint64_t v,
    std::array<std::byte, 9>& buf UNODB_DETAIL_LIFETIMEBOUND) {
  return copy_key(make_key(enc, tag, v), buf);
}

/// The bytes of short_key_a: 07 00 07.
constexpr std::array short_key_a_bytes{std::byte{0x07}, std::byte{0x00},
                                       std::byte{0x07}};
/// The bytes of short_key_b: 07 00 08.
constexpr std::array short_key_b_bytes{std::byte{0x07}, std::byte{0x00},
                                       std::byte{0x08}};
/// A key of the tree that insert_short_key_pair() builds.
constexpr unodb::key_view short_key_a{short_key_a_bytes};
/// A key of the tree that insert_short_key_pair() builds.
constexpr unodb::key_view short_key_b{short_key_b_bytes};

/// Insert short_key_a and short_key_b with \a vals[0] and \a vals[1]: one I4
/// with prefix 07 00 over them.  Call it through UNODB_ASSERT_NO_FATAL_FAILURE.
// Not noexcept: under --gtest_throw_on_failure a failed check throws.
UNODB_DETAIL_DISABLE_MSVC_WARNING(26440)
template <class Db, class Values>
void insert_short_key_pair(Db& db, const Values& vals) {
  UNODB_ASSERT_TRUE(db.insert(short_key_a, vals[0]));
  UNODB_ASSERT_TRUE(db.insert(short_key_b, vals[1]));
}
UNODB_DETAIL_RESTORE_MSVC_WARNINGS()

/// The bytes of short_sib: 07 01.
constexpr std::array short_sib_bytes{std::byte{0x07}, std::byte{0x01}};
/// A key of the tree that insert_short_collapse_tree() builds.
constexpr unodb::key_view short_sib{short_sib_bytes};

/// Insert short_key_a, short_key_b and short_sib with \a vals in that order.
/// Root-I4, prefix 07, then holds at 0x00 an I4 with an empty prefix over
/// short_key_a and short_key_b, and at 0x01 short_sib's leaf.  Removing
/// short_sib collapses Root-I4 into that I4, whose prefix becomes 07 00, longer
/// than the one byte of short_key_a left below Root-I4.  Call it through
/// UNODB_ASSERT_NO_FATAL_FAILURE.
// Not noexcept: under --gtest_throw_on_failure a failed check throws.
UNODB_DETAIL_DISABLE_MSVC_WARNING(26440)
template <class Db, class Values>
void insert_short_collapse_tree(Db& db, const Values& vals) {
  UNODB_ASSERT_NO_FATAL_FAILURE(insert_short_key_pair(db, vals));
  UNODB_ASSERT_TRUE(db.insert(short_sib, vals[2]));
}
UNODB_DETAIL_RESTORE_MSVC_WARNINGS()

/// The database type of the tests on insert_short_key_pair()'s and
/// insert_short_collapse_tree()'s trees.
using short_collapse_db = unodb::test::key_view_olc_db;

/// short_key_a's, short_key_b's and short_sib's values.
constexpr auto short_collapse_vals = distinct_values<3>();

/// The bytes of \a key.
[[nodiscard]] std::vector<std::byte> key_bytes(unodb::key_view key) {
  return {key.begin(), key.end()};
}

/// Keys, each as its bytes.
using key_list = std::vector<std::vector<std::byte>>;

/// The keys that scan_from(\a from) visits in \a db, in visiting order.
template <class Db>
[[nodiscard]] key_list keys_scanned_from(Db& db, unodb::key_view from,
                                         bool fwd) {
  key_list keys;
  db.scan_from(
      from,
      [&keys](const auto& v) {
        keys.push_back(key_bytes(v.get_key().view()));
        return false;
      },
      fwd);
  return keys;
}

/// Chain concurrency tests reuse ARTConcurrencyTest for QSBR setup and
/// parallel_test.  Thread functions access the db via verifier->get_db().
template <class Db>
class ARTChainConcurrencyTest : public ARTConcurrencyTest<Db> {
 protected:
  /// Thread function: insert/remove chain keys with a fixed tag.
  static void chain_insert_remove_thread(unodb::test::tree_verifier<Db>* tv,
                                         std::size_t thread_i,
                                         std::size_t ops_per_thread) {
    auto& db = tv->get_db();
    unodb::key_encoder enc;
    std::array<std::byte, 9> buf{};
    constexpr auto tag = static_cast<std::uint8_t>(0x42);
    const auto val = test_values[thread_i % test_values.size()];
    for (std::size_t i = 0; i < ops_per_thread; ++i) {
      const auto v = i;
      const auto key = make_chain_key(enc, tag, v, buf);
      if (i % 2 == 0) {
        std::ignore = db.insert(key, val);
      } else {
        if constexpr (unodb::test::is_olc_db<Db>) {
          const unodb::quiescent_state_on_scope_exit qsbr{};
          std::ignore = db.remove(key);
        } else {
          std::ignore = db.remove(key);
        }
      }
    }
    if constexpr (unodb::test::is_olc_db<Db>) unodb::this_thread().quiescent();
  }

  /// Thread function: random insert/remove/get on chain keys.
  static void chain_random_ops_thread(unodb::test::tree_verifier<Db>* tv,
                                      std::size_t thread_i,
                                      std::size_t ops_per_thread) {
    auto& db = tv->get_db();
    std::random_device rd;
    std::mt19937 gen{rd()};
    std::geometric_distribution<std::uint64_t> key_gen{0.05};
    unodb::key_encoder enc;
    std::array<std::byte, 9> buf{};
    constexpr auto tag = static_cast<std::uint8_t>(0x42);
    constexpr auto ntasks = 3;  // insert / remove / get
    for (std::size_t i = 0; i < ops_per_thread; ++i) {
      const auto v = key_gen(gen);
      const auto key = make_chain_key(enc, tag, v, buf);
      const auto val = test_values[v % test_values.size()];
      switch (thread_i % ntasks) {
        case 0:
          std::ignore = db.insert(key, val);
          break;
        case 1:
          if constexpr (unodb::test::is_olc_db<Db>) {
            const unodb::quiescent_state_on_scope_exit qsbr{};
            std::ignore = db.remove(key);
          } else {
            std::ignore = db.remove(key);
          }
          break;
        case 2:
          if constexpr (unodb::test::is_olc_db<Db>) {
            const unodb::quiescent_state_on_scope_exit qsbr{};
            std::ignore = db.get(key);
          } else {
            std::ignore = db.get(key);
          }
          break;
          // LCOV_EXCL_START
        default:
          UNODB_DETAIL_CANNOT_HAPPEN();
          // LCOV_EXCL_STOP
      }
    }
    if constexpr (unodb::test::is_olc_db<Db>) unodb::this_thread().quiescent();
  }

  /// Thread function: multi-tag random ops.
  static void chain_multi_tag_thread(unodb::test::tree_verifier<Db>* tv,
                                     std::size_t thread_i,
                                     std::size_t ops_per_thread) {
    auto& db = tv->get_db();
    std::random_device rd;
    std::mt19937 gen{rd()};
    std::geometric_distribution<std::uint64_t> key_gen{0.1};
    unodb::key_encoder enc;
    std::array<std::byte, 9> buf{};
    const auto tag = static_cast<std::uint8_t>(1 + (thread_i % 4));
    for (std::size_t i = 0; i < ops_per_thread; ++i) {
      const auto v = key_gen(gen);
      const auto key = make_chain_key(enc, tag, v, buf);
      const auto val = test_values[v % test_values.size()];
      switch (i % 3) {
        case 0:
          std::ignore = db.insert(key, val);
          break;
        case 1:
          if constexpr (unodb::test::is_olc_db<Db>) {
            const unodb::quiescent_state_on_scope_exit qsbr{};
            std::ignore = db.remove(key);
          } else {
            std::ignore = db.remove(key);
          }
          break;
        case 2:
          if constexpr (unodb::test::is_olc_db<Db>) {
            const unodb::quiescent_state_on_scope_exit qsbr{};
            std::ignore = db.get(key);
          } else {
            std::ignore = db.get(key);
          }
          break;
          // LCOV_EXCL_START
        default:
          UNODB_DETAIL_CANNOT_HAPPEN();
          // LCOV_EXCL_STOP
      }
    }
    if constexpr (unodb::test::is_olc_db<Db>) unodb::this_thread().quiescent();
  }
};

/// The database types the chain concurrency tests run against.
using ChainConcurrentTypes = ::testing::Types<unodb::test::key_view_mutex_db,
                                              unodb::test::key_view_olc_db>;

UNODB_TYPED_TEST_SUITE(ARTChainConcurrencyTest, ChainConcurrentTypes)

/// Concurrent insert/remove on chain keys — exercises chain creation
/// and two-pass removal under contention.
UNODB_TYPED_TEST(ARTChainConcurrencyTest, ChainInsertRemove) {
  this->template parallel_test<8, 200>(TestFixture::chain_insert_remove_thread);
  UNODB_ASSERT_TRUE(this->verifier.get_db().empty() ||
                    !this->verifier.get_db().empty());  // no crash
}

/// Concurrent random insert/remove/get on chain keys with geometric
/// key distribution — hot-spot contention on small v values.
UNODB_TYPED_TEST(ARTChainConcurrencyTest, ChainRandomOps) {
  this->template parallel_test<6, 500>(TestFixture::chain_random_ops_thread);
}

/// Multi-tag concurrent ops — independent chain subtrees with some
/// cross-tag contention at the root inode.
UNODB_TYPED_TEST(ARTChainConcurrencyTest, ChainMultiTagOps) {
  this->template parallel_test<8, 500>(TestFixture::chain_multi_tag_thread);
}

/// Higher contention stress test (DISABLED for CI, enable manually).
// LCOV_EXCL_START
UNODB_TYPED_TEST(ARTChainConcurrencyTest, DISABLED_ChainStressTest) {
  this->template parallel_test<12, 10'000>(
      TestFixture::chain_random_ops_thread);
}
// LCOV_EXCL_STOP

#ifndef NDEBUG

// Key construction helpers whose every use is debug-only; see
// art_test_data.hpp for the encoded key shapes. They sit inside this guard so
// that NDEBUG builds do not trip clang-tidy's misc-unused-using-decls; the
// helpers also used unconditionally are declared with the group above.
using unodb::test_data::make_key_26;
using unodb::test_data::make_key_34;

// ===================================================================
// Tests of run_parked_interleaving() itself.
// ===================================================================

/// A parked operation that never reaches its sync point must fail the test
/// rather than leave the interleaved thread waiting for the park forever.
UNODB_TEST(RunParkedInterleaving, MissedParkFailsInsteadOfHanging) {
  unodb::detail::sync_point point;
  bool interleaved_ran = false;
  EXPECT_NONFATAL_FAILURE(
      {
        const unodb::test::intercepted_failure_guard guard;
        // Only parked's hits are counted after a miss, none here; this test
        // checks the failure.
        std::ignore = run_parked_interleaving(
            point, []() noexcept {},
            [&interleaved_ran]() noexcept { interleaved_ran = true; });
      },
      "returned before hit 1 of its sync point");
  UNODB_ASSERT_TRUE(interleaved_ran);
}

/// A parked operation that returns between its first hit and its park must
/// fail the test too, and the interleaved operation's own hits of the point
/// must not park it.  ParkHit is one more than both callables' hits together,
/// so that a hit the disarm fails to stop is counted instead of parking.
UNODB_TEST(RunParkedInterleaving, PartialMissDisarmsBeforeRelease) {
  unodb::detail::sync_point point;
  const auto hit_point = [&point]() noexcept { unodb::detail::sync(point); };
  unsigned hits = 0;
  bool interleaved_ran = false;
  EXPECT_NONFATAL_FAILURE(
      {
        const unodb::test::intercepted_failure_guard guard;
        hits = run_parked_interleaving<3>(
            point, hit_point, [&interleaved_ran, &hit_point]() noexcept {
              interleaved_ran = true;
              hit_point();
            });
      },
      "returned before hit 3 of its sync point");
  // parked's hit only: interleaved's came after the disarm.
  UNODB_ASSERT_EQ(1U, hits);
  UNODB_ASSERT_TRUE(interleaved_ran);
}

// ===================================================================
// Remove descent vs. a concurrent prefix split of the first chain node of a
// run, below the root pointer or below a non-chain inode.  The descent reads
// that node's version only after its last check of the node's parent, and the
// split moves the node to a new depth without obsoleting it, so the descent
// must check the parent again before it walks on.  Every byte the stale walk
// reads above the bottom node is chain_key_filler_byte, so it matches each
// prefix and dispatch byte down there.  In the root-chain tests that includes
// the tag: the split empties the chain top's prefix, so the walk dispatches it
// on the tag.  With another tag it would find no child there and restart on
// the root pointer check, and the tests would pass without the fix.
// ===================================================================

/// Tag that sets T2's key apart from key_a's in the root-chain tests.
constexpr std::uint8_t root_chain_t2_tag = 0x20;

/// Tag of key_a and T2's key in the below-inode tests, under a root that
/// dispatches on it.
constexpr std::uint8_t below_inode_tag = 0x10;

/// Last byte of key_a, and of the sibling that a walk short of key_a's depth
/// reaches instead.
constexpr std::uint8_t key_a_last_byte = 0x01;

// The filler byte must differ from root_chain_t2_tag, or key_a and T2's key
// are equal in the root-chain tests, and from key_a_last_byte, or key_a is a
// prefix of its sibling.
static_assert(unodb::test_data::chain_key_filler_byte != root_chain_t2_tag &&
                  unodb::test_data::chain_key_filler_byte != key_a_last_byte,
              "key_a must differ from T2's key and must not be a prefix of "
              "its sibling");

/// chain_key_filler with its second byte changed, for keys that must diverge
/// from chain_key_filler at the second byte of the corresponding key word.
constexpr auto chain_key_filler_second_byte_changed =
    unodb::test_data::chain_key_filler ^
    std::uint64_t{0x0001'0000'0000'0000ULL};

/// Encode a 33-byte key: \a tag, \a word1, two chain_key_fillers, and
/// chain_key_filler with its last byte replaced by \a last.  Below a root
/// that dispatches on the tag, its chain starts with a seven-byte prefix.
[[nodiscard]] unodb::key_view make_key_33(unodb::key_encoder& enc
                                          UNODB_DETAIL_LIFETIMEBOUND,
                                          std::uint8_t tag, std::uint64_t word1,
                                          std::uint8_t last) {
  constexpr auto X = unodb::test_data::chain_key_filler;
  return enc.reset()
      .encode(tag)
      .encode(word1)
      .encode(X)
      .encode(X)
      .encode((X & ~std::uint64_t{0xFF}) | last)
      .get_key_view();
}

/// Encode a 35-byte key: \a tag, four chain_key_fillers, chain_key_filler_byte
/// and key_a_last_byte.  It extends key_a by filler bytes before key_a's last
/// byte, one for a 34-byte key_a and two for a 33-byte one, so that a walk that
/// many bytes short reaches it with one key byte left.
[[nodiscard]] unodb::key_view make_key_35(unodb::key_encoder& enc
                                          UNODB_DETAIL_LIFETIMEBOUND,
                                          std::uint8_t tag) {
  constexpr auto X = unodb::test_data::chain_key_filler;
  return enc.reset()
      .encode(tag)
      .encode(X)
      .encode(X)
      .encode(X)
      .encode(X)
      .encode(unodb::test_data::chain_key_filler_byte)
      .encode(key_a_last_byte)
      .get_key_view();
}

/// What T2 does with its key in assert_remove_keeps_other_keys().
enum class t2_action : std::uint8_t { insert, remove };

/// Insert \a key_a, then \a t2_key if \a action is t2_action::remove, then
/// \a sibs into a \a Db, each with its own value.  Remove \a key_a with T1
/// parked at the first hit of \a point while T2 inserts or removes \a t2_key
/// as \a action says.  Check that \a point was hit \a expected_hits times,
/// that every sibling keeps its value, that \a t2_key has its value after an
/// insert and is gone after a remove, and that \a key_a is gone, then run
/// \a post_check on the tree if given.
template <class Db>
void assert_remove_keeps_other_keys(unodb::detail::sync_point& point,
                                    unsigned expected_hits,
                                    unodb::key_view key_a,
                                    std::span<const unodb::key_view> sibs,
                                    unodb::key_view t2_key,
                                    t2_action action = t2_action::insert,
                                    void (*post_check)(const Db&) = nullptr) {
  // One value per key, so that a lookup that reaches another key's keyless
  // leaf or packed value finds the wrong value: test value 0 for key_a, 1 for
  // t2_key, and i + 2 for sibs[i].
  UNODB_DETAIL_ASSERT(sibs.size() + 2 <=
                      std::min(unodb::test_data::test_values.size(),
                               unodb::test_data::test_values_u64.size()));
  const auto val = [](std::size_t i) {
    return unodb::test::get_test_value<Db>(i);
  };

  Db db;
  UNODB_ASSERT_TRUE(db.insert(key_a, val(0)));
  if (action == t2_action::remove) UNODB_ASSERT_TRUE(db.insert(t2_key, val(1)));
  for (std::size_t i = 0; i < sibs.size(); ++i)
    UNODB_ASSERT_TRUE(db.insert(sibs[i], val(i + 2)));

  bool removed = false;
  bool t2_ok = false;
  const auto hits = run_parked_interleaving(
      point, [&] { removed = db.remove(key_a); },
      [&] {
        t2_ok = action == t2_action::insert ? db.insert(t2_key, val(1))
                                            : db.remove(t2_key);
      });
  UNODB_ASSERT_EQ(expected_hits, hits);
  UNODB_ASSERT_TRUE(removed);
  UNODB_ASSERT_TRUE(t2_ok);
  const unodb::quiescent_state_on_scope_exit q{};
  // The surviving keys first: a wrong remove can leave obsoleted nodes linked,
  // which a lookup of an absent key would then walk.
  for (std::size_t i = 0; i < sibs.size(); ++i)
    ASSERT_VALUE_FOR_KEY(Db, db, sibs[i], val(i + 2));
  if (action == t2_action::insert) {
    ASSERT_VALUE_FOR_KEY(Db, db, t2_key, val(1));
  } else {
    UNODB_ASSERT_FALSE(db.get(t2_key).has_value());
  }
  UNODB_ASSERT_FALSE(db.get(key_a).has_value());
  if (post_check != nullptr) post_check(db);
}

/// Fixture of the remove-descent tests.
template <class Db>
class OLCRemoveDescentVsSplit : public ::testing::Test {};

/// The OLC key_view databases without a heap, which
/// assert_remove_keeps_other_keys() and assert_step_reseeks_to_current_key()
/// default-construct.  The stale walk of each KeepsSibling test ends at the
/// sibling's keyless leaf in the first and at its packed value in the second,
/// which reach the chain cut through different arms, and an iterator step
/// re-seeks to a keyless leaf in the first and to a packed value in the
/// second.
using OLCKeyViewTypes = ::testing::Types<unodb::test::key_view_olc_db,
                                         unodb::test::key_view_u64val_olc_db>;

/// Whether a \a Db of OLCKeyViewTypes keeps values in leaf nodes rather than
/// packing them into inode slots.
template <class Db>
constexpr bool has_leaf_nodes_v =
    !unodb::detail::olc_art_policy<typename Db::key_type,
                                   typename Db::value_type>::can_eliminate_leaf;

UNODB_TYPED_TEST_SUITE(OLCRemoveDescentVsSplit, OLCKeyViewTypes)

/// key_a's chain is the whole tree, and its top's prefix is the tag.  T1
/// removes key_a and parks after checking the root pointer, before reading the
/// chain top.  T2 inserts a key with another tag, which splits that prefix and
/// installs a new root above the chain top, so T1 would walk the chain one
/// byte short and miss key_a at the bottom node.  The tag must be
/// chain_key_filler_byte; see the section comment.
UNODB_TYPED_TEST(OLCRemoveDescentVsSplit, SplitRootChainTopRemovesKey) {
  constexpr auto X = unodb::test_data::chain_key_filler;
  constexpr auto tag = unodb::test_data::chain_key_filler_byte;
  unodb::key_encoder enc;
  std::array<std::byte, 34> buf_a{};
  std::array<std::byte, 34> buf_t2{};
  const auto key_a = copy_key(make_key_34(enc, tag, X, key_a_last_byte), buf_a);
  const auto t2_key =
      copy_key(make_key_34(enc, root_chain_t2_tag, X, 0x01), buf_t2);
  // T1's park and its retry: T2's insert never reaches the point, and T1
  // restarts when it finds that the root pointer moved.
  assert_remove_keeps_other_keys<TypeParam>(
      unodb::detail::sync_before_remove_root_read, 2, key_a, {}, t2_key);
}

/// SplitRootChainTopRemovesKey with a sibling that T1's walk one byte short
/// would reach instead of key_a, and remove.
UNODB_TYPED_TEST(OLCRemoveDescentVsSplit, SplitRootChainTopKeepsSibling) {
  constexpr auto X = unodb::test_data::chain_key_filler;
  constexpr auto tag = unodb::test_data::chain_key_filler_byte;
  unodb::key_encoder enc;
  std::array<std::byte, 34> buf_a{};
  std::array<std::byte, 35> buf_sib{};
  std::array<std::byte, 34> buf_t2{};
  const auto key_a = copy_key(make_key_34(enc, tag, X, key_a_last_byte), buf_a);
  const auto sib = copy_key(make_key_35(enc, tag), buf_sib);
  const auto t2_key =
      copy_key(make_key_34(enc, root_chain_t2_tag, X, 0x01), buf_t2);
  // T1's park and its retry, as in SplitRootChainTopRemovesKey.
  assert_remove_keeps_other_keys<TypeParam>(
      unodb::detail::sync_before_remove_root_read, 2, key_a, std::array{sib},
      t2_key);
}

/// Root holds a one-byte key and key_a's chain, whose top's prefix is
/// seven bytes long.  T1 removes key_a and parks after checking the root,
/// before reading the chain top.  T2 inserts a key that differs in the chain
/// top's second prefix byte, which splits it, so T1 would walk the chain two
/// bytes short and miss key_a at the bottom node.
UNODB_TYPED_TEST(OLCRemoveDescentVsSplit, SplitChainTopBelowInodeRemovesKey) {
  constexpr auto X = unodb::test_data::chain_key_filler;
  unodb::key_encoder enc;
  std::array<std::byte, 33> buf_a{};
  std::array<std::byte, 1> buf_sib{};
  std::array<std::byte, 33> buf_t2{};
  const auto key_a =
      copy_key(make_key_33(enc, below_inode_tag, X, key_a_last_byte), buf_a);
  const auto sib = copy_key(make_short_key(enc, 0x20), buf_sib);
  const auto t2_key =
      copy_key(make_key_33(enc, below_inode_tag,
                           chain_key_filler_second_byte_changed, 0x01),
               buf_t2);
  // T1's park at the root, then its retry's reads below the root and below
  // the I4 that T2's split installed; the rest of the path is chain.
  assert_remove_keeps_other_keys<TypeParam>(
      unodb::detail::sync_before_remove_child_read, 3, key_a, std::array{sib},
      t2_key);
}

/// SplitChainTopBelowInodeRemovesKey with a second sibling that T1's walk two
/// bytes short would reach instead of key_a, and remove.
UNODB_TYPED_TEST(OLCRemoveDescentVsSplit, SplitChainTopBelowInodeKeepsSibling) {
  constexpr auto X = unodb::test_data::chain_key_filler;
  unodb::key_encoder enc;
  std::array<std::byte, 33> buf_a{};
  std::array<std::byte, 1> buf_sib1{};
  std::array<std::byte, 35> buf_sib2{};
  std::array<std::byte, 33> buf_t2{};
  const auto key_a =
      copy_key(make_key_33(enc, below_inode_tag, X, key_a_last_byte), buf_a);
  const std::array sibs{copy_key(make_short_key(enc, 0x20), buf_sib1),
                        copy_key(make_key_35(enc, below_inode_tag), buf_sib2)};
  const auto t2_key =
      copy_key(make_key_33(enc, below_inode_tag,
                           chain_key_filler_second_byte_changed, 0x01),
               buf_t2);
  // As in SplitChainTopBelowInodeRemovesKey, plus the retry's read below the
  // bottom node, which holds key_a and the sibling's chain and so is not a
  // chain node, if key_a is a keyless leaf there.  A packed value has no child
  // to read.
  constexpr unsigned key_a_read_hits = has_leaf_nodes_v<TypeParam> ? 1U : 0U;
  assert_remove_keeps_other_keys<TypeParam>(
      unodb::detail::sync_before_remove_child_read, 3 + key_a_read_hits, key_a,
      sibs, t2_key);
}

// ===================================================================
// Remove descent vs. an insert into a chain I4 whose only child is a packed
// value.  The descent loads the child and checks the node, then reads whether
// the child's slot holds a packed value.  An insert in between can shift the
// packed value to the next slot and leave a node pointer, whose value bit is
// clear, in its old one.
// ===================================================================

/// T1 removes chain_key, whose value is the only child of a chain I4 at 0x10
/// below Root-I4, and parks before it reads the value bit of that child's
/// slot.  T2 inserts a key that leaves the chain I4 at 0x11, before
/// chain_key's 0x22, and goes on below it, so the slot T1 found chain_key's
/// value in now holds T2's chain.  T1 must restart instead of taking the value
/// it loaded for a node pointer.
UNODB_TEST(OLCChainValueBitVsInsert, RemoveRestarts) {
  using db_type = unodb::test::key_view_u64val_olc_db;
  constexpr std::array other_key{std::byte{0x05}, std::byte{0x01}};
  constexpr std::array chain_key{std::byte{0x10}, std::byte{0x20},
                                 std::byte{0x21}, std::byte{0x22}};
  constexpr std::array t2_key{std::byte{0x10}, std::byte{0x20}, std::byte{0x21},
                              std::byte{0x11}, std::byte{0x33}};
  constexpr std::uint64_t other_val{0x42};
  // A packed value's low bits sit where a node pointer's tag does, and these
  // are I4's.  Read as a pointer, it is read-locked as an I4.  As a leaf, it
  // would take the branch that write-locks the node first, which restarts.
  constexpr std::uint64_t chain_val{0x61};
  constexpr std::uint64_t t2_val{0x43};

  db_type db;
  UNODB_ASSERT_TRUE(db.insert(unodb::key_view{other_key}, other_val));
  UNODB_ASSERT_TRUE(db.insert(unodb::key_view{chain_key}, chain_val));

  bool removed = false;
  bool inserted = false;
  const auto hits = run_parked_interleaving(
      unodb::detail::sync_before_chain_value_bit_read,
      [&] { removed = db.remove(unodb::key_view{chain_key}); },
      [&] { inserted = db.insert(unodb::key_view{t2_key}, t2_val); });
  // T1's park only: its retry finds two children at 0x10, so it does not take
  // the chain-I4 branch there.
  UNODB_ASSERT_EQ(1U, hits);
  UNODB_ASSERT_TRUE(removed);
  UNODB_ASSERT_TRUE(inserted);

  ASSERT_VALUE_FOR_KEY(db_type, db, unodb::key_view{other_key}, other_val);
  ASSERT_VALUE_FOR_KEY(db_type, db, unodb::key_view{t2_key}, t2_val);
  const unodb::quiescent_state_on_scope_exit q{};
  UNODB_ASSERT_FALSE(db.get(unodb::key_view{chain_key}).has_value());
}

// ===================================================================
// Iterator step vs. a concurrent write.  When next() or prior() fails its
// step, it re-seeks to the current key.  The step has already written its
// target's key prefix and dispatch bytes into the iterator's key buffer by
// then, so the key it re-seeks to cannot be a view of that buffer.
// ===================================================================

/// Run T1's \a op parked at the second hit of \a point, after it has
/// validated the first node on its path, while T2 removes \a removed.
/// \return The point's hit count, and whether T2 removed \a removed.
template <class Db, class Op>
[[nodiscard]] std::pair<unsigned, bool> parked_at_second_node(
    Db& db, unodb::detail::sync_point& point, const Op& op,
    typename Db::key_type removed) {
  bool removed_ok = false;
  const auto hits = run_parked_interleaving<2>(
      point, op, [&] { removed_ok = db.remove(removed); });
  return {hits, removed_ok};
}

/// The bytes of the step keys, 01 20 CC 01 05, 01 20 CC 01 06 and
/// 01 20 CC 01 07: an I4 holds them at 01 below the I4 with prefix CC that
/// Root-I4 holds at 20.
constexpr std::array step_key_s_bytes{std::byte{0x01}, std::byte{0x20},
                                      std::byte{0xCC}, std::byte{0x01},
                                      std::byte{0x05}};
/// See step_key_s_bytes.
constexpr std::array step_key_t_bytes{std::byte{0x01}, std::byte{0x20},
                                      std::byte{0xCC}, std::byte{0x01},
                                      std::byte{0x06}};
/// See step_key_s_bytes.
constexpr std::array step_key_x_bytes{std::byte{0x01}, std::byte{0x20},
                                      std::byte{0xCC}, std::byte{0x01},
                                      std::byte{0x07}};
/// The bytes of SR1's start key, 01 10 AA BB EE.
constexpr std::array sr1_from_bytes{std::byte{0x01}, std::byte{0x10},
                                    std::byte{0xAA}, std::byte{0xBB},
                                    std::byte{0xEE}};
/// The bytes of SR1's last key, 01 20 CC 02 07.
constexpr std::array sr1_u_bytes{std::byte{0x01}, std::byte{0x20},
                                 std::byte{0xCC}, std::byte{0x02},
                                 std::byte{0x07}};
/// The bytes of SR2's start key, 01 30 AA BB 00.
constexpr std::array sr2_from_bytes{std::byte{0x01}, std::byte{0x30},
                                    std::byte{0xAA}, std::byte{0xBB},
                                    std::byte{0x00}};
/// The bytes of SR2's last key, 01 20 CC 00 07.
constexpr std::array sr2_u_bytes{std::byte{0x01}, std::byte{0x20},
                                 std::byte{0xCC}, std::byte{0x00},
                                 std::byte{0x07}};

/// The size of the step tests' keys before padding.
constexpr std::size_t unpadded_key_size = step_key_s_bytes.size();

/// The size the padded step tests pad their keys to: over the inline capacity
/// of the buffers that hold, copy and encode keys, so that those buffers grow.
constexpr std::size_t padded_key_size = 300;
static_assert(padded_key_size > unodb::detail::INITIAL_BUFFER_CAPACITY);

/// The sync point hits of a padded step test's scan: the I4 with prefix CC and
/// the step keys' I4, both again after the re-seek, and, for each of the two
/// step keys left and the last key, the chain of 37 I4s that its last 295 or
/// 296 bytes take at key_prefix_capacity + 1 = 8 bytes per I4.  Another
/// padded_key_size can change the tree's shape, not just the chains' length.
constexpr unsigned padded_step_scan_hits = 4 + (3 * 37);

/// \a head padded with zero bytes to \a size bytes.
[[nodiscard]] std::vector<std::byte> padded_key(unodb::key_view head,
                                                std::size_t size) {
  std::vector<std::byte> key(size);
  std::ranges::copy(head, key.begin());
  return key;
}

/// Insert \a from_key, the three step keys and \a u_key, each padded to
/// \a key_size bytes, into a \a Db.  Scan from \a from_key, forward if \a fwd,
/// with the first step parked at the step keys' I4 while T2 removes
/// step_key_x_bytes, and check that the scan visits \a from_key, the two other
/// step keys in scan order, and \a u_key.
///
/// The step descends from Root-I4 through the I4 with prefix CC and parks at
/// the step keys' I4.  T2's remove moves that I4's version, so the step fails
/// after it has written 01 20 CC 01 over \a from_key's first four bytes.
template <class Db>
void assert_step_reseeks_to_current_key(unodb::key_view from_key,
                                        unodb::key_view u_key, bool fwd,
                                        std::size_t key_size) {
  const auto from = padded_key(from_key, key_size);
  const auto s_key = padded_key(unodb::key_view{step_key_s_bytes}, key_size);
  const auto t_key = padded_key(unodb::key_view{step_key_t_bytes}, key_size);
  const auto x_key = padded_key(unodb::key_view{step_key_x_bytes}, key_size);
  const auto u = padded_key(u_key, key_size);

  Db db;
  UNODB_ASSERT_TRUE(
      db.insert(unodb::key_view{from}, unodb::test::get_test_value<Db>(0)));
  UNODB_ASSERT_TRUE(
      db.insert(unodb::key_view{s_key}, unodb::test::get_test_value<Db>(1)));
  UNODB_ASSERT_TRUE(
      db.insert(unodb::key_view{t_key}, unodb::test::get_test_value<Db>(2)));
  UNODB_ASSERT_TRUE(
      db.insert(unodb::key_view{x_key}, unodb::test::get_test_value<Db>(3)));
  UNODB_ASSERT_TRUE(
      db.insert(unodb::key_view{u}, unodb::test::get_test_value<Db>(4)));

  key_list keys;
  const auto [hits, removed] = parked_at_second_node(
      db, unodb::detail::sync_in_scan_after_parent_unlock,
      [&] { keys = keys_scanned_from(db, unodb::key_view{from}, fwd); },
      unodb::key_view{x_key});
  // Unpadded keys: the I4 with prefix CC and the step keys' I4, both again
  // after the re-seek, and u_key's I4.  Leaf nodes add the leaves of the two
  // step keys left and u_key, where the descents after the re-seek end.
  const auto expected_hits =
      (key_size == padded_key_size ? padded_step_scan_hits : 5U) +
      (has_leaf_nodes_v<Db> ? 3U : 0U);
  UNODB_ASSERT_EQ(expected_hits, hits);
  UNODB_ASSERT_TRUE(removed);
  const auto expected =
      fwd ? key_list{from, s_key, t_key, u} : key_list{from, t_key, s_key, u};
  UNODB_ASSERT_EQ(keys, expected);
}

/// Fixture of the iterator step tests.
template <class Db>
class OLCIteratorStepRestart : public ::testing::Test {};

UNODB_TYPED_TEST_SUITE(OLCIteratorStepRestart, OLCKeyViewTypes)

/// SR1: Root-I4, prefix 01, holds at 10 an I4 with prefix AA BB that holds
/// 01 10 AA BB EE, and at 20 the I4 with prefix CC, which holds the I4 of
/// step_key_s_bytes at 01 and 01 20 CC 02 07 at 02.  next() from
/// 01 10 AA BB EE takes Root-I4's 20.  A re-seek to 01 20 CC 01 EE would
/// find no EE in the parked I4 and land on 01 20 CC 02 07.
UNODB_TYPED_TEST(OLCIteratorStepRestart, NextReseeksToCurrentKey) {
  assert_step_reseeks_to_current_key<TypeParam>(unodb::key_view{sr1_from_bytes},
                                                unodb::key_view{sr1_u_bytes},
                                                true, unpadded_key_size);
}

/// SR2: mirror of SR1 for prior().  Root-I4 holds the I4 with prefix CC at 20
/// and an I4 with prefix AA BB that holds 01 30 AA BB 00 at 30.  The I4 with
/// prefix CC holds 01 20 CC 00 07 at 00, so the I4 of step_key_s_bytes is its
/// right-most child.  prior() from 01 30 AA BB 00 takes Root-I4's 20.  A
/// re-seek to 01 20 CC 01 00 would find no key at or below 00 in the parked
/// I4 and land on 01 20 CC 00 07.
UNODB_TYPED_TEST(OLCIteratorStepRestart, PriorReseeksToCurrentKey) {
  assert_step_reseeks_to_current_key<TypeParam>(unodb::key_view{sr2_from_bytes},
                                                unodb::key_view{sr2_u_bytes},
                                                false, unpadded_key_size);
}

/// SR1 with its keys padded to padded_key_size bytes, so that the key the
/// step saves outgrows the inline capacity of the buffer it is saved into.
UNODB_TYPED_TEST(OLCIteratorStepRestart, NextReseeksToCurrentPaddedKey) {
  assert_step_reseeks_to_current_key<TypeParam>(unodb::key_view{sr1_from_bytes},
                                                unodb::key_view{sr1_u_bytes},
                                                true, padded_key_size);
}

/// SR2 with its keys padded as NextReseeksToCurrentPaddedKey pads SR1's.
UNODB_TYPED_TEST(OLCIteratorStepRestart, PriorReseeksToCurrentPaddedKey) {
  assert_step_reseeks_to_current_key<TypeParam>(unodb::key_view{sr2_from_bytes},
                                                unodb::key_view{sr2_u_bytes},
                                                false, padded_key_size);
}

/// A TestHeap whose extract_key() encodes the key into the buffer it gets, as
/// the TupleHeap contract's main case does, rather than return a view of its
/// own storage, so that a view of that buffer dies with it.
class EncodingTestHeap {
 public:
  /// Register \a key for \a id, as unodb::test::TestHeap::add_tuple() does.
  void add_tuple(std::uint64_t id, unodb::key_view key) {
    heap_.add_tuple(id, key);
  }

  /// Encode the key of \a id into \a buf.
  [[nodiscard]] unodb::key_view extract_key(std::uint64_t id,
                                            unodb::key_encoder& buf) const {
    ++extract_count_;
    return buf.reset().append_bytes(heap_.extract_key(id, buf)).get_key_view();
  }

  /// The number of extract_key() calls so far.
  [[nodiscard]] std::size_t extract_count() const noexcept {
    return extract_count_;
  }

 private:
  /// The registered keys.
  unodb::test::TestHeap heap_;
  /// See extract_count().  Not atomic, because only iterators call
  /// extract_key(), and each test scans from one thread.
  mutable std::size_t extract_count_{0};
};

/// Run the test of \a from_key and \a u_key, forward if \a fwd, as
/// assert_step_reseeks_to_current_key() does, on a heap tree whose heap is an
/// EncodingTestHeap.  Each key is padded to padded_key_size bytes, so that the
/// buffer it is encoded into outgrows its inline storage, and a re-seek key
/// that outlives that buffer reads freed memory.  The scan reads only values,
/// so the failed step's re-seek must be its only heap read.
void assert_heap_step_reseeks_to_current_key(unodb::key_view from_key,
                                             unodb::key_view u_key, bool fwd) {
  using db_type =
      unodb::olc_db<unodb::key_view, std::uint64_t, EncodingTestHeap>;
  // In the order of their tuple ids 0 to 4: the start key, the three step
  // keys, and the last key.
  const std::array keys{
      padded_key(from_key, padded_key_size),
      padded_key(unodb::key_view{step_key_s_bytes}, padded_key_size),
      padded_key(unodb::key_view{step_key_t_bytes}, padded_key_size),
      padded_key(unodb::key_view{step_key_x_bytes}, padded_key_size),
      padded_key(u_key, padded_key_size)};

  EncodingTestHeap heap;
  db_type db{heap};
  for (std::uint64_t id = 0; id < keys.size(); ++id) {
    heap.add_tuple(id, keys[id]);
    UNODB_ASSERT_TRUE(db.insert(unodb::key_view{keys[id]}, id));
  }

  std::vector<std::uint64_t> ids;
  const auto [hits, removed] = parked_at_second_node(
      db, unodb::detail::sync_in_scan_after_parent_unlock,
      [&] {
        db.scan_from(
            unodb::key_view{keys[0]},
            [&ids](const auto& v) {
              ids.push_back(v.get_value());
              return false;
            },
            fwd);
      },
      unodb::key_view{keys[3]});
  UNODB_ASSERT_EQ(padded_step_scan_hits, hits);
  UNODB_ASSERT_TRUE(removed);
  const auto expected = fwd ? std::vector<std::uint64_t>{0, 1, 2, 4}
                            : std::vector<std::uint64_t>{0, 2, 1, 4};
  UNODB_ASSERT_EQ(ids, expected);
  UNODB_ASSERT_EQ(1U, heap.extract_count());
}

/// SR3: SR1 on a heap tree.
UNODB_TEST(OLCIteratorStepRestartHeap, NextReseeksToCurrentKey) {
  assert_heap_step_reseeks_to_current_key(unodb::key_view{sr1_from_bytes},
                                          unodb::key_view{sr1_u_bytes}, true);
}

/// SR4: SR2 on a heap tree.
UNODB_TEST(OLCIteratorStepRestartHeap, PriorReseeksToCurrentKey) {
  assert_heap_step_reseeks_to_current_key(unodb::key_view{sr2_from_bytes},
                                          unodb::key_view{sr2_u_bytes}, false);
}

/// The u64 step keys, 0x200105, 0x200106 and 0x200107: an I4 holds them at
/// 01 below the I4 that Root-I4, whose prefix is five zero bytes, holds at 20.
constexpr std::array<std::uint64_t, 3> u64_step_keys{0x200105, 0x200106,
                                                     0x200107};

/// Run the test of \a from_key and \a u_key, forward if \a fwd, as
/// assert_step_reseeks_to_current_key() does, on a u64_olc_db with the u64
/// step keys.  Such a tree re-seeks to the key of the leaf it was on, which
/// the step cannot change.  The test covers reseek_key()'s read of that key,
/// which only a failed step makes.
void assert_u64_step_reseeks_to_current_key(std::uint64_t from_key,
                                            std::uint64_t u_key, bool fwd) {
  using db_type = unodb::test::u64_olc_db;
  const auto [s_key, t_key, x_key] = u64_step_keys;
  const std::array keys{from_key, s_key, t_key, x_key, u_key};

  db_type db;
  for (std::size_t i = 0; i < keys.size(); ++i) {
    UNODB_ASSERT_TRUE(
        db.insert(keys[i], unodb::test::get_test_value<db_type>(i)));
  }

  std::vector<std::uint64_t> visited;
  const auto [hits, removed] = parked_at_second_node(
      db, unodb::detail::sync_in_scan_after_parent_unlock,
      [&] {
        db.scan_from(
            from_key,
            [&visited](const auto& v) {
              visited.push_back(decode(v.get_key()));
              return false;
            },
            fwd);
      },
      x_key);
  // The I4 at 20 and the step keys' I4, both again after the re-seek, and the
  // leaves of the two step keys left and of u_key.
  UNODB_ASSERT_EQ(7U, hits);
  UNODB_ASSERT_TRUE(removed);
  const auto expected =
      fwd ? std::vector<std::uint64_t>{from_key, s_key, t_key, u_key}
          : std::vector<std::uint64_t>{from_key, t_key, s_key, u_key};
  UNODB_ASSERT_EQ(visited, expected);
}

/// SR5: SR1 on a u64 tree.  Root-I4 holds 0x100000 at 10, and at 20 an I4
/// that holds the I4 of the u64 step keys at 01 and 0x200207 at 02.
UNODB_TEST(OLCIteratorStepRestartU64, NextReseeksToCurrentKey) {
  assert_u64_step_reseeks_to_current_key(0x100000, 0x200207, true);
}

/// SR6: SR2 on a u64 tree.  Root-I4 holds 0x300000 at 30, and at 20 an I4
/// that holds 0x200007 at 00 and the I4 of the u64 step keys at 01.
UNODB_TEST(OLCIteratorStepRestartU64, PriorReseeksToCurrentKey) {
  assert_u64_step_reseeks_to_current_key(0x300000, 0x200007, false);
}

// ===================================================================
// Dangling chain bug test.
//
// Tests that the two-pass remove correctly handles the case where
// chain nodes are reclaimed during the upward walk but a concurrent
// modification to a higher ancestor causes a version mismatch.
//
// Setup: Parent-I4 with a 2-level chain subtree + a sibling leaf.
//   Parent-I4 → [tag1: Chain-head → Chain-tail → Leaf(target),
//                 tag2: Leaf(sibling)]
//
// The test uses sync_point to pause T1 after it has obsoleted
// the chain head but before it processes the parent.  T2 then
// modifies the parent (inserts a new sibling), causing T1's version
// check to fail on restart.
//
// Bug: T1 loops forever because Parent-I4 still points to the
// obsoleted chain head.
// Fix: T1 must handle the dangling chain correctly.
// ===================================================================

// ===================================================================
// Concurrent chain cut tests with explicit interleavings.
// Uses run_parked_interleaving() for coordination.
//
// Key constraint: T2 must only access nodes that T1 does NOT hold
// write guards on at the sync point.
//
// At SP1 (sync_after_chain_locked): T1 holds chain_bottom (ng), the
//   chain nodes Step 2 locked at their descent versions (chain_guards), and
//   leaf (lg).
//   The cut_point_parent is NOT locked.
//
// At SP2 (sync_between_chain_locks): T1 holds chain_bottom (ng),
//   stk[stk_n-1] (chain_guards[0]), and leaf (lg).
//   All other stack entries are NOT locked.
//
// Tree setup uses 26-byte keys (CT1/CT1b/CT3/CT3b, 3 chain levels), a lone
// 33-byte key (CT1c/CT1d, 3 chain levels) or 34-byte keys (CT2/CT2b/CT4, 4
// chain levels) to ensure enough chain depth that pg locks a deep chain node,
// not the cut_point_parent.
// ===================================================================

/// Remove key_a's chain from below Root-I4, which also holds a sibling 0x20,
/// while T2 inserts a key whose dispatch byte is \a new_tag into Root-I4
/// between Step 2 and Step 3.
void insert_into_cut_point_parent_during_cut(std::uint8_t new_tag) {
  unodb::key_encoder enc;
  // 26-byte keys: tag(1) + 3×uint64(24) + bottom(1).
  std::array<std::byte, 26> buf_a{};
  std::array<std::byte, 26> buf_sib{};
  std::array<std::byte, 26> buf_new{};
  const auto key_a = copy_key(make_key_26(enc, 0x10, 0x01), buf_a);
  const auto sib = copy_key(make_key_26(enc, 0x20, 0x01), buf_sib);
  const auto new_key = copy_key(make_key_26(enc, new_tag, 0x01), buf_new);
  // T1's park only: T2's insert never reaches the point, and T1's Step 3
  // locks Root-I4 at its moved version and cuts without restarting.
  assert_remove_keeps_other_keys<unodb::test::key_view_olc_db>(
      unodb::detail::sync_after_chain_locked, 1, key_a, std::array{sib},
      new_key);
}

/// Remove key_a, whose chain top below the root keeps a seven-byte prefix,
/// while T2 removes a one-byte key 0x05 from the root between Step 2 and
/// Step 3.  The root also holds \a sibs, which sort after key_a.  T2's remove
/// shifts key_a's chain index in the root down by one and leaves the root at
/// the size from which T1's cut shrinks or collapses it, which \a post_check
/// checks.
void remove_from_cut_point_parent_during_cut(
    std::span<const unodb::key_view> sibs,
    void (*post_check)(const unodb::test::key_view_olc_db&)) {
  constexpr auto X = unodb::test_data::chain_key_filler;
  unodb::key_encoder enc;
  std::array<std::byte, 33> buf_a{};
  std::array<std::byte, 1> buf_t2{};
  const auto key_a = copy_key(make_key_33(enc, 0x10, X, 0x01), buf_a);
  const auto t2_key = copy_key(make_short_key(enc, 0x05), buf_t2);
  // T1's park only: T2 removes a leaf directly in the root, not through a
  // chain cut, and T1's Step 3 locks the root at its moved version and cuts
  // without restarting.
  assert_remove_keeps_other_keys<unodb::test::key_view_olc_db>(
      unodb::detail::sync_after_chain_locked, 1, key_a, sibs, t2_key,
      t2_action::remove, post_check);
}

/// CT1: T1 removes A (chain cut).  T2 inserts a sibling into the
/// cut_point_parent (Root-I4) between Step 2 and Step 3.
UNODB_TEST(OLCChainCutInterleaved, ConcurrentInsertIntoCutPointParent) {
  insert_into_cut_point_parent_during_cut(0x30);
}

/// CT1b: as CT1, but T2's key sorts before chain[0]'s, which shifts chain[0]'s
/// child index in Root-I4.  T1's Step 3 locks Root-I4 at its new version and
/// must cut chain[0] at its new index, not the downward pass's.
UNODB_TEST(OLCChainCutInterleaved, ConcurrentInsertShiftsCutIndex) {
  insert_into_cut_point_parent_during_cut(0x05);
}

/// CT1c: T2's remove leaves Root-I4 with key_a's chain and a 26-byte sibling,
/// whose chain top has an empty prefix, so T1's cut collapses Root-I4 into
/// that top.  At the downward pass's index T1 would remove the sibling
/// instead, and the seven-byte prefix of key_a's chain top would not fit the
/// merge.
UNODB_TEST(OLCChainCutInterleaved, ConcurrentRemoveShiftsCollapseIndex) {
  unodb::key_encoder enc;
  std::array<std::byte, 26> buf_sib{};
  const auto sib = copy_key(make_key_26(enc, 0x20, 0x01), buf_sib);
  remove_from_cut_point_parent_during_cut(
      std::array{sib},
      []([[maybe_unused]] const unodb::test::key_view_olc_db& db) {
#ifdef UNODB_DETAIL_WITH_STATS
        // The sibling's chain, whose top is now the root, and nothing else.
        UNODB_ASSERT_EQ(
            4U, db.get_node_counts()[unodb::as_i<unodb::node_type::I4>]);
#endif
      });
}

/// CT1d: T2's remove leaves Root-I16 with key_a's chain and one-byte siblings,
/// at its minimum size, so T1's cut shrinks it to an I4.  At the downward
/// pass's index T1 would copy every child but the first sibling.
UNODB_TEST(OLCChainCutInterleaved, ConcurrentRemoveShiftsShrinkIndex) {
  using inode_16 =
      unodb::detail::olc_inode_16<unodb::key_view, unodb::value_view>;
  constexpr std::size_t sib_count = inode_16::min_size - 1U;
  unodb::key_encoder enc;
  std::array<std::array<std::byte, 1>, sib_count> buf_sibs{};
  std::array<unodb::key_view, sib_count> sibs{};
  for (std::size_t i = 0; i < sibs.size(); ++i) {
    sibs[i] = copy_key(
        make_short_key(enc, static_cast<std::uint8_t>(0x20 + (0x10 * i))),
        buf_sibs[i]);
  }
  remove_from_cut_point_parent_during_cut(
      sibs, []([[maybe_unused]] const unodb::test::key_view_olc_db& db) {
#ifdef UNODB_DETAIL_WITH_STATS
        UNODB_ASSERT_EQ(1U,
                        db.get_shrinking_inode_count<unodb::node_type::I16>());
#endif
      });
}

/// Remove key_a, a 34-byte key whose chain (chain[0] to chain[3] and the chain
/// bottom) sits below Root-I4 beside a sibling 0x20, while T2 inserts the key
/// with tag 0x10, first key word \a t2_v1 and key_a's remaining bytes.
/// chain[0]'s prefix is empty, so \a t2_v1's first byte is chain[0]'s dispatch
/// byte and its second byte is chain[1]'s first prefix byte.  T1 parks in
/// Step 2 holding chain[3] (stk[4]), the chain bottom and the leaf; chain[0] to
/// chain[2] and Root-I4 are unlocked.  Check that the sync point was hit
/// \a expected_hits times.
void cut_34_byte_chain_during_insert(unsigned expected_hits,
                                     std::uint64_t t2_v1) {
  constexpr auto X = unodb::test_data::chain_key_filler;
  unodb::key_encoder enc;
  std::array<std::byte, 34> buf_a{};
  std::array<std::byte, 34> buf_sib{};
  std::array<std::byte, 34> buf_t2{};
  const auto key_a = copy_key(make_key_34(enc, 0x10, X, 0x01), buf_a);
  const auto sib = copy_key(make_key_34(enc, 0x20, X, 0x01), buf_sib);
  const auto t2_key = copy_key(make_key_34(enc, 0x10, t2_v1, 0x01), buf_t2);
  assert_remove_keeps_other_keys<unodb::test::key_view_olc_db>(
      unodb::detail::sync_between_chain_locks, expected_hits, key_a,
      std::array{sib}, t2_key);
}

/// CT2: T1 removes A (chain cut).  T2's key sorts after key_a's chain in
/// chain[0], which Step 2 therefore fails to lock at its descent version.
/// T1's Step 3 takes chain[0], now two-child, as the cut-point parent at its
/// new version.
UNODB_TEST(OLCChainCutInterleaved, ConcurrentInsertIntoMidChain) {
  // T1's park only: T2's insert never reaches sync_between_chain_locks, and
  // T1's Step 2 stops below chain[0], now two-child, which Step 3 locks at its
  // moved version as the cut-point parent without restarting.
  cut_34_byte_chain_during_insert(1, unodb::test_data::chain_key_filler_alt);
}

/// CT2b: as CT2, but T2's key sorts before key_a's chain in chain[0], so its
/// insert moves chain[1] to slot 1 of chain[0].  T1's Step 3 takes chain[0] as
/// the cut-point parent at its new version and must cut chain[1] at slot 1,
/// not at the descent's slot 0.
UNODB_TEST(OLCChainCutInterleaved, ConcurrentInsertShiftsMidChainCutIndex) {
  // Every byte one below the filler's, so T2's key sorts first in chain[0].
  constexpr auto below_filler = unodb::test_data::chain_key_filler -
                                std::uint64_t{0x0101'0101'0101'0101ULL};
  // T1's park only, as in CT2: Step 3 locks chain[0] at its moved version as
  // the cut-point parent without restarting.
  cut_34_byte_chain_during_insert(1, below_filler);
}

/// Encode the KeySize-byte chain test key with \a tag and last byte \a last
/// into \a buf: make_key_26(), or make_key_34() with chain_key_filler.
/// \tparam KeySize Key length, 26 or 34
template <std::size_t KeySize>
[[nodiscard]] unodb::key_view make_deep_chain_key(unodb::key_encoder& enc,
                                                  std::uint8_t tag,
                                                  std::uint8_t last,
                                                  std::span<std::byte> buf
                                                  UNODB_DETAIL_LIFETIMEBOUND) {
  static_assert(KeySize == 26 || KeySize == 34);
  if constexpr (KeySize == 26) {
    return copy_key(make_key_26(enc, tag, last), buf);
  } else {
    return copy_key(
        make_key_34(enc, tag, unodb::test_data::chain_key_filler, last), buf);
  }
}

/// key_a and key_b of the chain tests: make_deep_chain_key() keys with tag
/// 0x10 that differ only in their last bytes, 0x01 and 0x02.  The views point
/// into this struct, so it is neither copied nor moved.
/// \tparam KeySize Key length, 26 or 34
template <std::size_t KeySize>
struct chain_key_pair {
  chain_key_pair() = default;
  ~chain_key_pair() = default;
  chain_key_pair(const chain_key_pair&) = delete;
  chain_key_pair& operator=(const chain_key_pair&) = delete;
  chain_key_pair(chain_key_pair&&) = delete;
  chain_key_pair& operator=(chain_key_pair&&) = delete;

  /// key_a's bytes.
  std::array<std::byte, KeySize> buf_a{};
  /// key_b's bytes.
  std::array<std::byte, KeySize> buf_b{};
  /// Encoder of both keys.
  unodb::key_encoder enc;
  /// The key whose chain the test cuts or collapses into.
  unodb::key_view key_a{make_deep_chain_key<KeySize>(enc, 0x10, 0x01, buf_a)};
  /// The key that setup inserts and removes beside key_a.
  unodb::key_view key_b{make_deep_chain_key<KeySize>(enc, 0x10, 0x02, buf_b)};
};

/// chain_key_pair's keys and a sibling with tag 0x20, which setup inserts
/// beside key_a's chain below Root-I4: the one-byte key 0x20, or the
/// make_deep_chain_key() key with last byte 0x01.  sib points into this
/// struct, so it is neither copied nor moved.
/// \tparam KeySize Length of key_a and key_b, 26 or 34
/// \tparam SibSize Length of sib, 1 or KeySize
template <std::size_t KeySize, std::size_t SibSize>
struct chain_keys_beside_sib : chain_key_pair<KeySize> {
  static_assert(SibSize == 1 || SibSize == KeySize);

  /// sib's bytes.
  std::array<std::byte, SibSize> buf_sib{};
  /// The sibling of key_a's chain.
  unodb::key_view sib{make_sib()};

 private:
  /// Encode sib into buf_sib.
  [[nodiscard]] unodb::key_view make_sib() {
    if constexpr (SibSize == 1) {
      return copy_key(make_short_key(this->enc, 0x20), buf_sib);
    } else {
      return make_deep_chain_key<KeySize>(this->enc, 0x20, 0x01, buf_sib);
    }
  }
};

/// Insert \a key_a and \a key_b with \a val and each of \a sibs with its
/// value in \a sib_vals, then remove \a key_b.  Call it through
/// UNODB_ASSERT_NO_FATAL_FAILURE: a failed check returns only from here.
template <class Db, class Value>
void insert_chain_beside(
    Db& db, unodb::key_view key_a, unodb::key_view key_b,
    std::span<const unodb::key_view> sibs, Value val,
    std::type_identity_t<std::span<const Value>> sib_vals) {
  UNODB_ASSERT_EQ(sibs.size(), sib_vals.size());
  UNODB_ASSERT_TRUE(db.insert(key_a, val));
  UNODB_ASSERT_TRUE(db.insert(key_b, val));
  for (std::size_t i = 0; i < sibs.size(); ++i)
    UNODB_ASSERT_TRUE(db.insert(sibs[i], sib_vals[i]));
  const unodb::quiescent_state_on_scope_exit q{};
  UNODB_ASSERT_TRUE(db.remove(key_b));
}

/// Insert \a key_a, \a key_b and \a sibs, all with \a val, then remove
/// \a key_b.  Call it through UNODB_ASSERT_NO_FATAL_FAILURE.
template <class Db, class Value>
void insert_chain_beside(Db& db, unodb::key_view key_a, unodb::key_view key_b,
                         std::span<const unodb::key_view> sibs, Value val) {
  const std::vector<Value> sib_vals(sibs.size(), val);
  insert_chain_beside(db, key_a, key_b, sibs, val, sib_vals);
}

// CT3: T1 removes A (chain cut).  T2 removes the sibling, which leaves the
// cut-point parent Root-I4 with one child and a new version.  T1's Step 3
// then extends the chain through Root-I4 (Case B) and cuts from the root.
// The sibling is a one-byte key whose value sits in Root-I4's slot, so its
// removal neither collapses Root-I4 nor locks chain[0], which T1 holds.  On a
// leaf-backed db the removal would need chain[0]'s lock to collapse Root-I4
// into it, so it would restart until T1 resumes, which T1 does only after the
// removal returns; SiblingRemoveRestarts resumes T1 from that restart instead.
UNODB_TEST(OLCChainCutInterleaved, ConcurrentRemoveOfSibling) {
  using db_type = unodb::olc_db<unodb::key_view, std::uint64_t>;
  constexpr std::uint64_t val{0x42};

  db_type db;
  const chain_keys_beside_sib<26, 1> keys{};
  UNODB_ASSERT_NO_FATAL_FAILURE(insert_chain_beside(db, keys.key_a, keys.key_b,
                                                    std::array{keys.sib}, val));

  bool key_a_removed = false;
  bool sib_removed = false;
  const auto hits = run_parked_interleaving(
      unodb::detail::sync_after_chain_locked,
      [&] { key_a_removed = db.remove(keys.key_a); },
      [&] { sib_removed = db.remove(keys.sib); });
  // T1's park only: T2's remove leaves Root-I4 alive with one child, so T1's
  // Step 3 extends the chain through it and cuts from the root pointer
  // without restarting.
  UNODB_ASSERT_EQ(1U, hits);
  UNODB_ASSERT_TRUE(key_a_removed);
  UNODB_ASSERT_TRUE(sib_removed);
  const unodb::quiescent_state_on_scope_exit q{};
  UNODB_ASSERT_FALSE(db.get(keys.key_a).has_value());
  UNODB_ASSERT_FALSE(db.get(keys.sib).has_value());
  UNODB_ASSERT_TRUE(db.empty());
}

/// CT3b: T1 removes A (chain cut).  T2 inserts a fifth child into the full
/// cut-point parent Root-I4, which grows it into an I16 and obsoletes it.
/// T1's Step 3 finds Root-I4 obsolete and restarts.
UNODB_TEST(OLCChainCutInterleaved, ConcurrentGrowthOfCutPointParent) {
  using db_type = unodb::olc_db<unodb::key_view, unodb::value_view>;
  // key_a's and key_b's value, then one per sibling: the grown I16 and the I4
  // that replaces it copy the siblings' key bytes and children, and a
  // misaligned copy sends one sibling's lookup to another's keyless leaf.
  constexpr auto vals = distinct_values<5>();

  db_type db;
  unodb::key_encoder enc;

  const chain_key_pair<26> keys{};
  // One-byte siblings sit directly in Root-I4, filling it with chain[0].
  std::array<std::array<std::byte, 1>, 4> buf_sibs{};
  const std::array<unodb::key_view, 4> sibs{
      copy_key(make_short_key(enc, 0x20), buf_sibs[0]),
      copy_key(make_short_key(enc, 0x30), buf_sibs[1]),
      copy_key(make_short_key(enc, 0x40), buf_sibs[2]),
      copy_key(make_short_key(enc, 0x50), buf_sibs[3])};
  UNODB_ASSERT_NO_FATAL_FAILURE(insert_chain_beside(
      db, keys.key_a, keys.key_b, std::span{sibs}.first<3>(), vals[0],
      std::span{vals}.subspan<1, 3>()));

  bool key_a_removed = false;
  bool sib_inserted = false;
  const auto hits = run_parked_interleaving(
      unodb::detail::sync_after_chain_locked,
      [&] { key_a_removed = db.remove(keys.key_a); },
      [&] { sib_inserted = db.insert(sibs[3], vals[4]); });
  // T1's park and its retry: T2's insert obsoletes Root-I4, so T1's Step 3
  // restarts, and the retry passes the point again before cutting key_a's
  // chain from the I16.
  UNODB_ASSERT_EQ(2U, hits);
  UNODB_ASSERT_TRUE(key_a_removed);
  UNODB_ASSERT_TRUE(sib_inserted);
  const unodb::quiescent_state_on_scope_exit q{};
  UNODB_ASSERT_FALSE(db.get(keys.key_a).has_value());
  for (std::size_t i = 0; i < sibs.size(); ++i)
    ASSERT_VALUE_FOR_KEY(db_type, db, sibs[i], vals[i + 1]);
}

// CT4: ABA on chain node.  T2 inserts then removes a key that
// modifies chain[0]'s version but restores it to single-child.
UNODB_TEST(OLCChainCutInterleaved, ABAOnChainNode) {
  using db_type = unodb::olc_db<unodb::key_view, unodb::value_view>;
  // key_a, key_b, sib and t2_key, in that order.
  constexpr auto vals = distinct_values<4>();

  db_type db;
  unodb::key_encoder enc;

  constexpr auto X = unodb::test_data::chain_key_filler;
  constexpr auto Z = unodb::test_data::chain_key_filler_alt;

  std::array<std::byte, 34> buf_a{};
  std::array<std::byte, 34> buf_b{};
  std::array<std::byte, 34> buf_sib{};
  std::array<std::byte, 34> buf_t2{};
  const auto key_a = copy_key(make_key_34(enc, 0x10, X, 0x01), buf_a);
  const auto key_b = copy_key(make_key_34(enc, 0x10, X, 0x02), buf_b);
  const auto sib = copy_key(make_key_34(enc, 0x20, X, 0x01), buf_sib);
  const auto t2_key = copy_key(make_key_34(enc, 0x10, Z, 0x01), buf_t2);

  UNODB_ASSERT_TRUE(db.insert(key_a, vals[0]));
  UNODB_ASSERT_TRUE(db.insert(key_b, vals[1]));
  UNODB_ASSERT_TRUE(db.insert(sib, vals[2]));
  {
    const unodb::quiescent_state_on_scope_exit q{};
    UNODB_ASSERT_TRUE(db.remove(key_b));
  }

  bool t1_removed = false;
  bool t2_inserted = false;
  bool t2_removed = false;
  const auto hits = run_parked_interleaving(
      unodb::detail::sync_between_chain_locks,
      [&] { t1_removed = db.remove(key_a); },
      [&] {
        {
          const unodb::quiescent_state_on_scope_exit q{};
          t2_inserted = db.insert(t2_key, vals[3]);
        }
        t2_removed = db.remove(t2_key);
      });
  // T1's park and T2's remove of t2_key cutting its own chain; T1 does not
  // restart: Step 2 cannot lock the moved chain[0], which Step 3's Case B
  // locks once it has one child again, and Step 3 then locks Root-I4, which
  // T2 never wrote, at its descent version.
  UNODB_ASSERT_EQ(2U, hits);
  // A cut of the chain top that key_a shares would delete key_a with t2_key;
  // T1 would then restart, find key_a gone and return false without another
  // hit.
  UNODB_ASSERT_TRUE(t1_removed);
  UNODB_ASSERT_TRUE(t2_inserted);
  UNODB_ASSERT_TRUE(t2_removed);

  const unodb::quiescent_state_on_scope_exit q{};
  UNODB_ASSERT_FALSE(db.get(key_a).has_value());
  ASSERT_VALUE_FOR_KEY(db_type, db, sib, vals[2]);
  UNODB_ASSERT_FALSE(db.get(t2_key).has_value());
}

// ===================================================================
// Chain cut vs. a concurrent split that moves a node the cut will lock or
// unlink without obsoleting it.  The cut must check every link it relies on:
// that each chain node still leads to the node below it, and that the root
// pointer still points to the topmost node it rewrites.
// ===================================================================

/// CL1: T1 removes key_a and parks in Step 2 holding chain[3] and the nodes
/// below it.  T2 inserts a key that splits chain[1]'s prefix, which puts a new
/// I4 between chain[0] and chain[1].  chain[0]'s version moved but it is still
/// a one-child I4, so T1 must see that it no longer leads to chain[1] instead
/// of cutting the new I4.
UNODB_TEST(OLCChainCutVsSplit, SplitBelowChainTop) {
  // T1's park and its retry: T2's insert never reaches the point.  Step 2
  // leaves the moved chain[1] to Step 3, whose Case B locks it and then finds
  // that chain[0] no longer leads to it, and restarts.
  cut_34_byte_chain_during_insert(2, chain_key_filler_second_byte_changed);
}

/// CL2: CL1 without the sibling, so chain[0] is the root.  T1 would cut from
/// the root pointer after accepting chain[0], and must still see that chain[0]
/// no longer leads to chain[1].
UNODB_TEST(OLCChainCutVsSplit, SplitBelowRootChainTop) {
  constexpr auto X = unodb::test_data::chain_key_filler;
  unodb::key_encoder enc;
  std::array<std::byte, 34> buf_a{};
  std::array<std::byte, 34> buf_t2{};
  const auto key_a = copy_key(make_key_34(enc, 0x10, X, 0x01), buf_a);
  const auto t2_key = copy_key(
      make_key_34(enc, 0x10, chain_key_filler_second_byte_changed, 0x01),
      buf_t2);
  // T1's park and its retry, as in CL1, with the restart at the root chain[0].
  assert_remove_keeps_other_keys<unodb::test::key_view_olc_db>(
      unodb::detail::sync_between_chain_locks, 2, key_a, {}, t2_key);
}

/// CL3: key_a's chain is the whole tree.  T1 removes key_a and parks in
/// Step 2.  T2 inserts a key that differs in the first byte, which splits
/// chain[0]'s prefix and installs a new root I4 above it.  chain[0] still
/// leads to chain[1], so T1 must see that the root pointer no longer points to
/// chain[0] instead of clearing it.
UNODB_TEST(OLCChainCutVsSplit, RootSplitAboveChainTop) {
  constexpr auto X = unodb::test_data::chain_key_filler;
  unodb::key_encoder enc;
  std::array<std::byte, 34> buf_a{};
  std::array<std::byte, 34> buf_t2{};
  const auto key_a = copy_key(make_key_34(enc, 0x10, X, 0x01), buf_a);
  const auto t2_key = copy_key(make_key_34(enc, 0x20, X, 0x01), buf_t2);
  // T1's park and its retry: T2's insert never reaches the point.  Step 2
  // leaves the moved chain[0] to Step 3, whose Case B locks it and then finds
  // that the root pointer no longer points to it, and restarts.
  assert_remove_keeps_other_keys<unodb::test::key_view_olc_db>(
      unodb::detail::sync_between_chain_locks, 2, key_a, {}, t2_key);
}

/// CL4: the root is an I16 at its minimum size whose one-byte prefix key_a's
/// chain hangs below, so cutting that chain shrinks the root.  T1 removes key_a
/// and parks after Step 2.  T2 inserts a key that differs in the first byte,
/// which splits the I16's prefix and installs a new root I4 above it.  T1 must
/// see that the root pointer no longer points to the I16 instead of replacing
/// the new root with the shrunk I16.
UNODB_TEST(OLCChainCutVsSplit, RootSplitAboveShrinkingCutPointParent) {
  unodb::key_encoder enc;
  std::array<std::byte, 26> buf_a{};
  std::array<std::byte, 1> buf_t2{};
  constexpr std::uint8_t tag = 0x10;
  const auto key_a = copy_key(make_key_26(enc, tag, 0x01), buf_a);
  const auto t2_key = copy_key(make_short_key(enc, 0x20), buf_t2);
  // Two-byte siblings below the root's one-byte prefix tag, beside key_a's
  // chain at chain_key_filler_byte, make the root an I16 at its minimum size.
  // Their second bytes follow the filler byte, so that none of them is a prefix
  // of key_a.
  using inode_16 =
      unodb::detail::olc_inode_16<unodb::key_view, unodb::value_view>;
  constexpr std::size_t sib_count = inode_16::min_size - 1U;
  std::array<std::array<std::byte, 2>, sib_count> buf_sibs{};
  std::array<unodb::key_view, sib_count> sibs{};
  for (std::size_t i = 0; i < sibs.size(); ++i) {
    sibs[i] = copy_key(enc.reset()
                           .encode(tag)
                           .encode(static_cast<std::uint8_t>(
                               unodb::test_data::chain_key_filler_byte + 1 + i))
                           .get_key_view(),
                       buf_sibs[i]);
  }

  // T1's park and its retry: T2's insert never reaches the point.  Step 4.1
  // finds that the root pointer no longer points to the I16, and restarts.
  assert_remove_keeps_other_keys<unodb::test::key_view_olc_db>(
      unodb::detail::sync_after_chain_locked, 2, key_a, sibs, t2_key,
      t2_action::insert,
      []([[maybe_unused]] const unodb::test::key_view_olc_db& db) {
#ifdef UNODB_DETAIL_WITH_STATS
        // T1's retry after Step 4.1's restart shrinks the I16 once, below the
        // new root.
        UNODB_ASSERT_EQ(1U,
                        db.get_shrinking_inode_count<unodb::node_type::I16>());
#endif
      });
}

// ===================================================================
// I4 collapse vs. concurrent get.  Collapsing an I4 prepends its prefix
// and dispatch byte to the surviving child's key prefix.  A get that has
// validated the I4 and read-locked that child would read the merged prefix
// at the old depth, so the collapse must write-lock the child to make the
// get fail its version check.
// ===================================================================

/// Look \a key up in \a db and copy the value found, if any: its value_view
/// lives only until the caller's quiescent state.
template <class Db>
[[nodiscard]] std::optional<std::vector<std::byte>> get_value_copy(
    const Db& db, typename Db::key_type key) {
  const auto r = db.get(key);
  if (!r.has_value()) return {};
  return std::vector<std::byte>{r->begin(), r->end()};
}

/// CG1: Root-I4 holds a one-byte sibling and chain[0].  T1's get(key_a) parks
/// at chain[0] while T2 removes the sibling, which collapses Root-I4 into
/// chain[0] in remove_or_choose_subtree.
UNODB_TEST(OLCI4CollapseVsGet, SiblingRemoveCollapse) {
  using db_type = unodb::olc_db<unodb::key_view, unodb::value_view>;
  // key_a's value, then sib's.
  constexpr auto vals = distinct_values<2>();

  db_type db;
  const chain_keys_beside_sib<26, 1> keys{};
  UNODB_ASSERT_NO_FATAL_FAILURE(
      insert_chain_beside(db, keys.key_a, keys.key_b, std::array{keys.sib},
                          vals[0], std::span{vals}.subspan<1, 1>()));

  std::optional<std::vector<std::byte>> found;
  const auto [hits, removed] = parked_at_second_node(
      db, unodb::detail::sync_in_get_after_parent_unlock,
      [&] { found = get_value_copy(db, keys.key_a); }, keys.sib);
  // Root-I4 and chain[0], then the retry from the new root: chain[0], two
  // chain I4s, the bottom I4 and key_a's leaf.  The collapse moved chain[0]'s
  // version, so T1's prefix mismatch restarts instead of reporting key_a
  // absent.
  UNODB_ASSERT_EQ(7U, hits);
  UNODB_ASSERT_EQ(found,
                  (std::vector<std::byte>{vals[0].begin(), vals[0].end()}));
  UNODB_ASSERT_TRUE(removed);

  const unodb::quiescent_state_on_scope_exit q{};
  ASSERT_VALUE_FOR_KEY(db_type, db, keys.key_a, vals[0]);
  UNODB_ASSERT_FALSE(db.get(keys.sib).has_value());
}

/// CG2: Root-I4 holds two chains.  T1's get(sib) parks at the sibling chain's
/// top while T2 removes key_a, whose chain cut collapses Root-I4 into that
/// top in collapse_i4.
UNODB_TEST(OLCI4CollapseVsGet, ChainCutCollapse) {
  using db_type = unodb::olc_db<unodb::key_view, unodb::value_view>;
  // key_a's value, then sib's.
  constexpr auto vals = distinct_values<2>();

  db_type db;
  const chain_keys_beside_sib<26, 26> keys{};
  UNODB_ASSERT_NO_FATAL_FAILURE(
      insert_chain_beside(db, keys.key_a, keys.key_b, std::array{keys.sib},
                          vals[0], std::span{vals}.subspan<1, 1>()));

  std::optional<std::vector<std::byte>> found;
  const auto [hits, removed] = parked_at_second_node(
      db, unodb::detail::sync_in_get_after_parent_unlock,
      [&] { found = get_value_copy(db, keys.sib); }, keys.key_a);
  // Root-I4 and the sibling chain's top, then the retry from the new root:
  // that top, two chain I4s, the bottom I4 and sib's leaf.  The collapse moved
  // the top's version, so T1's prefix mismatch restarts instead of reporting
  // sib absent.
  UNODB_ASSERT_EQ(7U, hits);
  UNODB_ASSERT_EQ(found,
                  (std::vector<std::byte>{vals[1].begin(), vals[1].end()}));
  UNODB_ASSERT_TRUE(removed);

  const unodb::quiescent_state_on_scope_exit q{};
  UNODB_ASSERT_FALSE(db.get(keys.key_a).has_value());
  ASSERT_VALUE_FOR_KEY(db_type, db, keys.sib, vals[1]);
}

/// CG3: CG1 on a fixed-width tree.  Root-I4, prefix 00 x 6, holds 1's leaf at
/// 0x00 and, at 0x01, an I4 with an empty prefix over 0x100 and 0x101.  T1's
/// get(0x100) parks at that I4 while T2's remove(1) collapses Root-I4 into it.
UNODB_TEST(OLCI4CollapseVsGet, FixedWidthSiblingRemoveCollapse) {
  using db_type = unodb::olc_db<std::uint64_t, unodb::value_view>;
  // The values of 1, 0x100 and 0x101.
  constexpr auto vals = distinct_values<3>();

  db_type db;
  UNODB_ASSERT_TRUE(db.insert(1, vals[0]));
  UNODB_ASSERT_TRUE(db.insert(0x100, vals[1]));
  UNODB_ASSERT_TRUE(db.insert(0x101, vals[2]));

  std::optional<std::vector<std::byte>> found;
  const auto [hits, removed] = parked_at_second_node(
      db, unodb::detail::sync_in_get_after_parent_unlock,
      [&] { found = get_value_copy(db, 0x100); }, 1);
  // Root-I4 and the I4 below it, then the retry from that I4, now the root,
  // and 0x100's leaf.  The collapse moved the I4's version, so T1's prefix
  // mismatch restarts instead of reporting 0x100 absent.
  UNODB_ASSERT_EQ(4U, hits);
  UNODB_ASSERT_EQ(found,
                  (std::vector<std::byte>{vals[1].begin(), vals[1].end()}));
  UNODB_ASSERT_TRUE(removed);

  const unodb::quiescent_state_on_scope_exit q{};
  UNODB_ASSERT_FALSE(db.get(1).has_value());
  ASSERT_VALUE_FOR_KEY(db_type, db, 0x100, vals[1]);
  ASSERT_VALUE_FOR_KEY(db_type, db, 0x101, vals[2]);
}

/// What T1's scan saw first: whether the iterator was valid, and a copy of its
/// key and value.
struct first_entry {
  /// Whether the iterator was valid.
  bool valid{false};
  /// The entry's key bytes.
  std::vector<std::byte> key;
  /// The entry's value bytes.
  std::vector<std::byte> val;
};

/// Record \a it's current entry into \a out.
template <class Iterator>
void record_first_entry(const Iterator& it, first_entry& out) {
  out.valid = it.valid();
  if (!out.valid) return;
  const auto key = it.get_key().view();
  out.key.assign(key.begin(), key.end());
  const auto val = it.get_val();
  out.val.assign(val.begin(), val.end());
}

/// CS1: CG1's tree.  T1's first() parks at chain[0] while T2 removes the
/// sibling, collapsing Root-I4 into chain[0].  T1 must still return key_a.
UNODB_TEST(OLCI4CollapseVsScan, SiblingRemoveCollapse) {
  using db_type = unodb::olc_db<unodb::key_view, unodb::value_view>;
  // key_a's value, then sib's.
  constexpr auto vals = distinct_values<2>();

  db_type db;
  const chain_keys_beside_sib<26, 1> keys{};
  UNODB_ASSERT_NO_FATAL_FAILURE(
      insert_chain_beside(db, keys.key_a, keys.key_b, std::array{keys.sib},
                          vals[0], std::span{vals}.subspan<1, 1>()));

  first_entry seen;
  const auto [hits, removed] = parked_at_second_node(
      db, unodb::detail::sync_in_scan_after_parent_unlock,
      [&] {
        auto it = db.test_only_iterator();
        it.first();
        record_first_entry(it, seen);
      },
      keys.sib);
  // Root-I4 and chain[0], then first()'s retry from the new root: chain[0],
  // two chain I4s, the bottom I4 and key_a's leaf.
  UNODB_ASSERT_EQ(7U, hits);
  UNODB_ASSERT_TRUE(removed);
  UNODB_ASSERT_TRUE(seen.valid);
  UNODB_ASSERT_TRUE(std::ranges::equal(seen.key, keys.key_a));
  UNODB_ASSERT_TRUE(std::ranges::equal(seen.val, vals[0]));
}

/// CS2: CG2's tree.  T1's last() parks at the sibling chain's top while T2
/// removes key_a, whose chain cut collapses Root-I4 into that top.  T1 must
/// still return sib.
UNODB_TEST(OLCI4CollapseVsScan, ChainCutCollapse) {
  using db_type = unodb::olc_db<unodb::key_view, unodb::value_view>;
  // key_a's value, then sib's.
  constexpr auto vals = distinct_values<2>();

  db_type db;
  const chain_keys_beside_sib<26, 26> keys{};
  UNODB_ASSERT_NO_FATAL_FAILURE(
      insert_chain_beside(db, keys.key_a, keys.key_b, std::array{keys.sib},
                          vals[0], std::span{vals}.subspan<1, 1>()));

  first_entry seen;
  const auto [hits, removed] = parked_at_second_node(
      db, unodb::detail::sync_in_scan_after_parent_unlock,
      [&] {
        auto it = db.test_only_iterator();
        it.last();
        record_first_entry(it, seen);
      },
      keys.key_a);
  // Root-I4 and the sibling chain's top, then last()'s retry from the new
  // root: that top, two chain I4s, the bottom I4 and sib's leaf.
  UNODB_ASSERT_EQ(7U, hits);
  UNODB_ASSERT_TRUE(removed);
  UNODB_ASSERT_TRUE(seen.valid);
  UNODB_ASSERT_TRUE(std::ranges::equal(seen.key, keys.sib));
  UNODB_ASSERT_TRUE(std::ranges::equal(seen.val, vals[1]));
}

// ===================================================================
// I4 collapse vs. a short key.  A collapse lengthens the surviving inode's
// prefix in place, so a descent at the old depth can find its remaining key
// ending inside the merged prefix.  It must restart on the collapse instead of
// shifting the key past its end.
// ===================================================================

/// Build insert_short_collapse_tree()'s tree in \a db, run \a op parked at the
/// second hit of \a point while T2 removes short_sib, and check that \a point
/// was hit \a expected_hits times and that T2 removed short_sib.  Call it
/// through UNODB_ASSERT_NO_FATAL_FAILURE.
template <class Op>
void park_during_short_sib_remove(short_collapse_db& db,
                                  unodb::detail::sync_point& point,
                                  unsigned expected_hits, const Op& op) {
  UNODB_ASSERT_NO_FATAL_FAILURE(
      insert_short_collapse_tree(db, short_collapse_vals));
  const auto [hits, removed] = parked_at_second_node(db, point, op, short_sib);
  UNODB_ASSERT_EQ(expected_hits, hits);
  UNODB_ASSERT_TRUE(removed);
}

/// Check that \a db keeps \a bystander with \a bystander_val and no longer
/// holds short_sib.  Call it through UNODB_ASSERT_NO_FATAL_FAILURE.
void assert_short_sib_collapse_bystanders(
    const short_collapse_db& db, unodb::key_view bystander = short_key_b,
    unodb::value_view bystander_val = short_collapse_vals[1]) {
  const unodb::quiescent_state_on_scope_exit q{};
  ASSERT_VALUE_FOR_KEY(short_collapse_db, db, bystander, bystander_val);
  UNODB_ASSERT_FALSE(db.get(short_sib).has_value());
}

/// SK1: T1's get(short_key_a) parks at the I4 below Root-I4 while T2's
/// removal of short_sib collapses Root-I4 into it.
UNODB_TEST(OLCI4CollapseVsShortKey, GetRestarts) {
  short_collapse_db db;
  std::optional<std::vector<std::byte>> found;
  // Root-I4 and the I4 below it, then the retry from that I4, now the root,
  // and short_key_a's leaf.  The collapse moved the I4's version, so T1's
  // check before the shift restarts it.
  UNODB_ASSERT_NO_FATAL_FAILURE(park_during_short_sib_remove(
      db, unodb::detail::sync_in_get_after_parent_unlock, 4,
      [&] { found = get_value_copy(db, short_key_a); }));
  UNODB_ASSERT_EQ(found,
                  (std::vector<std::byte>{short_collapse_vals[0].begin(),
                                          short_collapse_vals[0].end()}));

  ASSERT_VALUE_FOR_KEY(short_collapse_db, db, short_key_a,
                       short_collapse_vals[0]);
  UNODB_ASSERT_NO_FATAL_FAILURE(assert_short_sib_collapse_bystanders(db));
}

/// SK2: as SK1, with T1 removing short_key_a.
UNODB_TEST(OLCI4CollapseVsShortKey, RemoveRestarts) {
  short_collapse_db db;
  // MSVC's analysis misses the write in the lambda below.
  UNODB_DETAIL_DISABLE_MSVC_WARNING(26496)
  bool key_a_removed = false;
  UNODB_DETAIL_RESTORE_MSVC_WARNINGS()
  // Root-I4 and the I4 below it, T2's pass through Root-I4, then the retry
  // from that I4, now the root.  The collapse obsoleted Root-I4, so T1's check
  // of its parent before the shift restarts it.
  UNODB_ASSERT_NO_FATAL_FAILURE(park_during_short_sib_remove(
      db, unodb::detail::sync_before_prefix_load, 4,
      [&] { key_a_removed = db.remove(short_key_a); }));
  UNODB_ASSERT_TRUE(key_a_removed);

  const unodb::quiescent_state_on_scope_exit q{};
  UNODB_ASSERT_FALSE(db.get(short_key_a).has_value());
  UNODB_ASSERT_NO_FATAL_FAILURE(assert_short_sib_collapse_bystanders(db));
}

/// SK3: as SK1, with T1 inserting short_key_a again.
UNODB_TEST(OLCI4CollapseVsShortKey, InsertRestarts) {
  short_collapse_db db;
  // MSVC's analysis misses the write in the lambda below.
  UNODB_DETAIL_DISABLE_MSVC_WARNING(26496)
  bool inserted = true;
  UNODB_DETAIL_RESTORE_MSVC_WARNINGS()
  // Root-I4 and the I4 below it, T2's pass through Root-I4 in its remove,
  // then the retry from that I4, now the root.  The collapse obsoleted
  // Root-I4, so T1's check of its parent before the shift restarts it.
  UNODB_ASSERT_NO_FATAL_FAILURE(park_during_short_sib_remove(
      db, unodb::detail::sync_before_prefix_load, 4,
      [&] { inserted = db.insert(short_key_a, short_collapse_vals[0]); }));
  UNODB_ASSERT_FALSE(inserted);

  ASSERT_VALUE_FOR_KEY(short_collapse_db, db, short_key_a,
                       short_collapse_vals[0]);
  UNODB_ASSERT_NO_FATAL_FAILURE(assert_short_sib_collapse_bystanders(db));
}

/// SK4: as SK1, with T1 upserting short_key_a and keeping its value.
UNODB_TEST(OLCI4CollapseVsShortKey, UpsertRestarts) {
  short_collapse_db db;
  // MSVC's analysis misses the writes in the lambdas below.
  UNODB_DETAIL_DISABLE_MSVC_WARNING(26496)
  bool inserted = true;
  unsigned fn_calls = 0;
  UNODB_DETAIL_RESTORE_MSVC_WARNINGS()
  // Root-I4 and the I4 below it, T2's pass through Root-I4 in its remove,
  // then the retry from that I4, now the root.  The collapse obsoleted
  // Root-I4, so T1's check of its parent before the shift restarts it.
  UNODB_ASSERT_NO_FATAL_FAILURE(park_during_short_sib_remove(
      db, unodb::detail::sync_before_prefix_load, 4, [&] {
        inserted = db.upsert(short_key_a, short_collapse_vals[0],
                             [&fn_calls](unodb::value_view&) {
                               ++fn_calls;
                               return unodb::upsert_action::keep;
                             });
      }));
  UNODB_ASSERT_FALSE(inserted);
  // T1 parks before it reaches the leaf, so only its retry runs the lambda.
  UNODB_ASSERT_EQ(1U, fn_calls);

  ASSERT_VALUE_FOR_KEY(short_collapse_db, db, short_key_a,
                       short_collapse_vals[0]);
  UNODB_ASSERT_NO_FATAL_FAILURE(assert_short_sib_collapse_bystanders(db));
}

/// SK5: as SK1, with T1 upserting short_key_a and erasing it.  T1 parks in
/// the erase's own descent.
UNODB_TEST(OLCI4CollapseVsShortKey, UpsertEraseRestarts) {
  short_collapse_db db;
  // MSVC's analysis misses the writes in the lambdas below.
  UNODB_DETAIL_DISABLE_MSVC_WARNING(26496)
  bool inserted = true;
  unsigned fn_calls = 0;
  UNODB_DETAIL_RESTORE_MSVC_WARNINGS()
  // The erase's descent through Root-I4 and the I4 below it, then the retried
  // upsert's erase from that I4, now the root.  T1 restarts on reaching the
  // end of its key inside the merged prefix.
  UNODB_ASSERT_NO_FATAL_FAILURE(park_during_short_sib_remove(
      db, unodb::detail::sync_in_upsert_erase_before_prefix_load, 3, [&] {
        inserted = db.upsert(short_key_a, short_collapse_vals[0],
                             [&fn_calls](unodb::value_view&) {
                               ++fn_calls;
                               return unodb::upsert_action::erase;
                             });
      }));
  UNODB_ASSERT_FALSE(inserted);
  // T1 parks after the lambda returns erase, and its retry runs it again.
  UNODB_ASSERT_EQ(2U, fn_calls);

  const unodb::quiescent_state_on_scope_exit q{};
  UNODB_ASSERT_FALSE(db.get(short_key_a).has_value());
  UNODB_ASSERT_NO_FATAL_FAILURE(assert_short_sib_collapse_bystanders(db));
}

/// SK6: as SK1, with T1 scanning forward from short_key_a.  T1 parks in the
/// scan's seek.
UNODB_TEST(OLCI4CollapseVsShortKey, SeekForwardRestarts) {
  short_collapse_db db;
  key_list keys;
  // The seek through Root-I4 and the I4 below it, then the retried seek from
  // that I4, now the root.  T1 restarts on reaching the end of its key inside
  // the merged prefix.
  UNODB_ASSERT_NO_FATAL_FAILURE(park_during_short_sib_remove(
      db, unodb::detail::sync_in_seek_before_prefix_load, 3,
      [&] { keys = keys_scanned_from(db, short_key_a, true); }));
  UNODB_ASSERT_EQ(keys,
                  (key_list{key_bytes(short_key_a), key_bytes(short_key_b)}));

  ASSERT_VALUE_FOR_KEY(short_collapse_db, db, short_key_a,
                       short_collapse_vals[0]);
  UNODB_ASSERT_NO_FATAL_FAILURE(assert_short_sib_collapse_bystanders(db));
}

/// SK7: as SK6, with T1 scanning in reverse.
UNODB_TEST(OLCI4CollapseVsShortKey, SeekReverseRestarts) {
  short_collapse_db db;
  key_list keys;
  // The seek through Root-I4 and the I4 below it, then the retried seek from
  // that I4, now the root.  T1 restarts on reaching the end of its key inside
  // the merged prefix.
  UNODB_ASSERT_NO_FATAL_FAILURE(park_during_short_sib_remove(
      db, unodb::detail::sync_in_seek_before_prefix_load, 3,
      [&] { keys = keys_scanned_from(db, short_key_a, false); }));
  UNODB_ASSERT_EQ(keys, (key_list{key_bytes(short_key_a)}));

  ASSERT_VALUE_FOR_KEY(short_collapse_db, db, short_key_a,
                       short_collapse_vals[0]);
  UNODB_ASSERT_NO_FATAL_FAILURE(assert_short_sib_collapse_bystanders(db));
}

/// SK8: as SK5, with T1 upserting short_key_b and erasing it.  The merged
/// prefix 07 00 mismatches T1's remaining key 08, so T1 restarts from the
/// erase descent's prefix-mismatch arm.
UNODB_TEST(OLCI4CollapseVsShortKey, UpsertEraseMismatchRestarts) {
  short_collapse_db db;
  // MSVC's analysis misses the writes in the lambdas below.
  UNODB_DETAIL_DISABLE_MSVC_WARNING(26496)
  bool inserted = true;
  unsigned fn_calls = 0;
  UNODB_DETAIL_RESTORE_MSVC_WARNINGS()
  // As in SK5: the erase's descent through Root-I4 and the I4 below it, then
  // the retried upsert's erase from that I4, now the root.
  UNODB_ASSERT_NO_FATAL_FAILURE(park_during_short_sib_remove(
      db, unodb::detail::sync_in_upsert_erase_before_prefix_load, 3, [&] {
        inserted = db.upsert(short_key_b, short_collapse_vals[1],
                             [&fn_calls](unodb::value_view&) {
                               ++fn_calls;
                               return unodb::upsert_action::erase;
                             });
      }));
  UNODB_ASSERT_FALSE(inserted);
  // The retry runs the lambda again before erasing, which the hit count cannot
  // show.
  UNODB_ASSERT_EQ(2U, fn_calls);

  const unodb::quiescent_state_on_scope_exit q{};
  UNODB_ASSERT_FALSE(db.get(short_key_b).has_value());
  UNODB_ASSERT_NO_FATAL_FAILURE(assert_short_sib_collapse_bystanders(
      db, short_key_a, short_collapse_vals[0]));
}

/// SK9: as SK2, with T1 removing short_key_b.  The merged prefix 07 00
/// mismatches T1's remaining key 08, so T1 restarts from the prefix-mismatch
/// arm's parent unlock on the obsolete Root-I4.
UNODB_TEST(OLCI4CollapseVsShortKey, RemoveMismatchRestarts) {
  short_collapse_db db;
  // MSVC's analysis misses the write in the lambda below.
  UNODB_DETAIL_DISABLE_MSVC_WARNING(26496)
  bool removed = false;
  UNODB_DETAIL_RESTORE_MSVC_WARNINGS()
  // As in SK2: Root-I4 and the I4 below it, T2's pass through Root-I4, then
  // the retry from that I4, now the root.
  UNODB_ASSERT_NO_FATAL_FAILURE(park_during_short_sib_remove(
      db, unodb::detail::sync_before_prefix_load, 4,
      [&] { removed = db.remove(short_key_b); }));
  UNODB_ASSERT_TRUE(removed);

  const unodb::quiescent_state_on_scope_exit q{};
  UNODB_ASSERT_FALSE(db.get(short_key_b).has_value());
  UNODB_ASSERT_NO_FATAL_FAILURE(assert_short_sib_collapse_bystanders(
      db, short_key_a, short_collapse_vals[0]));
}

// ===================================================================
// I4 collapse or prefix split vs. an upsert of a value-in-slot key.  Both
// rewrite a node's prefix in place, and try_upsert checks the parent before it
// read-locks the next node, so after a rewrite in between the upsert locks the
// node at its rewritten version, and only the parent shows the rewrite.  These
// tests park the upsert in that gap.
// ===================================================================

/// The tree of the value-in-slot upsert tests.
using vis_upsert_db = unodb::test::key_view_u64val_olc_db;

/// Run an upsert of \a new_key with value 4 on \a db, parked at the top of its
/// descent step at the node below Root-I4, after checking Root-I4, while T2
/// runs \a interleaved, which rewrites that node's prefix so that \a new_key
/// now reaches another key's value slot.  Check that the upsert reached that
/// slot once and restarted there without calling its lambda, that its retry
/// inserted \a new_key, and that the sync point was hit \a expected_hits
/// times.  Call it through UNODB_ASSERT_NO_FATAL_FAILURE.
template <class Interleaved>
void assert_upsert_restarts_before_lambda(vis_upsert_db& db,
                                          unodb::key_view new_key,
                                          unsigned expected_hits,
                                          const Interleaved& interleaved) {
  unsigned dup_found_hits = 0;
  // Declared after the counter, so that it disarms the hook first.
  const sync_point_guard dup_found_guard{
      unodb::detail::sync_after_upsert_dup_found};
  unodb::detail::sync_after_upsert_dup_found.arm(
      [&dup_found_hits]() noexcept { ++dup_found_hits; });
  unsigned fn_calls = 0;
  bool inserted = false;
  const auto hits = run_parked_interleaving<2>(
      unodb::detail::sync_in_upsert_before_node_lock,
      [&] {
        inserted = db.upsert(new_key, 4, [&fn_calls](std::uint64_t&) {
          ++fn_calls;
          return unodb::upsert_action::keep;
        });
      },
      interleaved);
  UNODB_ASSERT_EQ(expected_hits, hits);
  // The upsert reached the other key's slot once, through the rewritten
  // prefix, and restarted there; its retry inserts without finding a
  // duplicate.
  UNODB_ASSERT_EQ(1U, dup_found_hits);
  UNODB_ASSERT_EQ(0U, fn_calls);
  UNODB_ASSERT_TRUE(inserted);
  ASSERT_VALUE_FOR_KEY(vis_upsert_db, db, new_key, 4);
}

/// UC1: on olc_db<key_view, std::uint64_t>, Root-I4 with an empty prefix
/// holds, at 05, an I4 with an empty prefix and the values of 05 07 and 05 08
/// in its slots, and at 10 a 26-byte key's chain.  T1 upserts 05 05 07 while
/// T2 removes the 26-byte key, whose chain cut collapses Root-I4 into that I4.
/// Its prefix becomes 05, so T1's remaining key 05 07 reaches 05 07's slot.
/// T1 must restart on the obsolete Root-I4 instead of running its lambda on
/// 05 07's value.
UNODB_TEST(OLCI4CollapseVsUpsert, RestartsBeforeLambda) {
  constexpr std::array key_a{std::byte{0x05}, std::byte{0x07}};
  constexpr std::array key_b{std::byte{0x05}, std::byte{0x08}};
  constexpr std::array new_key{std::byte{0x05}, std::byte{0x05},
                               std::byte{0x07}};

  vis_upsert_db db;
  unodb::key_encoder enc;
  std::array<std::byte, 26> buf_chain{};
  const auto chain_key = copy_key(make_key_26(enc, 0x10, 0x01), buf_chain);
  UNODB_ASSERT_TRUE(db.insert(unodb::key_view{key_a}, 1));
  UNODB_ASSERT_TRUE(db.insert(unodb::key_view{key_b}, 2));
  UNODB_ASSERT_TRUE(db.insert(chain_key, 3));

  // MSVC's analysis misses the write in the lambda below.
  UNODB_DETAIL_DISABLE_MSVC_WARNING(26496)
  bool removed = false;
  UNODB_DETAIL_RESTORE_MSVC_WARNINGS()
  // Root-I4 and the I4 below it, then the retried upsert from that I4, now the
  // root, which inserts 05 05 07 there.
  UNODB_ASSERT_NO_FATAL_FAILURE(assert_upsert_restarts_before_lambda(
      db, unodb::key_view{new_key}, 3,
      [&] { removed = db.remove(chain_key); }));
  UNODB_ASSERT_TRUE(removed);

  ASSERT_VALUE_FOR_KEY(vis_upsert_db, db, unodb::key_view{key_a}, 1);
  ASSERT_VALUE_FOR_KEY(vis_upsert_db, db, unodb::key_view{key_b}, 2);
  const unodb::quiescent_state_on_scope_exit q{};
  UNODB_ASSERT_FALSE(db.get(chain_key).has_value());
}

/// US1: on olc_db<key_view, std::uint64_t>, Root-I4 with an empty prefix
/// holds, at 05, an I4 with prefix 01 02 and the values of 05 01 02 07 and
/// 05 01 02 08 in its slots, and at 10 a 26-byte key's chain.  T1 upserts
/// 05 02 07 while T2 inserts 05 03 07, which splits that I4's prefix at its
/// first byte: a new I4 at 05 holds it at 01, and its prefix becomes 02, so
/// T1's remaining key 02 07 reaches 05 01 02 07's slot.  T1 must restart on
/// Root-I4, whose slot T2 rewrote, instead of running its lambda on
/// 05 01 02 07's value.
UNODB_TEST(OLCPrefixSplitVsUpsert, RestartsBeforeLambda) {
  constexpr std::array key_a{std::byte{0x05}, std::byte{0x01}, std::byte{0x02},
                             std::byte{0x07}};
  constexpr std::array key_b{std::byte{0x05}, std::byte{0x01}, std::byte{0x02},
                             std::byte{0x08}};
  constexpr std::array new_key{std::byte{0x05}, std::byte{0x02},
                               std::byte{0x07}};
  constexpr std::array split_key{std::byte{0x05}, std::byte{0x03},
                                 std::byte{0x07}};

  vis_upsert_db db;
  unodb::key_encoder enc;
  std::array<std::byte, 26> buf_chain{};
  const auto chain_key = copy_key(make_key_26(enc, 0x10, 0x01), buf_chain);
  UNODB_ASSERT_TRUE(db.insert(unodb::key_view{key_a}, 1));
  UNODB_ASSERT_TRUE(db.insert(unodb::key_view{key_b}, 2));
  UNODB_ASSERT_TRUE(db.insert(chain_key, 3));

  // MSVC's analysis misses the write in the lambda below.
  UNODB_DETAIL_DISABLE_MSVC_WARNING(26496)
  bool split_inserted = false;
  UNODB_DETAIL_RESTORE_MSVC_WARNINGS()
  // Root-I4 and the I4 below it, then the retried upsert's Root-I4 and the new
  // I4 at 05, which inserts 05 02 07 there.
  UNODB_ASSERT_NO_FATAL_FAILURE(assert_upsert_restarts_before_lambda(
      db, unodb::key_view{new_key}, 4,
      [&] { split_inserted = db.insert(unodb::key_view{split_key}, 5); }));
  UNODB_ASSERT_TRUE(split_inserted);

  ASSERT_VALUE_FOR_KEY(vis_upsert_db, db, unodb::key_view{key_a}, 1);
  ASSERT_VALUE_FOR_KEY(vis_upsert_db, db, unodb::key_view{key_b}, 2);
  ASSERT_VALUE_FOR_KEY(vis_upsert_db, db, unodb::key_view{split_key}, 5);
  ASSERT_VALUE_FOR_KEY(vis_upsert_db, db, chain_key, 3);
}

// ===================================================================
// I4 collapse vs. a writer that holds the surviving child.  A collapse
// write-locks a surviving inode without waiting and restarts if it is held,
// because the holder may be waiting for one of the collapsing writer's locks.
// Here the holder is a chain cut parked after Step 2, which holds its whole
// chain.  It cannot finish until it is resumed, so the collapsing writer
// resumes it from a later sync point and waits for it to return.  A collapse
// that waited for the survivor instead would never reach that sync point, so
// NW1 and NW2 would deadlock until the ctest timeout fails them.
// ===================================================================

/// Run \a parked parked at the first hit of \a park_point while \a interleaved
/// runs.  \a interleaved resumes \a parked at its own \a ReleaseHit-th hit of
/// \a release_point and then waits there for \a parked to return; if
/// \a interleaved returns first, \a parked resumes then.  Each callable runs
/// in a quiescent-state scope on its own QSBR thread.  If \a parked returns
/// without parking, the test fails instead of hanging: both points are
/// disarmed and \a interleaved then runs without a park.
/// \return The hit counts of \a park_point and \a release_point: both
/// threads' hits, or only \a parked's if the park was missed.
template <unsigned ReleaseHit, class Parked, class Interleaved>
[[nodiscard]] std::pair<unsigned, unsigned> run_parked_until_released(
    unodb::detail::sync_point& park_point,
    unodb::detail::sync_point& release_point, const Parked& parked,
    const Interleaved& interleaved) {
  unsigned park_hits = 0;
  unsigned release_hits = 0;
  unsigned interleaved_release_hits = 0;
  bool parked_reached = false;
  bool released = false;
  std::thread::id parked_id;
  unodb::detail::thread_sync park_sync;
  unodb::detail::thread_sync resume_sync;
  unodb::detail::thread_sync return_sync;
  const sync_point_guard park_guard{park_point};
  const sync_point_guard release_guard{release_point};
  park_point.arm([&park_hits, &parked_reached, &park_sync, &resume_sync] {
    if (++park_hits != 1) return;
    parked_reached = true;
    park_sync.notify();
    resume_sync.wait();
  });
  // parked_id is written before parked() runs, and the interleaved thread
  // hits release_point only after the park, so this read races with nothing.
  // A release on the parked thread's own hit would wait for itself.
  release_point.arm([&release_hits, &interleaved_release_hits, &parked_id,
                     &released, &resume_sync, &return_sync] {
    ++release_hits;
    if (std::this_thread::get_id() == parked_id) return;
    if (++interleaved_release_hits != ReleaseHit) return;
    released = true;
    resume_sync.notify();
    return_sync.wait();
  });

  unodb::this_thread().qsbr_pause();
  auto interleaved_thread =
      unodb::qsbr_thread([&interleaved, &released, &park_sync, &resume_sync] {
        park_sync.wait();
        {
          const unodb::quiescent_state_on_scope_exit q{};
          interleaved();
        }
        if (!released) resume_sync.notify();
      });
  auto parked_thread =
      unodb::qsbr_thread([&parked, &parked_id, &released, &return_sync] {
        parked_id = std::this_thread::get_id();
        {
          const unodb::quiescent_state_on_scope_exit q{};
          parked();
        }
        if (released) return_sync.notify();
      });
  parked_thread.join();
  // Only the park hook writes parked_reached, on the parked thread, so this
  // read races with nothing; park_hits may still be counting interleaved hits.
  if (!parked_reached) {
    // The interleaved operation may pass either point.
    release_after_missed_park(park_sync, resume_sync, park_point,
                              release_point);
  }
  interleaved_thread.join();
  unodb::this_thread().qsbr_resume();
  unodb::this_thread().quiescent();
  if (!parked_reached) ADD_FAILURE() << "parked returned before its park";
  return {park_hits, release_hits};
}

/// A parked operation that never reaches its park point must fail the test
/// rather than leave the interleaved thread waiting for the park forever, and
/// the interleaved operation must then run and find both points disarmed, so
/// that it neither waits in a release nor parks.  ReleaseHit is one more than
/// its hits of the release point, so that a release hit the disarm fails to
/// stop is counted instead of releasing.  The park point is checked for a hook
/// instead of hit, since a hit it fails to stop would park the interleaved
/// operation for good.
UNODB_TEST(RunParkedUntilReleased, MissedParkDisarmsBeforeRelease) {
  unodb::detail::sync_point park_point;
  unodb::detail::sync_point release_point;
  bool interleaved_ran = false;
  bool park_point_armed = true;
  const auto interleaved = [&park_point, &release_point, &interleaved_ran,
                            &park_point_armed]() noexcept {
    interleaved_ran = true;
    unodb::detail::sync(release_point);
    park_point_armed = static_cast<bool>(park_point.hook_);
  };
  std::pair<unsigned, unsigned> hits{};
  EXPECT_NONFATAL_FAILURE(
      {
        const unodb::test::intercepted_failure_guard guard;
        hits = run_parked_until_released<2>(
            park_point, release_point, []() noexcept {}, interleaved);
      },
      "returned before its park");
  UNODB_ASSERT_TRUE(interleaved_ran);
  // The release hook did not run: the interleaved hit came after the disarm.
  UNODB_ASSERT_EQ(0U, hits.second);
  UNODB_ASSERT_FALSE(park_point_armed);
}

/// The parked operation's own hits of the release point must not release it,
/// or it would wait there for itself to return.
UNODB_TEST(RunParkedUntilReleased, OwnReleaseHitDoesNotRelease) {
  const auto [park_hits, release_hits] = run_parked_until_released<1>(
      unodb::detail::sync_after_chain_locked,
      unodb::detail::sync_between_chain_locks,
      [] {
        unodb::detail::sync_after_chain_locked.hit();
        unodb::detail::sync_between_chain_locks.hit();
      },
      []() noexcept {});
  UNODB_ASSERT_EQ(1U, park_hits);
  UNODB_ASSERT_EQ(1U, release_hits);
}

/// NW1: Root-I4 holds chain[0] and a one-byte sibling leaf.  T1's chain cut
/// of key_a parks holding chain[0].  T2's removal of the sibling would
/// collapse Root-I4 into chain[0], so it fails to lock chain[0] and restarts;
/// its retry resumes T1.  T1 cannot collapse Root-I4 into the keyless sibling
/// leaf, so T2's pending write guards fail and its next retry cuts the
/// remaining one-child chain from the root.
UNODB_TEST(OLCI4CollapseVsHeldSurvivor, SiblingRemoveRestarts) {
  using db_type = unodb::olc_db<unodb::key_view, unodb::value_view>;
  constexpr auto val_bytes = std::array<std::byte, 1>{std::byte{0x42}};
  const auto val = unodb::value_view{val_bytes};

  db_type db;
  const chain_keys_beside_sib<26, 1> keys{};
  UNODB_ASSERT_NO_FATAL_FAILURE(insert_chain_beside(db, keys.key_a, keys.key_b,
                                                    std::array{keys.sib}, val));

  bool key_a_removed = false;
  bool sib_removed = false;
  const auto [chain_hits, remove_hits] = run_parked_until_released<2>(
      unodb::detail::sync_after_chain_locked,
      unodb::detail::sync_before_remove_write_guard,
      [&] { key_a_removed = db.remove(keys.key_a); },
      [&] { sib_removed = db.remove(keys.sib); });
  // T1's park, and T2's last retry, which cuts the one-child chain.
  UNODB_ASSERT_EQ(2U, chain_hits);
  // T2's first attempt, and the retry after it failed to lock chain[0].
  UNODB_ASSERT_EQ(2U, remove_hits);
  UNODB_ASSERT_TRUE(key_a_removed);
  UNODB_ASSERT_TRUE(sib_removed);
  UNODB_ASSERT_TRUE(db.empty());
}

/// NW2: Root-I4 holds two four-level chains.  T1's chain cut of key_a parks
/// holding its chain, top included.  T2's chain cut of sib would collapse
/// Root-I4 into T1's chain top, so it fails to lock it and restarts; its retry
/// resumes T1 in Step 2, before it locks its own chain top.
UNODB_TEST(OLCI4CollapseVsHeldSurvivor, ChainCutRestarts) {
  using db_type = unodb::olc_db<unodb::key_view, unodb::value_view>;
  constexpr auto val_bytes = std::array<std::byte, 1>{std::byte{0x42}};
  const auto val = unodb::value_view{val_bytes};

  db_type db;
  const chain_keys_beside_sib<34, 34> keys{};
  UNODB_ASSERT_NO_FATAL_FAILURE(insert_chain_beside(db, keys.key_a, keys.key_b,
                                                    std::array{keys.sib}, val));

  bool key_a_removed = false;
  bool sib_removed = false;
  const auto [chain_hits, between_hits] = run_parked_until_released<2>(
      unodb::detail::sync_after_chain_locked,
      unodb::detail::sync_between_chain_locks,
      [&] { key_a_removed = db.remove(keys.key_a); },
      [&] { sib_removed = db.remove(keys.sib); });
  // T1, T2's first attempt, T2's retry after it failed to lock T1's chain
  // top (T2's second between hit releases T1 here), and T2's next retry: T1
  // collapsed Root-I4 into T2's chain top, so T2's Step 3 restarts on it.
  UNODB_ASSERT_EQ(4U, chain_hits);
  UNODB_ASSERT_EQ(4U, between_hits);
  UNODB_ASSERT_TRUE(key_a_removed);
  UNODB_ASSERT_TRUE(sib_removed);
  UNODB_ASSERT_TRUE(db.empty());
}

/// NW3: Root-I4, with a one-byte prefix, holds key_a's chain top (seven-byte
/// prefix) and a two-byte sibling leaf.  T1's chain cut of key_a parks holding
/// that chain top.  Removing the sibling would merge 1 + 1 + 7 > 7 prefix
/// bytes, so Root-I4 keeps its one child: T2 does not lock T1's chain top, and
/// finishes without restarting.
UNODB_TEST(OLCI4CollapseVsHeldSurvivor, OverflowingMergeSkipsSurvivorLock) {
  using db_type = unodb::olc_db<unodb::key_view, unodb::value_view>;
  constexpr auto val_bytes = std::array<std::byte, 1>{std::byte{0x42}};
  const auto val = unodb::value_view{val_bytes};

  db_type db;
  unodb::key_encoder enc;

  const chain_key_pair<26> keys{};
  std::array<std::byte, 2> buf_sib{};
  const auto sib = copy_key(enc.reset()
                                .encode(std::uint8_t{0x10})
                                .encode(std::uint8_t{0x43})
                                .get_key_view(),
                            buf_sib);
  UNODB_ASSERT_NO_FATAL_FAILURE(
      insert_chain_beside(db, keys.key_a, keys.key_b, std::array{sib}, val));

  bool key_a_removed = false;
  bool sib_removed = false;
  const auto [chain_hits, remove_hits] = run_parked_until_released<2>(
      unodb::detail::sync_after_chain_locked,
      unodb::detail::sync_before_remove_write_guard,
      [&] { key_a_removed = db.remove(keys.key_a); },
      [&] { sib_removed = db.remove(sib); });
  // T1's park only: T2 leaves Root-I4 with one child, so T1's Step 3 extends
  // the chain through it.
  UNODB_ASSERT_EQ(1U, chain_hits);
  // T2's only attempt: no restart, so no release.
  UNODB_ASSERT_EQ(1U, remove_hits);
  UNODB_ASSERT_TRUE(key_a_removed);
  UNODB_ASSERT_TRUE(sib_removed);
  UNODB_ASSERT_TRUE(db.empty());
}

/// NW4: Root-I4 holds key_a's chain and a 33-byte sibling's chain, whose top,
/// built from depth 1, keeps a seven-byte prefix.  T1's chain cut of the
/// sibling parks holding that top.  T2's chain cut of key_a would merge
/// 0 + 1 + 7 > 7 prefix bytes, so Root-I4 keeps its one child: T2 does not
/// lock T1's chain top, and finishes without restarting.
UNODB_TEST(OLCI4CollapseVsHeldSurvivor,
           ChainCutOverflowingMergeSkipsSurvivorLock) {
  using db_type = unodb::olc_db<unodb::key_view, unodb::value_view>;
  constexpr auto val_bytes = std::array<std::byte, 1>{std::byte{0x42}};
  const auto val = unodb::value_view{val_bytes};

  db_type db;
  unodb::key_encoder enc;

  constexpr auto X = unodb::test_data::chain_key_filler;
  const chain_key_pair<34> keys{};
  std::array<std::byte, 33> buf_sib{};
  const auto sib = copy_key(
      make_key_33(enc, 0x20, X, unodb::test_data::chain_key_filler_byte),
      buf_sib);

  UNODB_ASSERT_NO_FATAL_FAILURE(
      insert_chain_beside(db, keys.key_a, keys.key_b, std::array{sib}, val));

  bool sib_removed = false;
  bool key_a_removed = false;
  const auto [chain_hits, between_hits] = run_parked_until_released<2>(
      unodb::detail::sync_after_chain_locked,
      unodb::detail::sync_between_chain_locks,
      [&] { sib_removed = db.remove(sib); },
      [&] { key_a_removed = db.remove(keys.key_a); });
  // T1's park and T2's only attempt; T2 never reaches the release hit.  T1
  // then extends its chain through the one-child Root-I4 and cuts from the
  // root.
  UNODB_ASSERT_EQ(2U, chain_hits);
  UNODB_ASSERT_EQ(2U, between_hits);
  UNODB_ASSERT_TRUE(sib_removed);
  UNODB_ASSERT_TRUE(key_a_removed);
  UNODB_ASSERT_TRUE(db.empty());
}

/// Encode a 27-byte key: tag 0x10, \a byte1, three filler words, then \a last.
/// Keys that differ at \a byte1 hang below one inode that branches on byte 1,
/// each in its own chain whose top has an empty prefix, so removing one
/// collapses that inode into the other's chain top.
[[nodiscard]] unodb::key_view make_key_27(unodb::key_encoder& enc
                                          UNODB_DETAIL_LIFETIMEBOUND,
                                          std::uint8_t byte1,
                                          std::uint8_t last) {
  constexpr auto X = unodb::test_data::chain_key_filler;
  return enc.reset()
      .encode(std::uint8_t{0x10})
      .encode(byte1)
      .encode(X)
      .encode(X)
      .encode(X)
      .encode(last)
      .get_key_view();
}

/// NW1 below the root: the I4 that Root-I4 holds at byte 0x10 holds key_a's
/// chain and a two-byte sibling leaf.  T2's removal of the sibling fails to
/// lock T1's held chain top, as in NW1, and restarts.  T1 cannot collapse the
/// I4 into the keyless sibling leaf, so it does not lock Root-I4 and does not
/// restart on T2's release of it.
UNODB_TEST(OLCI4CollapseVsHeldSurvivor, SiblingRemoveBelowRootRestarts) {
  using db_type = unodb::olc_db<unodb::key_view, unodb::value_view>;
  // key_a's value, then sib's and root_sib's.
  constexpr auto vals = distinct_values<3>();

  db_type db;
  unodb::key_encoder enc;

  std::array<std::byte, 27> buf_a{};
  std::array<std::byte, 27> buf_b{};
  const auto key_a = copy_key(make_key_27(enc, 0x01, 0x01), buf_a);
  const auto key_b = copy_key(make_key_27(enc, 0x01, 0x02), buf_b);
  std::array<std::byte, 2> buf_sib{};
  const auto sib = copy_key(enc.reset()
                                .encode(std::uint8_t{0x10})
                                .encode(std::uint8_t{0x20})
                                .get_key_view(),
                            buf_sib);
  std::array<std::byte, 1> buf_root_sib{};
  const auto root_sib = copy_key(make_short_key(enc, 0x30), buf_root_sib);
  UNODB_ASSERT_NO_FATAL_FAILURE(
      insert_chain_beside(db, key_a, key_b, std::array{sib, root_sib}, vals[0],
                          std::span{vals}.subspan<1, 2>()));

  bool key_a_removed = false;
  bool sib_removed = false;
  const auto [chain_hits, remove_hits] = run_parked_until_released<2>(
      unodb::detail::sync_after_chain_locked,
      unodb::detail::sync_before_remove_write_guard,
      [&] { key_a_removed = db.remove(key_a); },
      [&] { sib_removed = db.remove(sib); });
  // T1's park, and T2's last retry, which cuts the one-child chain.
  UNODB_ASSERT_EQ(2U, chain_hits);
  // T2's first attempt, and the retry after it failed to lock the chain top.
  UNODB_ASSERT_EQ(2U, remove_hits);
  UNODB_ASSERT_TRUE(key_a_removed);
  UNODB_ASSERT_TRUE(sib_removed);
  const unodb::quiescent_state_on_scope_exit q{};
  UNODB_ASSERT_FALSE(db.get(key_a).has_value());
  UNODB_ASSERT_FALSE(db.get(sib).has_value());
  ASSERT_VALUE_FOR_KEY(db_type, db, root_sib, vals[2]);
}

/// NW2 below the root: the I4 that Root-I4 holds at byte 0x10 holds two
/// chains.  T2's chain cut of sib fails to lock T1's held chain top, as in
/// NW2, and restarts without having locked Root-I4, so T1 does not restart on
/// Root-I4 when it resumes.
UNODB_TEST(OLCI4CollapseVsHeldSurvivor, ChainCutBelowRootRestarts) {
  using db_type = unodb::olc_db<unodb::key_view, unodb::value_view>;
  // key_a's value, then sib's and root_sib's.
  constexpr auto vals = distinct_values<3>();

  db_type db;
  unodb::key_encoder enc;

  std::array<std::byte, 27> buf_a{};
  std::array<std::byte, 27> buf_b{};
  std::array<std::byte, 27> buf_sib{};
  const auto key_a = copy_key(make_key_27(enc, 0x01, 0x01), buf_a);
  const auto key_b = copy_key(make_key_27(enc, 0x01, 0x02), buf_b);
  const auto sib = copy_key(make_key_27(enc, 0x02, 0x01), buf_sib);
  std::array<std::byte, 1> buf_root_sib{};
  const auto root_sib = copy_key(make_short_key(enc, 0x20), buf_root_sib);
  UNODB_ASSERT_NO_FATAL_FAILURE(
      insert_chain_beside(db, key_a, key_b, std::array{sib, root_sib}, vals[0],
                          std::span{vals}.subspan<1, 2>()));

  bool key_a_removed = false;
  bool sib_removed = false;
  const auto [chain_hits, between_hits] = run_parked_until_released<2>(
      unodb::detail::sync_after_chain_locked,
      unodb::detail::sync_between_chain_locks,
      [&] { key_a_removed = db.remove(key_a); },
      [&] { sib_removed = db.remove(sib); });
  // As in NW2: T1, T2's first attempt, T2's retry (which releases T1), and
  // T2's next retry after T1 collapsed the I4 into T2's chain top.
  UNODB_ASSERT_EQ(4U, chain_hits);
  UNODB_ASSERT_EQ(4U, between_hits);
  UNODB_ASSERT_TRUE(key_a_removed);
  UNODB_ASSERT_TRUE(sib_removed);
  const unodb::quiescent_state_on_scope_exit q{};
  UNODB_ASSERT_FALSE(db.get(key_a).has_value());
  UNODB_ASSERT_FALSE(db.get(sib).has_value());
  ASSERT_VALUE_FOR_KEY(db_type, db, root_sib, vals[2]);
}

/// After a chain cut whose I4 cut-point parent collapses has removed its key,
/// the root I4 and sib's four-node chain are all that is left.
// Not noexcept: under --gtest_throw_on_failure a failed check throws.
UNODB_DETAIL_DISABLE_MSVC_WARNING(26440)
void assert_collapse_retry_left_five_i4s(
    [[maybe_unused]] const unodb::test::key_view_olc_db& db) {
#ifdef UNODB_DETAIL_WITH_STATS
  // A retry that removed the entry without collapsing would leave six.
  UNODB_ASSERT_EQ(5U, db.get_node_counts()[unodb::as_i<unodb::node_type::I4>]);
#endif
}
UNODB_DETAIL_RESTORE_MSVC_WARNINGS()

/// CL5: CL4 with an I4 cut-point parent that collapses.  The root is an I4
/// with prefix 0x10 that holds key_a's chain and sib's at byte 1, each chain
/// top with an empty prefix, so cutting key_a's chain collapses the root into
/// sib's chain top.  T1 removes key_a and parks after Step 2.  T2 inserts a
/// one-byte key that splits the root's prefix and installs a new root I4
/// above it.  T1's Step 4.1 locks the survivor and must then see that the root
/// pointer no longer points to the old root, release the survivor and retry
/// below the new root, instead of replacing the new root with sib's chain top.
UNODB_TEST(OLCChainCutVsSplit, RootSplitAboveCollapsingCutPointParent) {
  unodb::key_encoder enc;
  std::array<std::byte, 27> buf_a{};
  std::array<std::byte, 27> buf_sib{};
  std::array<std::byte, 1> buf_t2{};
  const auto key_a = copy_key(make_key_27(enc, 0x01, 0x01), buf_a);
  const std::array sibs{copy_key(make_key_27(enc, 0x02, 0x01), buf_sib)};
  const auto t2_key = copy_key(make_short_key(enc, 0x20), buf_t2);
  // T1's park and its retry: T2's insert never reaches the point.  Step 4.1
  // locks the survivor, finds that the root pointer no longer points to the
  // collapsing I4, and restarts.
  assert_remove_keeps_other_keys<unodb::test::key_view_olc_db>(
      unodb::detail::sync_after_chain_locked, 2, key_a, sibs, t2_key,
      t2_action::insert, assert_collapse_retry_left_five_i4s);
}

/// CL6: CL5 one level down: Root-I4 holds the collapsing I4 at byte 0x10 and a
/// one-byte sibling 0x20.  T2 inserts the one-byte key 0x05 into Root-I4,
/// which moves Root-I4's version and shifts the collapsing I4's entry to the
/// next slot.  T1's Step 4.1 locks the survivor and must then see that Root-I4
/// moved, release the survivor and retry, instead of writing sib's chain top
/// into the slot that now holds T2's key.  T2's key must sort before 0x10, or
/// the stale slot would still hold the collapsing I4's entry.
UNODB_TEST(OLCChainCutVsSplit, GrandparentInsertAboveCollapsingCutPointParent) {
  unodb::key_encoder enc;
  std::array<std::byte, 27> buf_a{};
  std::array<std::byte, 27> buf_sib{};
  std::array<std::byte, 1> buf_root_sib{};
  std::array<std::byte, 1> buf_t2{};
  const auto key_a = copy_key(make_key_27(enc, 0x01, 0x01), buf_a);
  const std::array sibs{copy_key(make_key_27(enc, 0x02, 0x01), buf_sib),
                        copy_key(make_short_key(enc, 0x20), buf_root_sib)};
  const auto t2_key = copy_key(make_short_key(enc, 0x05), buf_t2);
  // T1's park and its retry, as in CL5.  Step 4.1 locks the survivor, finds
  // that Root-I4 moved, and restarts.
  assert_remove_keeps_other_keys<unodb::test::key_view_olc_db>(
      unodb::detail::sync_after_chain_locked, 2, key_a, sibs, t2_key,
      t2_action::insert, assert_collapse_retry_left_five_i4s);
}

#endif  // NDEBUG

// ===================================================================
// Concurrent chain cut stress tests (no sync points).
// ===================================================================

// CT5: Two threads removing last two keys under same chain.
UNODB_TEST(OLCChainCutStress, TwoRemoversFromChain) {
  using db_type = unodb::olc_db<unodb::key_view, unodb::value_view>;
  // key_a, key_b and sibling, in that order.
  constexpr auto vals = distinct_values<3>();

  for (int iter = 0; iter < 200; ++iter) {
    db_type db;
    unodb::key_encoder enc;
    std::array<std::byte, 9> buf_a{};
    std::array<std::byte, 9> buf_b{};
    std::array<std::byte, 9> buf_sib{};

    const auto key_a = make_chain_key(enc, 0x10, 1, buf_a);
    const auto key_b = make_chain_key(enc, 0x10, 2, buf_b);
    const auto sibling = make_chain_key(enc, 0x20, 1, buf_sib);

    UNODB_ASSERT_TRUE(db.insert(key_a, vals[0]));
    UNODB_ASSERT_TRUE(db.insert(key_b, vals[1]));
    UNODB_ASSERT_TRUE(db.insert(sibling, vals[2]));

    unodb::this_thread().qsbr_pause();

    auto t1 = unodb::qsbr_thread([&] {
      const unodb::quiescent_state_on_scope_exit q{};
      std::ignore = db.remove(key_a);
    });
    auto t2 = unodb::qsbr_thread([&] {
      const unodb::quiescent_state_on_scope_exit q{};
      std::ignore = db.remove(key_b);
    });
    t1.join();
    t2.join();
    unodb::this_thread().qsbr_resume();
    unodb::this_thread().quiescent();

    const unodb::quiescent_state_on_scope_exit q{};
    UNODB_ASSERT_FALSE(db.get(key_a).has_value());
    UNODB_ASSERT_FALSE(db.get(key_b).has_value());
    ASSERT_VALUE_FOR_KEY(db_type, db, sibling, vals[2]);
  }
}

// CT6: Multiple threads doing insert/remove on chain-triggering keys.
UNODB_TEST(OLCChainCutStress, InsertRemoveMix) {
  using db_type = unodb::olc_db<unodb::key_view, unodb::value_view>;
  // Key i of tag 0x10 at index i - 1, of tag 0x20 at i + 3 and of tag 0x30 at
  // i + 7.
  constexpr auto vals = distinct_values<12>();

  for (int iter = 0; iter < 100; ++iter) {
    db_type db;

    // Pre-insert some keys so the tree has chain structure.
    {
      unodb::key_encoder enc;
      UNODB_DETAIL_DISABLE_MSVC_WARNING(26496)
      std::array<std::byte, 9> buf{};
      UNODB_DETAIL_RESTORE_MSVC_WARNINGS()
      for (std::uint64_t i = 1; i <= 4; ++i) {
        UNODB_ASSERT_TRUE(
            db.insert(make_chain_key(enc, 0x10, i, buf), vals[i - 1]));
        UNODB_ASSERT_TRUE(
            db.insert(make_chain_key(enc, 0x20, i, buf), vals[i + 3]));
      }
    }

    unodb::this_thread().qsbr_pause();

    // T1: remove keys from tag=0x10.
    auto t1 = unodb::qsbr_thread([&] {
      unodb::key_encoder enc;
      std::array<std::byte, 9> buf{};
      for (std::uint64_t i = 1; i <= 4; ++i) {
        const unodb::quiescent_state_on_scope_exit q{};
        std::ignore = db.remove(make_chain_key(enc, 0x10, i, buf));
      }
    });
    // T2: remove keys from tag=0x20.
    auto t2 = unodb::qsbr_thread([&] {
      unodb::key_encoder enc;
      std::array<std::byte, 9> buf{};
      for (std::uint64_t i = 1; i <= 4; ++i) {
        const unodb::quiescent_state_on_scope_exit q{};
        std::ignore = db.remove(make_chain_key(enc, 0x20, i, buf));
      }
    });
    // T3: insert new keys concurrently.
    auto t3 = unodb::qsbr_thread([&] {
      unodb::key_encoder enc;
      std::array<std::byte, 9> buf{};
      for (std::uint64_t i = 1; i <= 4; ++i) {
        const unodb::quiescent_state_on_scope_exit q{};
        std::ignore = db.insert(make_chain_key(enc, 0x30, i, buf), vals[i + 7]);
      }
    });

    t1.join();
    t2.join();
    t3.join();
    unodb::this_thread().qsbr_resume();
    unodb::this_thread().quiescent();

    // All tag=0x10 and tag=0x20 keys removed. tag=0x30 keys inserted.
    const unodb::quiescent_state_on_scope_exit q{};
    unodb::key_encoder enc;
    UNODB_DETAIL_DISABLE_MSVC_WARNING(26496)
    std::array<std::byte, 9> buf{};
    UNODB_DETAIL_RESTORE_MSVC_WARNINGS()
    for (std::uint64_t i = 1; i <= 4; ++i) {
      UNODB_ASSERT_FALSE(db.get(make_chain_key(enc, 0x10, i, buf)).has_value());
      UNODB_ASSERT_FALSE(db.get(make_chain_key(enc, 0x20, i, buf)).has_value());
      ASSERT_VALUE_FOR_KEY(db_type, db, make_chain_key(enc, 0x30, i, buf),
                           vals[i + 7]);
    }
  }
}

// ===================================================================
// CT6: Deep chain stress test — 8 chain levels, N threads.
// ===================================================================

void deep_chain_stress(int n_threads, int ops_per_thread, int iterations) {
  using db_type = unodb::olc_db<unodb::key_view, unodb::value_view>;
  constexpr auto val_bytes = std::array<std::byte, 1>{std::byte{0x42}};
  const auto val = unodb::value_view{val_bytes};

  constexpr int chain_depth = 8;
  constexpr int chain_variants = 32;
  constexpr int short_count = 8;
  constexpr std::size_t chain_key_len = 1 + (chain_depth * 8) + 1;
  constexpr std::size_t short_key_len = 1;

  auto make_deep_key = [](unodb::key_encoder& enc, std::uint8_t variant,
                          std::array<std::byte, chain_key_len>& buf
                          UNODB_DETAIL_LIFETIMEBOUND) {
    enc.reset().encode(std::uint8_t{0x42});
    for (int d = 0; d < chain_depth; ++d) enc.encode(std::uint64_t{0});
    enc.encode(variant);
    return copy_key(enc.get_key_view(), buf);
  };

  auto make_short = [](unodb::key_encoder& enc, std::uint8_t tag,
                       std::array<std::byte, short_key_len>& buf
                       UNODB_DETAIL_LIFETIMEBOUND) {
    return copy_key(make_short_key(enc, tag), buf);
  };

  for (int iter = 0; iter < iterations; ++iter) {
    db_type db;
    unodb::key_encoder enc;

    UNODB_DETAIL_DISABLE_MSVC_WARNING(26496)
    std::array<std::array<std::byte, chain_key_len>, 4> ck_bufs{};
    for (std::size_t v = 1; v <= 4; ++v)
      UNODB_ASSERT_TRUE(db.insert(
          make_deep_key(enc, static_cast<std::uint8_t>(v), ck_bufs[v - 1]),
          val));
    std::array<std::array<std::byte, short_key_len>, short_count> sk_bufs{};
    for (std::size_t t = 1; t <= short_count; ++t)
      UNODB_ASSERT_TRUE(db.insert(
          make_short(enc, static_cast<std::uint8_t>(t), sk_bufs[t - 1]), val));
    UNODB_DETAIL_RESTORE_MSVC_WARNINGS()

    unodb::this_thread().qsbr_pause();

    std::vector<unodb::qsbr_thread> threads;
    threads.reserve(static_cast<std::size_t>(n_threads));
    for (int ti = 0; ti < n_threads; ++ti) {
      threads.emplace_back([&, ti] {
        unodb::key_encoder e;
        std::minstd_rand rng{static_cast<unsigned>((ti * 1000) + iter)};
        for (int op = 0; op < ops_per_thread; ++op) {
          const unodb::quiescent_state_on_scope_exit q{};
          const auto r = rng() % 6;
          if (r < 2) {
            const auto v =
                static_cast<std::uint8_t>(1 + (rng() % chain_variants));
            std::array<std::byte, chain_key_len> buf{};
            std::ignore = db.insert(make_deep_key(e, v, buf), val);
          } else if (r < 4) {
            const auto v =
                static_cast<std::uint8_t>(1 + (rng() % chain_variants));
            std::array<std::byte, chain_key_len> buf{};
            std::ignore = db.remove(make_deep_key(e, v, buf));
          } else if (r == 4) {
            if (rng() % 2 == 0) {
              const auto v =
                  static_cast<std::uint8_t>(1 + (rng() % chain_variants));
              std::array<std::byte, chain_key_len> buf{};
              std::ignore = db.get(make_deep_key(e, v, buf));
            } else {
              const auto t =
                  static_cast<std::uint8_t>(1 + (rng() % short_count));
              std::array<std::byte, short_key_len> buf{};
              std::ignore = db.get(make_short(e, t, buf));
            }
          } else {
            const auto t = static_cast<std::uint8_t>(1 + (rng() % short_count));
            std::array<std::byte, short_key_len> buf{};
            if (rng() % 2 == 0)
              std::ignore = db.insert(make_short(e, t, buf), val);
            else
              std::ignore = db.remove(make_short(e, t, buf));
          }
        }
      });
    }
    for (auto& t : threads) t.join();

    unodb::this_thread().qsbr_resume();
    unodb::this_thread().quiescent();

    const unodb::quiescent_state_on_scope_exit q{};
    for (std::uint8_t v = 1; v <= chain_variants; ++v) {
      std::array<std::byte, chain_key_len> buf{};
      std::ignore = db.remove(make_deep_key(enc, v, buf));
    }
    for (std::uint8_t t = 1; t <= short_count; ++t) {
      std::array<std::byte, short_key_len> buf{};
      std::ignore = db.remove(make_short(enc, t, buf));
    }
    UNODB_ASSERT_TRUE(db.empty());
  }
}

// Default: 4 threads, 1000 ops, 10 iterations.
UNODB_TEST(OLCChainCut, DeepChainStress) { deep_chain_stress(4, 1000, 10); }

// Heavy: all cores, 100k ops, 10 iterations.  DISABLED by default.
UNODB_TEST(OLCChainCut, DISABLED_DeepChainStressHeavy) {
  const int cores = static_cast<int>(std::thread::hardware_concurrency());
  deep_chain_stress(cores, 100000, 10);
}

// ===================================================================
// A key that ends inside an inode's key prefix is a strict prefix of every
// key below that inode, so it is not stored.  A descent must report it absent
// without shifting the key past its end.
// ===================================================================

/// Check that \a misses(db, key) holds for keys that end inside or exactly at
/// the end of insert_short_key_pair()'s I4 prefix, and for one that Root-I4's
/// dispatch byte uses up above insert_short_collapse_tree()'s empty-prefix I4,
/// and that every key keeps its own value.  Call it through
/// UNODB_ASSERT_NO_FATAL_FAILURE.
template <class Misses>
void assert_strict_prefix_keys_miss(const Misses& misses) {
  const auto& vals = short_collapse_vals;
  short_collapse_db db;
  UNODB_ASSERT_NO_FATAL_FAILURE(insert_short_key_pair(db, vals));
  short_collapse_db collapse_db;
  UNODB_ASSERT_NO_FATAL_FAILURE(insert_short_collapse_tree(collapse_db, vals));

  const unodb::quiescent_state_on_scope_exit q{};
  // Ends inside the I4's two-byte prefix, and exactly at its end.
  UNODB_ASSERT_TRUE(misses(db, short_key_a.first(1)));
  UNODB_ASSERT_TRUE(misses(db, short_key_a.first(2)));
  ASSERT_VALUE_FOR_KEY(short_collapse_db, db, short_key_a, vals[0]);
  ASSERT_VALUE_FOR_KEY(short_collapse_db, db, short_key_b, vals[1]);
  // Root-I4's dispatch byte 00 uses up 07 00, above the empty-prefix I4.  The
  // byte after it in short_key_a's buffer, 07, is mapped there, so a descent
  // that read it would reach short_key_a.
  UNODB_ASSERT_TRUE(misses(collapse_db, short_key_a.first(2)));
  ASSERT_VALUE_FOR_KEY(short_collapse_db, collapse_db, short_key_a, vals[0]);
  ASSERT_VALUE_FOR_KEY(short_collapse_db, collapse_db, short_key_b, vals[1]);
  ASSERT_VALUE_FOR_KEY(short_collapse_db, collapse_db, short_sib, vals[2]);
}

/// assert_strict_prefix_keys_miss() with get().
UNODB_TEST(OLCKeyEndsInPrefix, GetMisses) {
  UNODB_ASSERT_NO_FATAL_FAILURE(assert_strict_prefix_keys_miss(
      [](auto& db, unodb::key_view key) { return !db.get(key).has_value(); }));
}

/// assert_strict_prefix_keys_miss() with remove().
UNODB_TEST(OLCKeyEndsInPrefix, RemoveMisses) {
  UNODB_ASSERT_NO_FATAL_FAILURE(assert_strict_prefix_keys_miss(
      [](auto& db, unodb::key_view key) { return !db.remove(key); }));
}

#ifndef NDEBUG

/// In insert_short_key_pair()'s tree, run remove(\a key) parked at its first
/// sync_before_prefix_load hit, at the I4, while T2 inserts 07 00 09.  T2
/// write-locks the I4 but only read-locks the root pointer, so the remove's
/// unlock of the root pointer succeeds and that of the I4 fails.  Check that
/// the remove misses after one restart, that T2 inserted, and that every key
/// keeps its own value.  Call it through UNODB_ASSERT_NO_FATAL_FAILURE.
void assert_remove_restarts_on_node_write(unodb::key_view key) {
  // short_key_a's, short_key_b's and t2_key's values.
  constexpr auto vals = distinct_values<3>();
  constexpr std::array t2_key_bytes{std::byte{0x07}, std::byte{0x00},
                                    std::byte{0x09}};
  const unodb::key_view t2_key{t2_key_bytes};

  short_collapse_db db;
  UNODB_ASSERT_NO_FATAL_FAILURE(insert_short_key_pair(db, vals));

  bool removed = true;
  bool inserted = false;
  const auto hits = run_parked_interleaving(
      unodb::detail::sync_before_prefix_load, [&] { removed = db.remove(key); },
      [&] { inserted = db.insert(t2_key, vals[2]); });
  // T1's park at the I4, T2's insert passing the I4, and T1's retry after its
  // unlock of the I4 fails.
  UNODB_ASSERT_EQ(3U, hits);
  UNODB_ASSERT_FALSE(removed);
  UNODB_ASSERT_TRUE(inserted);

  ASSERT_VALUE_FOR_KEY(short_collapse_db, db, short_key_a, vals[0]);
  ASSERT_VALUE_FOR_KEY(short_collapse_db, db, short_key_b, vals[1]);
  ASSERT_VALUE_FOR_KEY(short_collapse_db, db, t2_key, vals[2]);
}

/// assert_remove_restarts_on_node_write() with a key that ends inside the
/// I4's prefix.
UNODB_TEST(OLCKeyEndsInPrefix, RemoveRestartsOnNodeWrite) {
  UNODB_ASSERT_NO_FATAL_FAILURE(
      assert_remove_restarts_on_node_write(short_key_a.first(1)));
}

/// assert_remove_restarts_on_node_write() with a key that mismatches the I4's
/// prefix: 07 01 07.
UNODB_TEST(OLCPrefixMismatch, RemoveRestartsOnNodeWrite) {
  constexpr std::array mismatch_bytes{std::byte{0x07}, std::byte{0x01},
                                      std::byte{0x07}};
  UNODB_ASSERT_NO_FATAL_FAILURE(
      assert_remove_restarts_on_node_write(unodb::key_view{mismatch_bytes}));
}

#endif  // NDEBUG

/// scan_from() a key that ends inside an I4's prefix, where the prefix byte
/// past the key is zero, or exactly at the prefix's end.  The key precedes
/// every key below the I4, so a forward scan visits them all and a reverse
/// scan visits none.
UNODB_TEST(OLCKeyEndsInPrefix, ScanFromZeroPaddedKey) {
  // short_key_a's and short_key_b's values.
  constexpr auto vals = distinct_values<2>();

  short_collapse_db db;
  UNODB_ASSERT_NO_FATAL_FAILURE(insert_short_key_pair(db, vals));

  const unodb::quiescent_state_on_scope_exit q{};
  for (const auto len : {1U, 2U}) {
    const auto from = short_key_a.first(len);
    UNODB_ASSERT_EQ(keys_scanned_from(db, from, true),
                    (key_list{key_bytes(short_key_a), key_bytes(short_key_b)}));
    UNODB_ASSERT_TRUE(keys_scanned_from(db, from, false).empty());
  }
}

/// As ScanFromZeroPaddedKey, with a key that ends before a non-zero prefix
/// byte.  The byte past the key in its buffer is above that prefix byte, so a
/// seek that compared it would order the key after every key below the I4.
UNODB_TEST(OLCKeyEndsInPrefix, ScanFromKeyEndingBeforePrefixByte) {
  using db_type = unodb::olc_db<unodb::key_view, unodb::value_view>;
  constexpr auto vals = distinct_values<2>();
  constexpr std::array key_a{std::byte{0x41}, std::byte{0x42}, std::byte{0x00}};
  constexpr std::array key_b{std::byte{0x41}, std::byte{0x42}, std::byte{0x01}};
  constexpr std::array from_buf{std::byte{0x41}, std::byte{0x43}};

  db_type db;
  // One I4 with prefix 41 42 over key_a and key_b.
  UNODB_ASSERT_TRUE(db.insert(unodb::key_view{key_a}, vals[0]));
  UNODB_ASSERT_TRUE(db.insert(unodb::key_view{key_b}, vals[1]));

  const unodb::quiescent_state_on_scope_exit q{};
  const auto from = unodb::key_view{from_buf}.first(1);
  UNODB_ASSERT_EQ(keys_scanned_from(db, from, true),
                  (key_list{key_bytes(unodb::key_view{key_a}),
                            key_bytes(unodb::key_view{key_b})}));
  UNODB_ASSERT_TRUE(keys_scanned_from(db, from, false).empty());
}

/// As ScanFromKeyEndingBeforePrefixByte, with a key that ends at Root-I4's
/// dispatch byte for an I4 with a non-empty prefix.
UNODB_TEST(OLCKeyEndsInPrefix, ScanFromKeyEndingAtDispatchByte) {
  using db_type = unodb::olc_db<unodb::key_view, unodb::value_view>;
  constexpr auto vals = distinct_values<3>();
  constexpr std::array key_a{std::byte{0x07}, std::byte{0x41}, std::byte{0x42},
                             std::byte{0x00}};
  constexpr std::array key_b{std::byte{0x07}, std::byte{0x41}, std::byte{0x42},
                             std::byte{0x01}};
  constexpr std::array key_c{std::byte{0x08}, std::byte{0x00}};
  constexpr std::array from_buf{std::byte{0x07}, std::byte{0x43}};

  db_type db;
  // Root-I4 with an empty prefix holds, at 07, an I4 with prefix 41 42 over
  // key_a and key_b, and at 08 key_c.
  UNODB_ASSERT_TRUE(db.insert(unodb::key_view{key_a}, vals[0]));
  UNODB_ASSERT_TRUE(db.insert(unodb::key_view{key_b}, vals[1]));
  UNODB_ASSERT_TRUE(db.insert(unodb::key_view{key_c}, vals[2]));

  const unodb::quiescent_state_on_scope_exit q{};
  const auto from = unodb::key_view{from_buf}.first(1);
  UNODB_ASSERT_EQ(keys_scanned_from(db, from, true),
                  (key_list{key_bytes(unodb::key_view{key_a}),
                            key_bytes(unodb::key_view{key_b}),
                            key_bytes(unodb::key_view{key_c})}));
  UNODB_ASSERT_TRUE(keys_scanned_from(db, from, false).empty());
}

/// get(), remove() and scan_from() of keys of the three shapes above, each
/// below an I4 that has smaller siblings, so that a reverse scan must climb
/// from the I4's left-most leaf to its predecessor.  get() and remove() must
/// miss and keep every key with its own value.  Root-I4, with an empty prefix,
/// holds 06 00, at 07 an I4 with prefix 00 00 over 07 00 00 07 and 07 00 00 08,
/// and at 08 an I4 with prefix 41 42 over 08 41 42 00 and 08 41 42 01.  Call it
/// through UNODB_ASSERT_NO_FATAL_FAILURE.
template <class Db>
void assert_strict_prefix_misses_and_scans_to_siblings() {
  constexpr std::array key_0{std::byte{0x06}, std::byte{0x00}};
  constexpr std::array key_1{std::byte{0x07}, std::byte{0x00}, std::byte{0x00},
                             std::byte{0x07}};
  constexpr std::array key_2{std::byte{0x07}, std::byte{0x00}, std::byte{0x00},
                             std::byte{0x08}};
  constexpr std::array key_3{std::byte{0x08}, std::byte{0x41}, std::byte{0x42},
                             std::byte{0x00}};
  constexpr std::array key_4{std::byte{0x08}, std::byte{0x41}, std::byte{0x42},
                             std::byte{0x01}};
  // A seek that read the byte past 08 or 08 41 in this buffer would order the
  // key after the I4 at 08.
  constexpr std::array past_42_buf{std::byte{0x08}, std::byte{0x41},
                                   std::byte{0x43}};

  const std::array keys{unodb::key_view{key_0}, unodb::key_view{key_1},
                        unodb::key_view{key_2}, unodb::key_view{key_3},
                        unodb::key_view{key_4}};
  // One value per key, so that a get() that reaches another key's keyless
  // leaf or packed value finds the wrong value.
  const auto val = [](std::size_t i) {
    return unodb::test::get_test_value<Db>(i);
  };

  Db db;
  for (std::size_t i = 0; i < keys.size(); ++i)
    UNODB_ASSERT_TRUE(db.insert(keys[i], val(i)));
  const auto bytes_0 = key_bytes(keys[0]);
  const auto bytes_1 = key_bytes(keys[1]);
  const auto bytes_2 = key_bytes(keys[2]);
  const auto bytes_3 = key_bytes(keys[3]);
  const auto bytes_4 = key_bytes(keys[4]);

  const unodb::quiescent_state_on_scope_exit q{};
  // Below the zero prefix: 07, used up by Root-I4's dispatch byte, 07 00,
  // which ends inside the prefix, and 07 00 00, which ends at its end.
  for (const auto len : {1U, 2U, 3U}) {
    const auto from = unodb::key_view{key_1}.first(len);
    UNODB_ASSERT_FALSE(db.get(from).has_value());
    UNODB_ASSERT_FALSE(db.remove(from));
    UNODB_ASSERT_EQ(keys_scanned_from(db, from, true),
                    (key_list{bytes_1, bytes_2, bytes_3, bytes_4}));
    UNODB_ASSERT_EQ(keys_scanned_from(db, from, false), (key_list{bytes_0}));
  }
  // Below the non-zero prefix: 08, used up by the dispatch byte, 08 41, which
  // ends before the prefix byte 42, and 08 41 42, which ends at its end.
  for (const auto from : {unodb::key_view{past_42_buf}.first(1),
                          unodb::key_view{past_42_buf}.first(2),
                          unodb::key_view{key_4}.first(3)}) {
    UNODB_ASSERT_FALSE(db.get(from).has_value());
    UNODB_ASSERT_FALSE(db.remove(from));
    UNODB_ASSERT_EQ(keys_scanned_from(db, from, true),
                    (key_list{bytes_3, bytes_4}));
    UNODB_ASSERT_EQ(keys_scanned_from(db, from, false),
                    (key_list{bytes_2, bytes_1, bytes_0}));
  }
  for (std::size_t i = 0; i < keys.size(); ++i)
    ASSERT_VALUE_FOR_KEY(Db, db, keys[i], val(i));
}

/// assert_strict_prefix_misses_and_scans_to_siblings() on a keyless-leaf Db.
UNODB_TEST(OLCKeyEndsInPrefix, StrictPrefixMissesAndScansToSiblings) {
  UNODB_ASSERT_NO_FATAL_FAILURE(
      assert_strict_prefix_misses_and_scans_to_siblings<
          unodb::test::key_view_olc_db>());
}

/// assert_strict_prefix_misses_and_scans_to_siblings() on a value-in-slot Db.
UNODB_TEST(OLCKeyEndsInPrefix,
           StrictPrefixMissesAndScansToSiblingsValueInSlot) {
  UNODB_ASSERT_NO_FATAL_FAILURE(
      assert_strict_prefix_misses_and_scans_to_siblings<
          unodb::test::key_view_u64val_olc_db>());
}

#ifndef NDEBUG

/// In a \a Db holding 07 42 07 with \a val_a and 07 42 08 with \a val_b,
/// insert and upsert two strict prefixes of both keys and expect each write
/// to fail the prefix-free key assert: 07 42, which ends exactly at the end of
/// their I4's prefix 07 42, and 07, which ends before its non-zero byte 42.
/// The byte past each key in its buffer, 09 or 43, is not the tree's, so a
/// write that did not assert would store the key under it.
template <class Db>
void assert_write_of_strict_prefix_key_dies(typename Db::value_type val_a,
                                            typename Db::value_type val_b) {
  constexpr std::array key_a{std::byte{0x07}, std::byte{0x42}, std::byte{0x07}};
  constexpr std::array key_b{std::byte{0x07}, std::byte{0x42}, std::byte{0x08}};
  constexpr std::array at_end_buf{std::byte{0x07}, std::byte{0x42},
                                  std::byte{0x09}};
  constexpr std::array before_byte_buf{std::byte{0x07}, std::byte{0x43}};

  Db db;
  UNODB_ASSERT_TRUE(db.insert(unodb::key_view{key_a}, val_a));
  UNODB_ASSERT_TRUE(db.insert(unodb::key_view{key_b}, val_b));

  const unodb::quiescent_state_on_scope_exit q{};
  for (const auto key : {unodb::key_view{at_end_buf}.first(2),
                         unodb::key_view{before_byte_buf}.first(1)}) {
    UNODB_ASSERT_DEATH(
        { std::ignore = db.insert(key, val_a); },
        "!key_matches_prefix_to_end\\(remaining_key, shared_length\\)");
    UNODB_ASSERT_DEATH(
        {
          std::ignore = db.upsert(
              key, val_a, [](auto&) { return unodb::upsert_action::keep; });
        },
        "!key_matches_prefix_to_end\\(remaining_key, shared_length\\)");
  }
}

/// assert_write_of_strict_prefix_key_dies() on a keyless-leaf Db.
UNODB_TEST(OLCKeyEndsInPrefixDeathTest, WriteOfStrictPrefixKeyDies) {
  constexpr auto vals = distinct_values<2>();
  assert_write_of_strict_prefix_key_dies<unodb::test::key_view_olc_db>(vals[0],
                                                                       vals[1]);
}

/// assert_write_of_strict_prefix_key_dies() on a value-in-slot Db.
UNODB_TEST(OLCKeyEndsInPrefixDeathTest, WriteOfStrictPrefixKeyDiesValueInSlot) {
  assert_write_of_strict_prefix_key_dies<unodb::test::key_view_u64val_olc_db>(
      1, 2);
}

#endif  // NDEBUG

// ===================================================================
// RT tests: verify remove_or_choose_subtree OLC guards under
// concurrent modification.  These validate that the version checks
// inside remove_or_choose_subtree correctly detect concurrent
// mutations — the precondition for making matches() unconditional
// for keyless key_view leaves.
//
// Sync point: sync_before_remove_write_guard fires after leaf match
// confirmed and is_min_size read, before write guard acquisition.
// ===================================================================

#ifndef NDEBUG

// RT1: T1 removes key_a.  T2 inserts a new key into the same inode
// between T1's match check and write guard acquisition.  T1's write
// guard upgrade should detect the version change and restart.
UNODB_TEST(OLCRemoveChooseSubtree, ConcurrentInsertIntoSameInode) {
  using db_type = unodb::olc_db<unodb::key_view, unodb::value_view>;
  // key_a, key_b, key_c and new_key, in that order.
  constexpr auto vals = distinct_values<4>();

  db_type db;
  unodb::key_encoder enc;

  // 9-byte keys with same tag → same chain → same bottom inode.
  // Insert 3 keys so the bottom inode is I4(3), above min_size.
  std::array<std::byte, 9> buf_a{};
  std::array<std::byte, 9> buf_b{};
  std::array<std::byte, 9> buf_c{};
  std::array<std::byte, 9> buf_new{};
  const auto key_a = make_chain_key(enc, 0x10, 1, buf_a);
  const auto key_b = make_chain_key(enc, 0x10, 2, buf_b);
  const auto key_c = make_chain_key(enc, 0x10, 3, buf_c);
  const auto new_key = make_chain_key(enc, 0x10, 4, buf_new);

  UNODB_ASSERT_TRUE(db.insert(key_a, vals[0]));
  UNODB_ASSERT_TRUE(db.insert(key_b, vals[1]));
  UNODB_ASSERT_TRUE(db.insert(key_c, vals[2]));

  const auto hits = run_parked_interleaving(
      unodb::detail::sync_before_remove_write_guard,
      [&] { std::ignore = db.remove(key_a); },
      [&] { std::ignore = db.insert(new_key, vals[3]); });
  // T1's park and its retry: T2's insert never reaches the point but moves
  // the bottom I4's version, failing T1's node write-guard upgrade.
  UNODB_ASSERT_EQ(2U, hits);

  const unodb::quiescent_state_on_scope_exit q{};
  UNODB_ASSERT_FALSE(db.get(key_a).has_value());
  ASSERT_VALUE_FOR_KEY(db_type, db, key_b, vals[1]);
  ASSERT_VALUE_FOR_KEY(db_type, db, key_c, vals[2]);
  ASSERT_VALUE_FOR_KEY(db_type, db, new_key, vals[3]);
}

// RT2: T1 removes key_a from an I4 at min_size (2 children).
// T2 removes key_b (the other child) concurrently, so T1 restarts
// and cuts key_a's chain instead.
UNODB_TEST(OLCRemoveChooseSubtree, ConcurrentRemoveFromMinSizeI4) {
  using db_type = unodb::olc_db<unodb::key_view, unodb::value_view>;
  // key_a, key_b and sibling, in that order.
  constexpr auto vals = distinct_values<3>();

  db_type db;
  unodb::key_encoder enc;

  // 9-byte keys: 2 keys with same tag → chain + I4(2) at bottom.
  // Plus a sibling with different tag so the chain has a parent.
  std::array<std::byte, 9> buf_a{};
  std::array<std::byte, 9> buf_b{};
  std::array<std::byte, 9> buf_sib{};
  const auto key_a = make_chain_key(enc, 0x10, 1, buf_a);
  const auto key_b = make_chain_key(enc, 0x10, 2, buf_b);
  const auto sibling = make_chain_key(enc, 0x20, 1, buf_sib);

  UNODB_ASSERT_TRUE(db.insert(key_a, vals[0]));
  UNODB_ASSERT_TRUE(db.insert(key_b, vals[1]));
  UNODB_ASSERT_TRUE(db.insert(sibling, vals[2]));

  unsigned chain_locked_hits = 0;
  // Declared after the counter, so that it disarms the hook first.
  const sync_point_guard chain_locked_guard{
      unodb::detail::sync_after_chain_locked};
  unodb::detail::sync_after_chain_locked.arm(
      [&chain_locked_hits]() noexcept { ++chain_locked_hits; });
  bool t1_removed = false;
  bool t2_removed = false;
  const auto hits = run_parked_interleaving(
      unodb::detail::sync_before_remove_write_guard,
      [&] { t1_removed = db.remove(key_a); },
      [&] { t2_removed = db.remove(key_b); });
  // T1's park and T2's remove of key_b; T1's parent write-guard upgrade fails,
  // and its retry finds the bottom I4 single-child and chain-cuts it instead.
  UNODB_ASSERT_EQ(2U, hits);
  // Both hits above come before T1's retry; this one is the retry's cut.
  UNODB_ASSERT_EQ(1U, chain_locked_hits);
  UNODB_ASSERT_TRUE(t1_removed);
  UNODB_ASSERT_TRUE(t2_removed);

  const unodb::quiescent_state_on_scope_exit q{};
  UNODB_ASSERT_FALSE(db.get(key_a).has_value());
  UNODB_ASSERT_FALSE(db.get(key_b).has_value());
  ASSERT_VALUE_FOR_KEY(db_type, db, sibling, vals[2]);
}

// RT3: T1 removes key_a.  T2 replaces key_a's leaf by removing and
// re-inserting it with a different value.  T2's remove and re-insert both
// write-lock the I4 above the leaf, so T1's node write-guard upgrade fails
// before its child write guard is tried, and T1 restarts.
UNODB_TEST(OLCRemoveChooseSubtree, ConcurrentLeafReplacement) {
  using db_type = unodb::olc_db<unodb::key_view, unodb::value_view>;
  // key_a's value before and after T2 replaces it, then key_b's and key_c's.
  constexpr auto vals = distinct_values<4>();

  db_type db;
  unodb::key_encoder enc;

  std::array<std::byte, 9> buf_a{};
  std::array<std::byte, 9> buf_b{};
  std::array<std::byte, 9> buf_c{};
  const auto key_a = make_chain_key(enc, 0x10, 1, buf_a);
  const auto key_b = make_chain_key(enc, 0x10, 2, buf_b);
  const auto key_c = make_chain_key(enc, 0x10, 3, buf_c);

  UNODB_ASSERT_TRUE(db.insert(key_a, vals[0]));
  UNODB_ASSERT_TRUE(db.insert(key_b, vals[2]));
  UNODB_ASSERT_TRUE(db.insert(key_c, vals[3]));

  bool t1_removed = false;
  bool t2_removed = false;
  bool t2_reinserted = false;
  const auto hits = run_parked_interleaving(
      unodb::detail::sync_before_remove_write_guard,
      [&] { t1_removed = db.remove(key_a); },
      [&] {
        {
          const unodb::quiescent_state_on_scope_exit q{};
          t2_removed = db.remove(key_a);
        }
        t2_reinserted = db.insert(key_a, vals[1]);
      });
  // T1's park, T2's remove of key_a, and T1's retry after its node
  // write-guard upgrade fails; T2's reinsert never reaches the point.
  UNODB_ASSERT_EQ(3U, hits);

  UNODB_ASSERT_TRUE(t2_removed);
  UNODB_ASSERT_TRUE(t2_reinserted);
  // T1 restarts, and its retry removes T2's re-inserted key_a.
  UNODB_ASSERT_TRUE(t1_removed);
  const unodb::quiescent_state_on_scope_exit q2{};
  ASSERT_VALUE_FOR_KEY(db_type, db, key_b, vals[2]);
  ASSERT_VALUE_FOR_KEY(db_type, db, key_c, vals[3]);
  UNODB_ASSERT_FALSE(db.get(key_a).has_value());
}

/// Values 1, 2, ..., \a N, one per key of a test.  Distinct, so that a lookup
/// that reaches another key's slot finds the wrong value, and nonzero, so that
/// a null slot word, which unpacks to 0, fails the check.
template <std::size_t N>
[[nodiscard]] constexpr std::array<std::uint64_t, N>
distinct_u64_values() noexcept {
  std::array<std::uint64_t, N> vals{};
  for (std::size_t i = 0; i < N; ++i) vals[i] = i + 1;
  return vals;
}

// ===================================================================
// Insert growth contention: sync_before_insert_grow_guard fires in
// add_or_choose_subtree after building the larger node but before
// acquiring write guards.  T2 modifies the same inode, causing T1's
// write guard to detect a version mismatch and restart.
// Covers: olc_art.hpp must_restart + delete_subtree in growth path.
// ===================================================================

// IGT1: T1 inserts the 5th key (triggers I4→I16 growth).  T2 removes
// a sibling key between T1's larger-node creation and write guard.
UNODB_TEST(OLCInsertGrowth, ConcurrentRemoveDuringGrowth) {
  using db_type = unodb::olc_db<unodb::key_view, std::uint64_t>;
  // key_a to key_d, then new_key.
  constexpr auto vals = distinct_u64_values<5>();

  db_type db;
  unodb::key_encoder enc;

  // 4 keys with DIFFERENT first bytes → root I4 with 4 children.
  // 5th key triggers I4→I16 growth at the root.
  std::array<std::byte, 9> buf_a{};
  std::array<std::byte, 9> buf_b{};
  std::array<std::byte, 9> buf_c{};
  std::array<std::byte, 9> buf_d{};
  std::array<std::byte, 9> buf_new{};
  const auto key_a = make_chain_key(enc, 0x10, 1, buf_a);
  const auto key_b = make_chain_key(enc, 0x20, 1, buf_b);
  const auto key_c = make_chain_key(enc, 0x30, 1, buf_c);
  const auto key_d = make_chain_key(enc, 0x40, 1, buf_d);
  const auto new_key = make_chain_key(enc, 0x50, 1, buf_new);

  UNODB_ASSERT_TRUE(db.insert(key_a, vals[0]));
  UNODB_ASSERT_TRUE(db.insert(key_b, vals[1]));
  UNODB_ASSERT_TRUE(db.insert(key_c, vals[2]));
  UNODB_ASSERT_TRUE(db.insert(key_d, vals[3]));

  const auto hits = run_parked_interleaving(
      unodb::detail::sync_before_insert_grow_guard,
      // T1: insert new_key, triggering I4→I16 growth at root.
      // Hits sync point, pauses while T2 modifies the node, then restarts.
      [&] { std::ignore = db.insert(new_key, vals[4]); },
      // T2: remove key_d from the root I4, modifying its version.
      [&] { std::ignore = db.remove(key_d); });
  // T1's park only: T2's remove never reaches the point, and after T1's node
  // write-guard upgrade fails its retry finds the root I4 non-full.
  UNODB_ASSERT_EQ(1U, hits);

  const unodb::quiescent_state_on_scope_exit q{};
  UNODB_ASSERT_FALSE(db.get(key_d).has_value());
  ASSERT_VALUE_FOR_KEY(db_type, db, new_key, vals[4]);
  ASSERT_VALUE_FOR_KEY(db_type, db, key_a, vals[0]);
  ASSERT_VALUE_FOR_KEY(db_type, db, key_b, vals[1]);
  ASSERT_VALUE_FOR_KEY(db_type, db, key_c, vals[2]);
}

// IGT2: T1 inserts the 5th key (triggers I4→I16 growth).  T2 inserts
// a different key into the same inode.  T1's write guard detects the
// version change and restarts.
UNODB_TEST(OLCInsertGrowth, ConcurrentInsertDuringGrowth) {
  using db_type = unodb::olc_db<unodb::key_view, std::uint64_t>;
  // key_a to key_d, t1_key, then t2_key.
  constexpr auto vals = distinct_u64_values<6>();

  db_type db;
  unodb::key_encoder enc;

  std::array<std::byte, 9> buf_a{};
  std::array<std::byte, 9> buf_b{};
  std::array<std::byte, 9> buf_c{};
  std::array<std::byte, 9> buf_d{};
  std::array<std::byte, 9> buf_t1{};
  std::array<std::byte, 9> buf_t2{};
  const auto key_a = make_chain_key(enc, 0x10, 1, buf_a);
  const auto key_b = make_chain_key(enc, 0x20, 1, buf_b);
  const auto key_c = make_chain_key(enc, 0x30, 1, buf_c);
  const auto key_d = make_chain_key(enc, 0x40, 1, buf_d);
  const auto t1_key = make_chain_key(enc, 0x50, 1, buf_t1);
  const auto t2_key = make_chain_key(enc, 0x60, 1, buf_t2);

  UNODB_ASSERT_TRUE(db.insert(key_a, vals[0]));
  UNODB_ASSERT_TRUE(db.insert(key_b, vals[1]));
  UNODB_ASSERT_TRUE(db.insert(key_c, vals[2]));
  UNODB_ASSERT_TRUE(db.insert(key_d, vals[3]));

  const auto hits = run_parked_interleaving(
      unodb::detail::sync_before_insert_grow_guard,
      // T1: insert t1_key, triggering growth.  Hits sync point, pauses.
      [&] { std::ignore = db.insert(t1_key, vals[4]); },
      // T2: insert t2_key, also triggering growth and modifying the inode.
      [&] { std::ignore = db.insert(t2_key, vals[5]); });
  // T1's park and T2's own growth; T1's root-pointer write-guard upgrade
  // fails, and its retry adds to the non-full I16 off this point.
  UNODB_ASSERT_EQ(2U, hits);

  const unodb::quiescent_state_on_scope_exit q{};
  ASSERT_VALUE_FOR_KEY(db_type, db, t1_key, vals[4]);
  ASSERT_VALUE_FOR_KEY(db_type, db, t2_key, vals[5]);
  ASSERT_VALUE_FOR_KEY(db_type, db, key_a, vals[0]);
  ASSERT_VALUE_FOR_KEY(db_type, db, key_b, vals[1]);
  ASSERT_VALUE_FOR_KEY(db_type, db, key_c, vals[2]);
  ASSERT_VALUE_FOR_KEY(db_type, db, key_d, vals[3]);
}

// T1 builds a chain for a non-full inode insert, then a concurrent T2
// removes a sibling, invalidating T1's node version.  T1's write guard
// must_restart, so T1 deletes the chain it built and retries.
UNODB_DETAIL_DISABLE_MSVC_WARNING(6326)
UNODB_TEST(OLCNonfullChainRestart, ConcurrentRemoveDuringChainInsert) {
  using db_type = unodb::olc_db<unodb::key_view, std::uint64_t>;
  db_type db;
  unodb::key_encoder enc;
  // k_seed1 and k_seed2 stay live to the end of the test, and each key view
  // aliases its encoder's buffer, so each needs its own encoder that is never
  // re-driven afterwards.
  unodb::key_encoder enc2;
  // k_seed1, k_seed2, then t1_key.
  constexpr auto vals = distinct_u64_values<3>();

  // Seed: two keys with different first bytes → root I4 with 2 children.
  const auto k_seed1 = make_short_key(enc, 0x10);
  std::ignore = db.insert(k_seed1, vals[0]);
  const auto k_seed2 = make_short_key(enc2, 0x20);
  std::ignore = db.insert(k_seed2, vals[1]);

  // T1 will insert a long key under 0x30 → add_to_nonfull builds a chain.
  unodb::key_encoder enc_t1;
  const auto t1_key = make_key(enc_t1, 0x30, 1);

  const auto hits = run_parked_interleaving(
      unodb::detail::sync_before_nonfull_chain_guard,
      // T1: insert t1_key.  Hits sync point after building chain, pauses.
      // T2 removes seed2, invalidating node version.  T1 resumes, write guard
      // must_restart → deletes chain, retries, succeeds.
      [&] { std::ignore = db.insert(t1_key, vals[2]); },
      // T2: remove seed2 (modifies root I4).
      [&] { std::ignore = db.remove(k_seed2); });
  // T1's park and its retry: T2's remove never reaches the point but moves
  // the root I4's version, failing T1's write-guard upgrade.
  UNODB_ASSERT_EQ(2U, hits);

  const unodb::quiescent_state_on_scope_exit q{};
  ASSERT_VALUE_FOR_KEY(db_type, db, t1_key, vals[2]);
  ASSERT_VALUE_FOR_KEY(db_type, db, k_seed1, vals[0]);
  UNODB_ASSERT_FALSE(db.get(k_seed2).has_value());
}
UNODB_DETAIL_RESTORE_MSVC_WARNINGS()

// Torn-read scan tests.
//
// An OLC reader scanning an I48's or I256's 256 key-byte slots can observe
// every slot empty, because the occupancy set migrates from slots it has not
// reached to slots it has already passed.  The node is never empty at any
// instant; the composite of the scan's loads, taken at different times,
// corresponds to no state it ever held.  Before
// basic_inode_impl::torn_read_result, begin()/last() fell through to
// UNODB_DETAIL_CANNOT_HAPPEN() here.

/// Which scan iteration parks the reader.  The layout derived below places
/// the concurrently filled key bytes strictly on the already-scanned side of
/// the park slot and the seeded ones strictly on the not-yet-scanned side.
constexpr unsigned torn_read_park_hit = 51;
/// The slot the parked forward scan stops before: 50.
constexpr unsigned torn_read_park_slot_fwd = torn_read_park_hit - 1;
/// The slot the parked backward scan stops before: 205.
constexpr unsigned torn_read_park_slot_bwd = 256 - torn_read_park_hit;

/// Where the seeded children go.  One value serves every scenario.
constexpr std::uint8_t torn_read_seed_lo = 100;

/// The inode type a scenario exercises, selected by node type rather than by
/// a bare child count at the call site.
template <unodb::node_type ExpectedType>
using torn_read_inode_type = std::conditional_t<
    ExpectedType == unodb::node_type::I48,
    unodb::detail::olc_inode_48<unodb::key_view, std::uint64_t>,
    unodb::detail::olc_inode_256<unodb::key_view, std::uint64_t>>;

/// Park first() (\a Forward) or last() on \a db at the \a ParkHit-th hit of
/// \a point while \a writer runs, then check the resumed traversal: \a point
/// was hit exactly \a expected_hits times, and the iterator is valid on
/// \a expected_key with \a expected_val.
///
/// Only the traversal may reach \a point — no write path may — so that every
/// hit is the traversal's.  Pinning the count exactly catches an unexpected
/// extra restart and an early exit from the parked attempt, both of which a
/// lower bound would silently tolerate.  Call it through
/// UNODB_ASSERT_NO_FATAL_FAILURE: its checks are fatal asserts, which return
/// only from this function.
template <bool Forward, unsigned ParkHit, class Writer>
void run_parked_extreme_traversal(
    unodb::olc_db<unodb::key_view, std::uint64_t>& db,
    unodb::detail::sync_point& point, const Writer& writer,
    unsigned expected_hits, unodb::key_view expected_key,
    std::uint64_t expected_val) {
  bool valid = false;
  // A copy: the iterator's key buffer goes with it.
  std::vector<std::byte> observed_key;
  std::uint64_t observed_val{0};
  const auto hits = run_parked_interleaving<ParkHit>(
      point,
      [&] {
        auto it = db.test_only_iterator();
        if constexpr (Forward)
          it.first();
        else
          it.last();
        valid = it.valid();
        if (valid) {
          observed_key = key_bytes(it.get_key().view());
          observed_val = it.get_val();
        }
      },
      writer);
  UNODB_ASSERT_EQ(expected_hits, hits);
  UNODB_ASSERT_TRUE(valid);
  UNODB_ASSERT_EQ(observed_key, key_bytes(expected_key));
  UNODB_ASSERT_EQ(observed_val, expected_val);
}

/// Run one torn-read scan scenario on \a ExpectedType.
///
/// Seeds that node type's min_size children at key bytes
/// [torn_read_seed_lo, torn_read_seed_lo + count), which is what makes the
/// root that type.  Parks the scanning thread mid-scan, then has a writer
/// fill an equally sized run of key bytes the parked reader has already
/// passed and erase the seeded ones, which it has not yet reached.  Neither
/// step changes the node type.
///
/// The resumed scan therefore finds every slot empty, check() fails, and the
/// traversal must restart rather than abort, landing on the extreme filled
/// key.  Every layout value is derived here, so a call site cannot
/// perturb the invariants the static_asserts below pin.
template <unodb::node_type ExpectedType, bool Forward>
void run_torn_read_scan_test() {
  using db_type = unodb::olc_db<unodb::key_view, std::uint64_t>;
  using inode_type = torn_read_inode_type<ExpectedType>;
  // Seeding exactly min_size children makes the root ExpectedType.
  constexpr auto count = static_cast<std::uint8_t>(inode_type::min_size);
  // Each key hangs below the root in its own chain I4, so every occupied root
  // slot holds a node pointer, and neither scan depends on the values.
  // seeded[i] gets vals[i], and filled[i] gets vals[count + i].
  constexpr auto vals = distinct_u64_values<2U * count>();
  // The filled run must fit strictly on the already-scanned side of the park
  // slot, in both directions.
  static_assert(count <= torn_read_park_slot_fwd);
  static_assert(256U - count > torn_read_park_slot_bwd);
  // The fill must not grow the node out of ExpectedType.
  static_assert(2U * count <= inode_type::capacity);
  // The seeded run must stay strictly on the not-yet-scanned side, likewise in
  // both directions.
  static_assert(torn_read_seed_lo >= torn_read_park_slot_fwd);
  static_assert(torn_read_seed_lo + count - 1U <= torn_read_park_slot_bwd);

  // Pack the filled run against the extreme of the already-scanned side, so
  // the restart lands on a known key.
  constexpr auto fill_lo =
      static_cast<std::uint8_t>(Forward ? 0U : 256U - count);
  constexpr std::size_t expected_i = Forward ? 0U : count - 1U;

  db_type db;
  unodb::key_encoder enc;
  std::vector<std::array<std::byte, 9>> bufs(static_cast<std::size_t>(count) *
                                             2);
  std::vector<unodb::key_view> seeded;
  std::vector<unodb::key_view> filled;

  for (std::uint8_t i = 0; i < count; ++i) {
    seeded.push_back(make_chain_key(
        enc, static_cast<std::uint8_t>(torn_read_seed_lo + i), 1, bufs[i]));
    UNODB_ASSERT_TRUE(db.insert(seeded.back(), vals[i]));
    filled.push_back(make_chain_key(enc, static_cast<std::uint8_t>(fill_lo + i),
                                    1,
                                    bufs[static_cast<std::size_t>(count) + i]));
  }

#ifdef UNODB_DETAIL_WITH_STATS
  // Pin the node type under test.  Both types scan 256 slots and return the
  // same sentinel, so no other assertion here would notice a mix-up that left
  // one of them uncovered.
  UNODB_ASSERT_EQ(1U, db.get_node_counts()[unodb::as_i<ExpectedType>]);
#endif

  // No write path calls I48/I256 begin()/last(), so only the traversal reaches
  // the instrumented scans.  The expected hits are the 256 slots of the parked
  // scan, which found none occupied -- that fall-through is what returns
  // torn_read_result rather than an ordinary child -- plus the single restart
  // scan, which finds an occupied slot on its first iteration.
  UNODB_ASSERT_NO_FATAL_FAILURE(
      (run_parked_extreme_traversal<Forward, torn_read_park_hit>(
          db, unodb::detail::sync_in_inode_scan,
          // Migrate the occupancy set past the parked reader's cursor.
          [&] {
            for (std::size_t i = 0; i < count; ++i)
              std::ignore = db.insert(filled[i], vals[count + i]);
            for (const auto k : seeded) std::ignore = db.remove(k);
          },
          256U + 1U, filled[expected_i], vals[count + expected_i])));

  const unodb::quiescent_state_on_scope_exit q{};
  for (std::size_t i = 0; i < count; ++i)
    ASSERT_VALUE_FOR_KEY(db_type, db, filled[i], vals[count + i]);
  for (const auto k : seeded) UNODB_ASSERT_FALSE(db.get(k).has_value());
}

/// Run the begin() (forward traversal) torn-read scenario on \a ExpectedType.
template <unodb::node_type ExpectedType>
void run_torn_read_begin_test() {
  run_torn_read_scan_test<ExpectedType, true>();
}

/// Run the last() (backward traversal) torn-read scenario on \a ExpectedType.
template <unodb::node_type ExpectedType>
void run_torn_read_last_test() {
  run_torn_read_scan_test<ExpectedType, false>();
}

/// TR1: root I48 (17 children, basic_inode_48's min_size) at key bytes
/// 100-116.  Reader parks at slot 50 having passed 0-16; writer fills 0-16
/// (34 children, under the 48 grow threshold) and erases 100-116.  The
/// resumed scan finds nothing; first() restarts onto key byte 0.
UNODB_TEST(OLCIteratorTornRead, I48BeginObservesAllSlotsEmpty) {
  run_torn_read_begin_test<unodb::node_type::I48>();
}

/// TR2: mirror of TR1 for last(), which scans 255 -> 0.  Reader parks at slot
/// 205 having passed 255-239; writer fills 239-255 and erases 100-116.
UNODB_TEST(OLCIteratorTornRead, I48LastObservesAllSlotsEmpty) {
  run_torn_read_last_test<unodb::node_type::I48>();
}

/// TR3: root I256 (49 children, basic_inode_256's min_size) at key bytes
/// 100-148.  Reader parks at slot 50 having passed 0-48; writer fills 0-48
/// (98 children) and erases 100-148.
UNODB_TEST(OLCIteratorTornRead, I256BeginObservesAllSlotsEmpty) {
  run_torn_read_begin_test<unodb::node_type::I256>();
}

/// TR4: mirror of TR3 for last().  Reader parks at slot 205 having passed
/// 255-207; writer fills 207-255 and erases 100-148.
UNODB_TEST(OLCIteratorTornRead, I256LastObservesAllSlotsEmpty) {
  run_torn_read_last_test<unodb::node_type::I256>();
}

// Torn-read descent tests.
//
// After an I48 scan resolved a key byte, try_left_most_traversal() and
// try_right_most_traversal() re-resolve it twice through child_indexes, in
// is_value_in_slot() and then get_child().  A removal landing between the scan
// and those loads clears the entry in place, so both observe empty_child; each
// has a guard for exactly that, and neither guard had a deterministic test.

/// Run one torn-read descent scenario.
///
/// Seeds an I48 root with min_size + 1 one-byte keys, packed values at key
/// bytes [torn_read_seed_lo, torn_read_seed_lo + count), so that removing one
/// child keeps the root an I48: the writer takes the in-place removal path,
/// and the reader resumes on a live node whose version check() rejects.  A
/// shrinking removal would clear the entry too, but the reader would then
/// resume on an obsoleted node.  Parks the reader after its scan resolved
/// the extreme key byte, before is_value_in_slot() and get_child() re-resolve
/// it, and has a writer remove that key.  Both re-resolutions then observe
/// empty_child, check() fails, and the traversal must restart, landing on the
/// next key in.
template <bool Forward>
void run_torn_read_descent_test() {
  using db_type = unodb::olc_db<unodb::key_view, std::uint64_t>;
  using inode_type = torn_read_inode_type<unodb::node_type::I48>;

  constexpr auto count = static_cast<std::uint8_t>(inode_type::min_size + 1);
  static_assert(count <= inode_type::capacity);
  constexpr auto vals = distinct_u64_values<count>();
  // Only value-in-slot mode makes the descent call is_value_in_slot(); in a
  // leaf-backed Db the test would still pass but cover get_child()'s guard
  // alone.
  static_assert(!has_leaf_nodes_v<db_type>,
                "is_value_in_slot()'s empty_child guard is reached only in "
                "value-in-slot mode");
  constexpr std::size_t removed_i = Forward ? 0 : count - 1U;
  constexpr std::size_t expected_i = Forward ? 1U : count - 2U;

  db_type db;
  unodb::key_encoder enc;
  std::vector<std::array<std::byte, 1>> bufs(count);
  std::vector<unodb::key_view> keys;
  for (std::uint8_t i = 0; i < count; ++i) {
    keys.push_back(copy_key(
        make_short_key(enc, static_cast<std::uint8_t>(torn_read_seed_lo + i)),
        bufs[i]));
    UNODB_ASSERT_TRUE(db.insert(keys.back(), vals[i]));
  }
  const auto removed = keys[removed_i];

#ifdef UNODB_DETAIL_WITH_STATS
  // Pin the shape: a single I48 root with no I4 below it, so the descent
  // re-resolves the scanned key byte in that node and nowhere else.
  UNODB_ASSERT_EQ(1U, db.get_node_counts()[unodb::as_i<unodb::node_type::I48>]);
  UNODB_ASSERT_EQ(0U, db.get_node_counts()[unodb::as_i<unodb::node_type::I4>]);
#endif

  // MSVC's analysis misses the write in the lambda below.
  UNODB_DETAIL_DISABLE_MSVC_WARNING(26496)
  bool removed_ok = false;
  UNODB_DETAIL_RESTORE_MSVC_WARNINGS()
  // Only the traversal reaches the sync point.  The expected hits are the
  // parked traversal and the single restart, which passes the point once more
  // on its way to the packed value.
  UNODB_ASSERT_NO_FATAL_FAILURE((run_parked_extreme_traversal<Forward, 1>(
      db, unodb::detail::sync_before_descent_child_loads,
      // Remove the child the parked reader has resolved.
      [&] { removed_ok = db.remove(removed); }, 2U, keys[expected_i],
      vals[expected_i])));
  UNODB_ASSERT_TRUE(removed_ok);

  const unodb::quiescent_state_on_scope_exit q{};
  UNODB_ASSERT_FALSE(db.get(removed).has_value());
  for (std::size_t i = 0; i < keys.size(); ++i) {
    if (i == removed_i) continue;
    ASSERT_VALUE_FOR_KEY(db_type, db, keys[i], vals[i]);
  }
}

/// TR5: root I48 (18 children, one above basic_inode_48's min_size) at key
/// bytes 100-117.  Reader parks after begin() resolved 100; writer removes it
/// in place.  is_value_in_slot() and get_child() both observe empty_child,
/// check() fails, and first() restarts onto key byte 101.
UNODB_TEST(OLCIteratorTornRead, I48BeginObservesScannedChildRemoved) {
  run_torn_read_descent_test<true>();
}

/// TR6: mirror of TR5 for last().  Reader parks after last() resolved 117;
/// writer removes it in place; last() restarts onto key byte 116.
UNODB_TEST(OLCIteratorTornRead, I48LastObservesScannedChildRemoved) {
  run_torn_read_descent_test<false>();
}

// ===================================================================
// Torn key-prefix read: an I4 collapse lengthens the surviving inode's key
// prefix in place, so a descent must take its prefix length and prefix bytes
// from one read.
// ===================================================================

/// T1's duplicate insert of 0x100 parks at the I4 below Root-I4, once it has
/// read the length of that I4's key prefix, while T2's remove of 1 collapses
/// Root-I4 into it, prepending Root-I4's six-byte prefix and the I4's dispatch
/// byte.  T1 must take the shared length from the prefix it read, not from
/// the merged one: clamped at the merged length 7, that gives 6, above the
/// length 0 T1 read, and trips the assert that the two agree before T1 can
/// restart on the collapse.
UNODB_TEST(OLCTornPrefixRead, CollapseAfterInsertReadsPrefix) {
  using db_type = unodb::olc_db<std::uint64_t, unodb::value_view>;
  // The values of 1, 0x100 and 0x101.
  constexpr auto vals = distinct_values<3>();

  db_type db;
  // Root-I4, prefix 00 x 6, holds 1's leaf at 0x00 and, at 0x01, an I4 with an
  // empty prefix that holds 0x100 and 0x101.
  UNODB_ASSERT_TRUE(db.insert(1, vals[0]));
  UNODB_ASSERT_TRUE(db.insert(0x100, vals[1]));
  UNODB_ASSERT_TRUE(db.insert(0x101, vals[2]));

  bool inserted = true;
  bool removed = false;
  const auto hits = run_parked_interleaving<2>(
      unodb::detail::sync_after_prefix_length_load,
      [&] { inserted = db.insert(0x100, vals[1]); },
      [&] { removed = db.remove(1); });

  UNODB_ASSERT_TRUE(removed);
  UNODB_ASSERT_FALSE(inserted);
  // Root-I4 and the I4 below it, T2's pass through Root-I4 in its remove,
  // then T1's retry from that I4, now the root, after the obsolete Root-I4
  // failed its check.
  UNODB_ASSERT_EQ(4U, hits);
  const unodb::quiescent_state_on_scope_exit q{};
  UNODB_ASSERT_FALSE(db.get(1).has_value());
  ASSERT_VALUE_FOR_KEY(db_type, db, 0x100, vals[1]);
  ASSERT_VALUE_FOR_KEY(db_type, db, 0x101, vals[2]);
}

#endif  // NDEBUG

}  // namespace

UNODB_DETAIL_RESTORE_MSVC_WARNINGS()
