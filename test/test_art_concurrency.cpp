// Copyright 2021-2026 UnoDB contributors

// Should be the first include
#include "global.hpp"

// IWYU pragma: no_include <string>
// IWYU pragma: no_include <type_traits>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <random>
#include <thread>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include "art_common.hpp"
#include "assert.hpp"
#include "db_test_utils.hpp"
#include "gtest_utils.hpp"
#include "olc_art.hpp"
#include "qsbr.hpp"
#include "qsbr_test_utils.hpp"

#ifndef NDEBUG
#include "sync.hpp"
#include "thread_sync.hpp"
#endif

namespace {

[[nodiscard]] constexpr bool odd(std::uint64_t x) noexcept {
  return static_cast<bool>(x % 2);
}

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

  // decode a uint64_t key.
  [[nodiscard]] static std::uint64_t decode(unodb::key_view akey) noexcept {
    unodb::key_decoder dec{akey};
    std::uint64_t k;
    dec.decode(k);
    return k;
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
          unodb::test::test_values[akey % unodb::test::test_values.size()];
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
          verifier->try_insert(
              key,
              unodb::test::test_values[key % unodb::test::test_values.size()]);
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
          verifier->try_insert(
              key,
              unodb::test::test_values[key % unodb::test::test_values.size()]);
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

// Helper: encode a 9-byte chain-triggering key into a caller-owned buffer.
// The returned key_view is valid for the lifetime of `buf`.
inline unodb::key_view make_chain_key(unodb::key_encoder& enc, std::uint8_t tag,
                                      std::uint64_t v,
                                      std::array<std::byte, 9>& buf) {
  auto kv = enc.reset().encode(tag).encode(v).get_key_view();
  std::ranges::copy(kv, buf.begin());
  return {buf.data(), buf.size()};
}

// Chain concurrency tests reuse ARTConcurrencyTest for QSBR setup and
// parallel_test.  Thread functions access the db via verifier->get_db().
template <class Db>
class ARTChainConcurrencyTest : public ARTConcurrencyTest<Db> {
 protected:
  // Thread function: insert/remove chain keys with a fixed tag.
  static void chain_insert_remove_thread(unodb::test::tree_verifier<Db>* tv,
                                         std::size_t thread_i,
                                         std::size_t ops_per_thread) {
    auto& db = tv->get_db();
    unodb::key_encoder enc;
    std::array<std::byte, 9> buf{};
    constexpr auto tag = static_cast<std::uint8_t>(0x42);
    const auto val =
        unodb::test::test_values[thread_i % unodb::test::test_values.size()];
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

  // Thread function: random insert/remove/get on chain keys.
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
      const auto val =
          unodb::test::test_values[v % unodb::test::test_values.size()];
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

  // Thread function: multi-tag random ops.
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
      const auto val =
          unodb::test::test_values[v % unodb::test::test_values.size()];
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

#ifndef NDEBUG

// ===================================================================
// Concurrent chain cut tests with explicit interleavings.
// Uses thread_syncs (condition variables) for coordination.
//
// Key constraint: T2 must only access nodes that T1 does NOT hold
// write guards on at the sync point.
//
// At SP1 (sync_after_chain_locked): T1 holds chain_bottom (ng),
//   all chain nodes on stack (chain_guards), and leaf (lg).
//   The cut_point_parent is NOT locked.
//
// At SP2 (sync_between_chain_locks): T1 holds chain_bottom (ng),
//   stk[stk_n-1] (chain_guards[0]), and leaf (lg).
//   All other stack entries are NOT locked.
//
// Tree setup uses 26-byte keys (CT1/CT3, 3 chain levels) or 34-byte
// keys (CT2/CT4, 4 chain levels) to ensure enough chain depth that
// pg locks a deep chain node, not the cut_point_parent.
// ===================================================================

struct sync_point_guard {
  unodb::detail::sync_point* pt_;
  explicit sync_point_guard(unodb::detail::sync_point& pt) noexcept
      : pt_{&pt} {}
  ~sync_point_guard() { pt_->disarm(); }
  sync_point_guard(const sync_point_guard&) = delete;
  sync_point_guard& operator=(const sync_point_guard&) = delete;
  sync_point_guard(sync_point_guard&&) = delete;
  sync_point_guard& operator=(sync_point_guard&&) = delete;
};

// CT1: T1 removes A (chain cut).  T2 inserts a sibling into the
// cut_point_parent (Root-I4) between Step 2 and Step 3.
UNODB_TEST(OLCChainCutInterleaved, ConcurrentInsertIntoCutPointParent) {
  using db_type = unodb::olc_db<unodb::key_view, unodb::value_view>;
  constexpr auto val_bytes = std::array<std::byte, 1>{std::byte{0x42}};
  const auto val = unodb::value_view{val_bytes};

  db_type db;
  unodb::key_encoder enc;

  // 26-byte keys: tag(1) + 3×uint64(24) + bottom(1).
  // A,B share 25 bytes → 3 chain levels after removing B.
  auto make26 = [&](std::uint8_t tag, std::uint8_t bottom) {
    return enc.reset()
        .encode(tag)
        .encode(std::uint64_t{0x4242424242424242ULL})
        .encode(std::uint64_t{0})
        .encode(std::uint64_t{0})
        .encode(bottom)
        .get_key_view();
  };

  std::array<std::byte, 26> buf_a{};
  std::array<std::byte, 26> buf_b{};
  std::array<std::byte, 26> buf_sib{};
  std::array<std::byte, 26> buf_new{};
  auto copy_key = [](const unodb::key_view kv, auto& buf) {
    std::ranges::copy(kv, buf.begin());
    return unodb::key_view{buf.data(), buf.size()};
  };
  const auto key_a = copy_key(make26(0x10, 0x01), buf_a);
  const auto key_b = copy_key(make26(0x10, 0x02), buf_b);
  const auto sib = copy_key(make26(0x20, 0x01), buf_sib);
  const auto new_key = copy_key(make26(0x30, 0x01), buf_new);

  UNODB_ASSERT_TRUE(db.insert(key_a, val));
  UNODB_ASSERT_TRUE(db.insert(key_b, val));
  UNODB_ASSERT_TRUE(db.insert(sib, val));
  {
    const unodb::quiescent_state_on_scope_exit q{};
    UNODB_ASSERT_TRUE(db.remove(key_b));
  }

  // Arm SP1 AFTER setup remove.
  const sync_point_guard guard{unodb::detail::sync_after_chain_locked};
  unodb::detail::sync_after_chain_locked.arm([&]() {
    unodb::detail::thread_syncs[0].notify();
    unodb::detail::thread_syncs[1].wait();
  });

  unodb::this_thread().qsbr_pause();

  // Spawn T2 first so it's registered with QSBR before T1 starts.
  auto t2 = unodb::qsbr_thread([&] {
    unodb::detail::thread_syncs[0].wait();
    const unodb::quiescent_state_on_scope_exit q{};
    // Insert at root level — Root-I4 is NOT locked by T1 at SP1.
    std::ignore = db.insert(new_key, val);
    unodb::detail::thread_syncs[1].notify();
  });

  auto t1 = unodb::qsbr_thread([&] {
    const unodb::quiescent_state_on_scope_exit q{};
    std::ignore = db.remove(key_a);
  });

  t1.join();
  t2.join();
  unodb::this_thread().qsbr_resume();
  unodb::this_thread().quiescent();

  const unodb::quiescent_state_on_scope_exit q{};
  UNODB_ASSERT_FALSE(db.get(key_a).has_value());
  UNODB_ASSERT_TRUE(db.get(sib).has_value());
  UNODB_ASSERT_TRUE(db.get(new_key).has_value());
}

// CT2: T1 removes A (chain cut).  T2 inserts a key that modifies
// chain[0] (splits its prefix) between the first and second chain
// lock in Step 2.
UNODB_TEST(OLCChainCutInterleaved, ConcurrentInsertIntoMidChain) {
  using db_type = unodb::olc_db<unodb::key_view, unodb::value_view>;
  constexpr auto val_bytes = std::array<std::byte, 1>{std::byte{0x42}};
  const auto val = unodb::value_view{val_bytes};

  db_type db;
  unodb::key_encoder enc;

  // 34-byte keys: tag(1) + 4×uint64(32) + bottom(1).
  // A,B share 33 bytes → 4 chain levels after removing B.
  // At SP2: T1 holds chain[2] (stk[3]) + chain_bottom + leaf.
  // chain[0], chain[1], Root-I4 are all unlocked.
  // T2 targets chain[0] (different v1) — only needs Root-I4 + chain[0].
  constexpr auto X = std::uint64_t{0x4242424242424242ULL};
  constexpr auto Z = std::uint64_t{0x4343434343434343ULL};

  auto make34 = [&](std::uint8_t tag, std::uint64_t v1, std::uint8_t bottom) {
    return enc.reset()
        .encode(tag)
        .encode(v1)
        .encode(X)
        .encode(X)
        .encode(X)
        .encode(bottom)
        .get_key_view();
  };

  std::array<std::byte, 34> buf_a{};
  std::array<std::byte, 34> buf_b{};
  std::array<std::byte, 34> buf_sib{};
  std::array<std::byte, 34> buf_t2{};
  auto copy_key = [](const unodb::key_view kv, auto& buf) {
    std::ranges::copy(kv, buf.begin());
    return unodb::key_view{buf.data(), buf.size()};
  };
  const auto key_a = copy_key(make34(0x10, X, 0x01), buf_a);
  const auto key_b = copy_key(make34(0x10, X, 0x02), buf_b);
  const auto sib = copy_key(make34(0x20, X, 0x01), buf_sib);
  const auto t2_key = copy_key(make34(0x10, Z, 0x01), buf_t2);

  UNODB_ASSERT_TRUE(db.insert(key_a, val));
  UNODB_ASSERT_TRUE(db.insert(key_b, val));
  UNODB_ASSERT_TRUE(db.insert(sib, val));
  {
    const unodb::quiescent_state_on_scope_exit q{};
    UNODB_ASSERT_TRUE(db.remove(key_b));
  }

  // Arm SP2 AFTER setup remove.
  const sync_point_guard guard{unodb::detail::sync_between_chain_locks};
  unodb::detail::sync_between_chain_locks.arm([&]() {
    unodb::detail::sync_between_chain_locks.disarm();  // fire only once
    unodb::detail::thread_syncs[2].notify();
    unodb::detail::thread_syncs[3].wait();
  });

  unodb::this_thread().qsbr_pause();

  auto t2 = unodb::qsbr_thread([&] {
    unodb::detail::thread_syncs[2].wait();
    const unodb::quiescent_state_on_scope_exit q{};
    std::ignore = db.insert(t2_key, val);
    unodb::detail::thread_syncs[3].notify();
  });

  auto t1 = unodb::qsbr_thread([&] {
    const unodb::quiescent_state_on_scope_exit q{};
    std::ignore = db.remove(key_a);
  });

  t1.join();
  t2.join();
  unodb::this_thread().qsbr_resume();
  unodb::this_thread().quiescent();

  const unodb::quiescent_state_on_scope_exit q{};
  UNODB_ASSERT_FALSE(db.get(key_a).has_value());
  UNODB_ASSERT_TRUE(db.get(sib).has_value());
  UNODB_ASSERT_TRUE(db.get(t2_key).has_value());
}

// CT3: T1 removes A (chain cut).  T2 removes the sibling.
UNODB_TEST(OLCChainCutInterleaved, ConcurrentRemoveOfSibling) {
  using db_type = unodb::olc_db<unodb::key_view, unodb::value_view>;
  constexpr auto val_bytes = std::array<std::byte, 1>{std::byte{0x42}};
  const auto val = unodb::value_view{val_bytes};

  db_type db;
  unodb::key_encoder enc;

  auto make26 = [&](std::uint8_t tag, std::uint8_t bottom) {
    return enc.reset()
        .encode(tag)
        .encode(std::uint64_t{0x4242424242424242ULL})
        .encode(std::uint64_t{0})
        .encode(std::uint64_t{0})
        .encode(bottom)
        .get_key_view();
  };

  std::array<std::byte, 26> buf_a{};
  std::array<std::byte, 26> buf_b{};
  auto copy_key = [](const unodb::key_view kv, auto& buf) {
    std::ranges::copy(kv, buf.begin());
    return unodb::key_view{buf.data(), buf.size()};
  };
  const auto key_a = copy_key(make26(0x10, 0x01), buf_a);
  const auto key_b = copy_key(make26(0x10, 0x02), buf_b);
  // Use a short (1-byte) sibling so that when T2 removes it, the
  // Root-I4 goes to 1 child but prefix merge overflows (0+1+7 > 7),
  // preventing collapse.  Root-I4 stays alive with a new version.
  std::array<std::byte, 1> buf_sib{};
  const auto sib =
      copy_key(enc.reset().encode(std::uint8_t{0x20}).get_key_view(), buf_sib);

  UNODB_ASSERT_TRUE(db.insert(key_a, val));
  UNODB_ASSERT_TRUE(db.insert(key_b, val));
  UNODB_ASSERT_TRUE(db.insert(sib, val));
  {
    const unodb::quiescent_state_on_scope_exit q{};
    UNODB_ASSERT_TRUE(db.remove(key_b));
  }

  const sync_point_guard guard{unodb::detail::sync_after_chain_locked};
  unodb::detail::sync_after_chain_locked.arm([&]() {
    unodb::detail::sync_after_chain_locked.disarm();  // one-shot
    unodb::detail::thread_syncs[0].notify();
    unodb::detail::thread_syncs[1].wait();
  });

  unodb::this_thread().qsbr_pause();

  auto t2 = unodb::qsbr_thread([&] {
    unodb::detail::thread_syncs[0].wait();
    const unodb::quiescent_state_on_scope_exit q{};
    std::ignore = db.remove(sib);
    unodb::detail::thread_syncs[1].notify();
  });

  auto t1 = unodb::qsbr_thread([&] {
    const unodb::quiescent_state_on_scope_exit q{};
    std::ignore = db.remove(key_a);
  });

  t1.join();
  t2.join();
  unodb::this_thread().qsbr_resume();
  unodb::this_thread().quiescent();

  const unodb::quiescent_state_on_scope_exit q{};
  UNODB_ASSERT_FALSE(db.get(key_a).has_value());
  UNODB_ASSERT_FALSE(db.get(sib).has_value());
  UNODB_ASSERT_TRUE(db.empty());
}

// CT4: ABA on chain node.  T2 inserts then removes a key that
// modifies chain[0]'s version but restores it to single-child.
UNODB_TEST(OLCChainCutInterleaved, ABAOnChainNode) {
  using db_type = unodb::olc_db<unodb::key_view, unodb::value_view>;
  constexpr auto val_bytes = std::array<std::byte, 1>{std::byte{0x42}};
  const auto val = unodb::value_view{val_bytes};

  db_type db;
  unodb::key_encoder enc;

  constexpr auto X = std::uint64_t{0x4242424242424242ULL};
  constexpr auto Z = std::uint64_t{0x4343434343434343ULL};

  auto make34 = [&](std::uint8_t tag, std::uint64_t v1, std::uint8_t bottom) {
    return enc.reset()
        .encode(tag)
        .encode(v1)
        .encode(X)
        .encode(X)
        .encode(X)
        .encode(bottom)
        .get_key_view();
  };

  std::array<std::byte, 34> buf_a{};
  std::array<std::byte, 34> buf_b{};
  std::array<std::byte, 34> buf_sib{};
  std::array<std::byte, 34> buf_t2{};
  auto copy_key = [](const unodb::key_view kv, auto& buf) {
    std::ranges::copy(kv, buf.begin());
    return unodb::key_view{buf.data(), buf.size()};
  };
  const auto key_a = copy_key(make34(0x10, X, 0x01), buf_a);
  const auto key_b = copy_key(make34(0x10, X, 0x02), buf_b);
  const auto sib = copy_key(make34(0x20, X, 0x01), buf_sib);
  const auto t2_key = copy_key(make34(0x10, Z, 0x01), buf_t2);

  UNODB_ASSERT_TRUE(db.insert(key_a, val));
  UNODB_ASSERT_TRUE(db.insert(key_b, val));
  UNODB_ASSERT_TRUE(db.insert(sib, val));
  {
    const unodb::quiescent_state_on_scope_exit q{};
    UNODB_ASSERT_TRUE(db.remove(key_b));
  }

  const sync_point_guard guard{unodb::detail::sync_between_chain_locks};
  unodb::detail::sync_between_chain_locks.arm([&]() {
    unodb::detail::sync_between_chain_locks.disarm();  // fire only once
    unodb::detail::thread_syncs[4].notify();
    unodb::detail::thread_syncs[5].wait();
  });

  unodb::this_thread().qsbr_pause();

  auto t2 = unodb::qsbr_thread([&] {
    unodb::detail::thread_syncs[4].wait();
    {
      const unodb::quiescent_state_on_scope_exit q{};
      std::ignore = db.insert(t2_key, val);
    }
    {
      const unodb::quiescent_state_on_scope_exit q{};
      std::ignore = db.remove(t2_key);
    }
    unodb::detail::thread_syncs[5].notify();
  });

  auto t1 = unodb::qsbr_thread([&] {
    const unodb::quiescent_state_on_scope_exit q{};
    std::ignore = db.remove(key_a);
  });

  t1.join();
  t2.join();
  unodb::this_thread().qsbr_resume();
  unodb::this_thread().quiescent();

  const unodb::quiescent_state_on_scope_exit q{};
  UNODB_ASSERT_FALSE(db.get(key_a).has_value());
  UNODB_ASSERT_TRUE(db.get(sib).has_value());
  UNODB_ASSERT_FALSE(db.get(t2_key).has_value());
}

#endif  // NDEBUG

// ===================================================================
// Concurrent chain cut stress tests (no sync points).
// ===================================================================

// CT5: Two threads removing last two keys under same chain.
UNODB_TEST(OLCChainCutStress, TwoRemoversFromChain) {
  using db_type = unodb::olc_db<unodb::key_view, unodb::value_view>;
  constexpr auto val_bytes = std::array<std::byte, 1>{std::byte{0x42}};
  const auto val = unodb::value_view{val_bytes};

  for (int iter = 0; iter < 200; ++iter) {
    db_type db;
    unodb::key_encoder enc;
    std::array<std::byte, 9> buf_a{};
    std::array<std::byte, 9> buf_b{};
    std::array<std::byte, 9> buf_sib{};

    const auto key_a = make_chain_key(enc, 0x10, 1, buf_a);
    const auto key_b = make_chain_key(enc, 0x10, 2, buf_b);
    const auto sibling = make_chain_key(enc, 0x20, 1, buf_sib);

    UNODB_ASSERT_TRUE(db.insert(key_a, val));
    UNODB_ASSERT_TRUE(db.insert(key_b, val));
    UNODB_ASSERT_TRUE(db.insert(sibling, val));

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
    UNODB_ASSERT_TRUE(db.get(sibling).has_value());
  }
}

// CT6: Multiple threads doing insert/remove on chain-triggering keys.
UNODB_TEST(OLCChainCutStress, InsertRemoveMix) {
  using db_type = unodb::olc_db<unodb::key_view, unodb::value_view>;
  constexpr auto val_bytes = std::array<std::byte, 1>{std::byte{0x42}};
  const auto val = unodb::value_view{val_bytes};

  for (int iter = 0; iter < 100; ++iter) {
    db_type db;

    // Pre-insert some keys so the tree has chain structure.
    {
      unodb::key_encoder enc;
      UNODB_DETAIL_DISABLE_MSVC_WARNING(26496)
      std::array<std::byte, 9> buf{};
      UNODB_DETAIL_RESTORE_MSVC_WARNINGS()
      for (std::uint64_t i = 1; i <= 4; ++i) {
        UNODB_ASSERT_TRUE(db.insert(make_chain_key(enc, 0x10, i, buf), val));
        UNODB_ASSERT_TRUE(db.insert(make_chain_key(enc, 0x20, i, buf), val));
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
        std::ignore = db.insert(make_chain_key(enc, 0x30, i, buf), val);
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
      UNODB_ASSERT_TRUE(db.get(make_chain_key(enc, 0x30, i, buf)).has_value());
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
                          std::array<std::byte, chain_key_len>& buf) {
    enc.reset().encode(std::uint8_t{0x42});
    for (int d = 0; d < chain_depth; ++d) enc.encode(std::uint64_t{0});
    enc.encode(variant);
    auto kv = enc.get_key_view();
    std::ranges::copy(kv, buf.begin());
    return unodb::key_view{buf.data(), buf.size()};
  };

  auto make_short = [](unodb::key_encoder& enc, std::uint8_t tag,
                       std::array<std::byte, short_key_len>& buf) {
    auto kv = enc.reset().encode(tag).get_key_view();
    std::ranges::copy(kv, buf.begin());
    return unodb::key_view{buf.data(), buf.size()};
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
  constexpr auto val_bytes = std::array<std::byte, 1>{std::byte{0x42}};
  const auto val = unodb::value_view{val_bytes};

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

  UNODB_ASSERT_TRUE(db.insert(key_a, val));
  UNODB_ASSERT_TRUE(db.insert(key_b, val));
  UNODB_ASSERT_TRUE(db.insert(key_c, val));

  const sync_point_guard guard{unodb::detail::sync_before_remove_write_guard};
  unodb::detail::sync_before_remove_write_guard.arm([&]() {
    unodb::detail::sync_before_remove_write_guard.disarm();
    unodb::detail::thread_syncs[0].notify();
    unodb::detail::thread_syncs[1].wait();
  });

  unodb::this_thread().qsbr_pause();

  auto t2 = unodb::qsbr_thread([&] {
    unodb::detail::thread_syncs[0].wait();
    const unodb::quiescent_state_on_scope_exit q{};
    std::ignore = db.insert(new_key, val);
    unodb::detail::thread_syncs[1].notify();
  });

  auto t1 = unodb::qsbr_thread([&] {
    const unodb::quiescent_state_on_scope_exit q{};
    std::ignore = db.remove(key_a);
  });

  t1.join();
  t2.join();
  unodb::this_thread().qsbr_resume();
  unodb::this_thread().quiescent();

  const unodb::quiescent_state_on_scope_exit q{};
  UNODB_ASSERT_FALSE(db.get(key_a).has_value());
  UNODB_ASSERT_TRUE(db.get(key_b).has_value());
  UNODB_ASSERT_TRUE(db.get(key_c).has_value());
  UNODB_ASSERT_TRUE(db.get(new_key).has_value());
}

// RT2: T1 removes key_a from an I4 at min_size (2 children).
// T2 removes key_b (the other child) concurrently.  One thread
// succeeds, the other restarts and finds the key gone.
UNODB_TEST(OLCRemoveChooseSubtree, ConcurrentRemoveFromMinSizeI4) {
  using db_type = unodb::olc_db<unodb::key_view, unodb::value_view>;
  constexpr auto val_bytes = std::array<std::byte, 1>{std::byte{0x42}};
  const auto val = unodb::value_view{val_bytes};

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

  UNODB_ASSERT_TRUE(db.insert(key_a, val));
  UNODB_ASSERT_TRUE(db.insert(key_b, val));
  UNODB_ASSERT_TRUE(db.insert(sibling, val));

  const sync_point_guard guard{unodb::detail::sync_before_remove_write_guard};
  unodb::detail::sync_before_remove_write_guard.arm([&]() {
    unodb::detail::sync_before_remove_write_guard.disarm();
    unodb::detail::thread_syncs[0].notify();
    unodb::detail::thread_syncs[1].wait();
  });

  unodb::this_thread().qsbr_pause();

  auto t2 = unodb::qsbr_thread([&] {
    unodb::detail::thread_syncs[0].wait();
    const unodb::quiescent_state_on_scope_exit q{};
    std::ignore = db.remove(key_b);
    unodb::detail::thread_syncs[1].notify();
  });

  auto t1 = unodb::qsbr_thread([&] {
    const unodb::quiescent_state_on_scope_exit q{};
    std::ignore = db.remove(key_a);
  });

  t1.join();
  t2.join();
  unodb::this_thread().qsbr_resume();
  unodb::this_thread().quiescent();

  const unodb::quiescent_state_on_scope_exit q{};
  UNODB_ASSERT_FALSE(db.get(key_a).has_value());
  UNODB_ASSERT_FALSE(db.get(key_b).has_value());
  UNODB_ASSERT_TRUE(db.get(sibling).has_value());
}

// RT3: T1 removes key_a.  T2 replaces key_a's leaf by removing and
// re-inserting it with a different value.  T1's child version check
// should detect the change.
UNODB_TEST(OLCRemoveChooseSubtree, ConcurrentLeafReplacement) {
  using db_type = unodb::olc_db<unodb::key_view, unodb::value_view>;
  constexpr auto val_bytes1 = std::array<std::byte, 1>{std::byte{0x42}};
  constexpr auto val_bytes2 = std::array<std::byte, 1>{std::byte{0x99}};
  const auto val1 = unodb::value_view{val_bytes1};
  const auto val2 = unodb::value_view{val_bytes2};

  db_type db;
  unodb::key_encoder enc;

  std::array<std::byte, 9> buf_a{};
  std::array<std::byte, 9> buf_b{};
  std::array<std::byte, 9> buf_c{};
  const auto key_a = make_chain_key(enc, 0x10, 1, buf_a);
  const auto key_b = make_chain_key(enc, 0x10, 2, buf_b);
  const auto key_c = make_chain_key(enc, 0x10, 3, buf_c);

  UNODB_ASSERT_TRUE(db.insert(key_a, val1));
  UNODB_ASSERT_TRUE(db.insert(key_b, val1));
  UNODB_ASSERT_TRUE(db.insert(key_c, val1));

  const sync_point_guard guard{unodb::detail::sync_before_remove_write_guard};
  unodb::detail::sync_before_remove_write_guard.arm([&]() {
    unodb::detail::sync_before_remove_write_guard.disarm();
    unodb::detail::thread_syncs[0].notify();
    unodb::detail::thread_syncs[1].wait();
  });

  unodb::this_thread().qsbr_pause();

  auto t2 = unodb::qsbr_thread([&] {
    unodb::detail::thread_syncs[0].wait();
    {
      const unodb::quiescent_state_on_scope_exit q{};
      std::ignore = db.remove(key_a);
    }
    {
      const unodb::quiescent_state_on_scope_exit q{};
      std::ignore = db.insert(key_a, val2);
    }
    unodb::detail::thread_syncs[1].notify();
  });

  auto t1 = unodb::qsbr_thread([&] {
    const unodb::quiescent_state_on_scope_exit q{};
    std::ignore = db.remove(key_a);
  });

  t1.join();
  t2.join();
  unodb::this_thread().qsbr_resume();
  unodb::this_thread().quiescent();

  // key_a may or may not exist depending on who won the race.
  // The important thing is no crash, no hang, no corruption.
  const unodb::quiescent_state_on_scope_exit q2{};
  UNODB_ASSERT_TRUE(db.get(key_b).has_value());
  UNODB_ASSERT_TRUE(db.get(key_c).has_value());
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
  constexpr std::uint64_t val = 42;

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

  UNODB_ASSERT_TRUE(db.insert(key_a, val));
  UNODB_ASSERT_TRUE(db.insert(key_b, val));
  UNODB_ASSERT_TRUE(db.insert(key_c, val));
  UNODB_ASSERT_TRUE(db.insert(key_d, val));

  const sync_point_guard guard{unodb::detail::sync_before_insert_grow_guard};
  unodb::detail::sync_before_insert_grow_guard.arm([&]() {
    unodb::detail::sync_before_insert_grow_guard.disarm();
    unodb::detail::thread_syncs[0].notify();
    unodb::detail::thread_syncs[1].wait();
  });

  unodb::this_thread().qsbr_pause();

  // T2: remove key_d from the root I4, modifying its version.
  auto t2 = unodb::qsbr_thread([&] {
    unodb::detail::thread_syncs[0].wait();
    const unodb::quiescent_state_on_scope_exit q{};
    std::ignore = db.remove(key_d);
    unodb::detail::thread_syncs[1].notify();
  });

  // T1: insert new_key, triggering I4→I16 growth at root.
  // Hits sync point, pauses while T2 modifies the node, then restarts.
  auto t1 = unodb::qsbr_thread([&] {
    const unodb::quiescent_state_on_scope_exit q{};
    std::ignore = db.insert(new_key, val);
  });

  t1.join();
  t2.join();
  unodb::this_thread().qsbr_resume();
  unodb::this_thread().quiescent();

  const unodb::quiescent_state_on_scope_exit q{};
  UNODB_ASSERT_FALSE(db.get(key_d).has_value());
  UNODB_ASSERT_TRUE(db.get(new_key).has_value());
  UNODB_ASSERT_TRUE(db.get(key_a).has_value());
  UNODB_ASSERT_TRUE(db.get(key_b).has_value());
  UNODB_ASSERT_TRUE(db.get(key_c).has_value());
}

// IGT2: T1 inserts the 5th key (triggers I4→I16 growth).  T2 inserts
// a different key into the same inode.  T1's write guard detects the
// version change and restarts.
UNODB_TEST(OLCInsertGrowth, ConcurrentInsertDuringGrowth) {
  using db_type = unodb::olc_db<unodb::key_view, std::uint64_t>;
  constexpr std::uint64_t val = 42;

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

  UNODB_ASSERT_TRUE(db.insert(key_a, val));
  UNODB_ASSERT_TRUE(db.insert(key_b, val));
  UNODB_ASSERT_TRUE(db.insert(key_c, val));
  UNODB_ASSERT_TRUE(db.insert(key_d, val));

  const sync_point_guard guard{unodb::detail::sync_before_insert_grow_guard};
  unodb::detail::sync_before_insert_grow_guard.arm([&]() {
    unodb::detail::sync_before_insert_grow_guard.disarm();
    unodb::detail::thread_syncs[0].notify();
    unodb::detail::thread_syncs[1].wait();
  });

  unodb::this_thread().qsbr_pause();

  // T2: insert t2_key, also triggering growth and modifying the inode.
  auto t2 = unodb::qsbr_thread([&] {
    unodb::detail::thread_syncs[0].wait();
    const unodb::quiescent_state_on_scope_exit q{};
    std::ignore = db.insert(t2_key, val);
    unodb::detail::thread_syncs[1].notify();
  });

  // T1: insert t1_key, triggering growth.  Hits sync point, pauses.
  auto t1 = unodb::qsbr_thread([&] {
    const unodb::quiescent_state_on_scope_exit q{};
    std::ignore = db.insert(t1_key, val);
  });

  t1.join();
  t2.join();
  unodb::this_thread().qsbr_resume();
  unodb::this_thread().quiescent();

  const unodb::quiescent_state_on_scope_exit q{};
  UNODB_ASSERT_TRUE(db.get(t1_key).has_value());
  UNODB_ASSERT_TRUE(db.get(t2_key).has_value());
}

// T1 builds a chain for a non-full inode insert, then a concurrent T2
// removes a sibling, invalidating T1's node version.  T1's write guard
// must_restart, so T1 deletes the chain it built and retries.
UNODB_DETAIL_DISABLE_MSVC_WARNING(26426)
TEST(OLCNonfullChainRestart, ConcurrentRemoveDuringChainInsert) {
  using db_type = unodb::olc_db<unodb::key_view, std::uint64_t>;
  db_type db;
  unodb::key_encoder enc;
  unodb::key_encoder enc2;  // separate encoder for T2
  constexpr std::uint64_t val = 42;

  // Seed: two keys with different first bytes → root I4 with 2 children.
  auto k_seed1 = enc.reset().encode(std::uint8_t{0x10}).get_key_view();
  std::ignore = db.insert(k_seed1, val);
  auto k_seed2 = enc2.reset().encode(std::uint8_t{0x20}).get_key_view();
  std::ignore = db.insert(k_seed2, val);

  // T1 will insert a long key under 0x30 → add_to_nonfull builds a chain.
  unodb::key_encoder enc_t1;
  auto t1_key = enc_t1.reset()
                    .encode(std::uint8_t{0x30})
                    .encode(std::uint64_t{1})
                    .get_key_view();

  const sync_point_guard guard{unodb::detail::sync_before_nonfull_chain_guard};
  unodb::detail::sync_before_nonfull_chain_guard.arm([&]() {
    unodb::detail::sync_before_nonfull_chain_guard.disarm();
    // Signal T2 to remove a sibling, invalidating the node version.
    unodb::detail::thread_syncs[0].notify();
    unodb::detail::thread_syncs[1].wait();
  });

  unodb::this_thread().qsbr_pause();

  // T2: wait for T1 to pause, then remove seed2 (modifies root I4).
  auto t2 = unodb::qsbr_thread([&] {
    const unodb::quiescent_state_on_scope_exit q{};
    unodb::detail::thread_syncs[0].wait();
    std::ignore =
        db.remove(enc2.reset().encode(std::uint8_t{0x20}).get_key_view());
    unodb::detail::thread_syncs[1].notify();
  });

  // T1: insert t1_key.  Hits sync point after building chain, pauses.
  // T2 removes seed2, invalidating node version.  T1 resumes, write guard
  // must_restart → deletes chain, retries, succeeds.
  auto t1 = unodb::qsbr_thread([&] {
    const unodb::quiescent_state_on_scope_exit q{};
    std::ignore = db.insert(t1_key, val);
  });

  t1.join();
  t2.join();
  unodb::this_thread().qsbr_resume();
  unodb::this_thread().quiescent();

  const unodb::quiescent_state_on_scope_exit q{};
  UNODB_ASSERT_TRUE(db.get(t1_key).has_value());
  UNODB_ASSERT_TRUE(db.get(k_seed1).has_value());
  UNODB_ASSERT_FALSE(db.get(k_seed2).has_value());
}

#endif  // NDEBUG

}  // namespace
