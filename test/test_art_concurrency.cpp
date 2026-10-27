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
#include <type_traits>
#include <utility>

#include "art_internal_impl.hpp"
#include "node_type.hpp"
#include "sync.hpp"
#include "sync_point_test_utils.hpp"
#include "thread_sync.hpp"
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
// The namespace ASSERT_VALUE_FOR_KEY expands against.
namespace detail = unodb::test::detail;

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

/// Run T1's remove(\a removed) parked once at \a point while T2 runs \a t2_op.
/// \return Whether T1 removed \a removed, and \a t2_op's result.
template <class Db, class T2Op>
[[nodiscard]] std::pair<bool, bool> remove_parked_during(
    Db& db, unodb::detail::sync_point& point, unodb::key_view removed,
    T2Op t2_op) {
  const sync_point_guard guard{point};
  point.arm([&point]() {
    point.disarm();  // one-shot
    unodb::detail::thread_syncs[0].notify();
    unodb::detail::thread_syncs[1].wait();
  });

  unodb::this_thread().qsbr_pause();

  bool t2_ok = false;
  auto t2 = unodb::qsbr_thread([&] {
    unodb::detail::thread_syncs[0].wait();
    const unodb::quiescent_state_on_scope_exit q{};
    t2_ok = t2_op();
    unodb::detail::thread_syncs[1].notify();
  });

  bool removed_ok = false;
  auto t1 = unodb::qsbr_thread([&] {
    const unodb::quiescent_state_on_scope_exit q{};
    removed_ok = db.remove(removed);
  });

  t1.join();
  t2.join();
  unodb::this_thread().qsbr_resume();
  unodb::this_thread().quiescent();

  return {removed_ok, t2_ok};
}

/// What T2 does with its key in assert_remove_keeps_other_keys().
enum class t2_action : std::uint8_t { insert, remove };

/// Insert \a key_a, then \a t2_key if \a action is t2_action::remove, then
/// \a sibs into a \a Db, each with its own value.  Remove \a key_a with T1
/// parked once at \a point while T2 inserts or removes \a t2_key as \a action
/// says.  Check that every sibling keeps its value, that \a t2_key has its
/// value after an insert and is gone after a remove, and that \a key_a is
/// gone, then run \a post_check on the tree if given.
template <class Db>
void assert_remove_keeps_other_keys(unodb::detail::sync_point& point,
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

  const auto [removed, t2_ok] = remove_parked_during(db, point, key_a, [&] {
    return action == t2_action::insert ? db.insert(t2_key, val(1))
                                       : db.remove(t2_key);
  });
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
/// assert_remove_keeps_other_keys() default-constructs.  The stale walk of each
/// KeepsSibling test ends at the sibling's keyless leaf in the first and at its
/// packed value in the second, which reach the chain cut through different
/// arms.
using OLCKeyViewTypes = ::testing::Types<unodb::test::key_view_olc_db,
                                         unodb::test::key_view_u64val_olc_db>;

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
  assert_remove_keeps_other_keys<TypeParam>(
      unodb::detail::sync_before_remove_root_read, key_a, {}, t2_key);
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
  assert_remove_keeps_other_keys<TypeParam>(
      unodb::detail::sync_before_remove_root_read, key_a, std::array{sib},
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
  assert_remove_keeps_other_keys<TypeParam>(
      unodb::detail::sync_before_remove_child_read, key_a, std::array{sib},
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
  assert_remove_keeps_other_keys<TypeParam>(
      unodb::detail::sync_before_remove_child_read, key_a, sibs, t2_key);
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
// Uses thread_syncs (condition variables) for coordination.
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
// Tree setup uses 26-byte keys (CT1/CT1b/CT3, 3 chain levels), a lone 33-byte
// key (CT1c/CT1d, 3 chain levels) or 34-byte keys (CT2/CT2b/CT4, 4 chain
// levels) to ensure enough chain depth that pg locks a deep chain node, not
// the cut_point_parent.
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
  assert_remove_keeps_other_keys<unodb::test::key_view_olc_db>(
      unodb::detail::sync_after_chain_locked, key_a, std::array{sib}, new_key);
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
  assert_remove_keeps_other_keys<unodb::test::key_view_olc_db>(
      unodb::detail::sync_after_chain_locked, key_a, sibs, t2_key,
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
/// chain[2] and Root-I4 are unlocked.
void cut_34_byte_chain_during_insert(std::uint64_t t2_v1) {
  constexpr auto X = unodb::test_data::chain_key_filler;
  unodb::key_encoder enc;
  std::array<std::byte, 34> buf_a{};
  std::array<std::byte, 34> buf_sib{};
  std::array<std::byte, 34> buf_t2{};
  const auto key_a = copy_key(make_key_34(enc, 0x10, X, 0x01), buf_a);
  const auto sib = copy_key(make_key_34(enc, 0x20, X, 0x01), buf_sib);
  const auto t2_key = copy_key(make_key_34(enc, 0x10, t2_v1, 0x01), buf_t2);
  assert_remove_keeps_other_keys<unodb::test::key_view_olc_db>(
      unodb::detail::sync_between_chain_locks, key_a, std::array{sib}, t2_key);
}

/// CT2: T1 removes A (chain cut).  T2's key sorts after key_a's chain in
/// chain[0], which Step 2 therefore fails to lock at its descent version.
/// T1's Step 3 takes chain[0], now two-child, as the cut-point parent at its
/// new version.
UNODB_TEST(OLCChainCutInterleaved, ConcurrentInsertIntoMidChain) {
  cut_34_byte_chain_during_insert(unodb::test_data::chain_key_filler_alt);
}

/// CT2b: as CT2, but T2's key sorts before key_a's chain in chain[0], so its
/// insert moves chain[1] to slot 1 of chain[0].  T1's Step 3 takes chain[0] as
/// the cut-point parent at its new version and must cut chain[1] at slot 1,
/// not at the descent's slot 0.
UNODB_TEST(OLCChainCutInterleaved, ConcurrentInsertShiftsMidChainCutIndex) {
  // Every byte one below the filler's, so T2's key sorts first in chain[0].
  constexpr auto below_filler = unodb::test_data::chain_key_filler -
                                std::uint64_t{0x0101'0101'0101'0101ULL};
  cut_34_byte_chain_during_insert(below_filler);
}

// CT3: T1 removes A (chain cut).  T2 removes the sibling.
UNODB_TEST(OLCChainCutInterleaved, ConcurrentRemoveOfSibling) {
  using db_type = unodb::olc_db<unodb::key_view, unodb::value_view>;
  constexpr auto val_bytes = std::array<std::byte, 1>{std::byte{0x42}};
  const auto val = unodb::value_view{val_bytes};

  db_type db;
  unodb::key_encoder enc;

  std::array<std::byte, 26> buf_a{};
  std::array<std::byte, 26> buf_b{};
  const auto key_a = copy_key(make_key_26(enc, 0x10, 0x01), buf_a);
  const auto key_b = copy_key(make_key_26(enc, 0x10, 0x02), buf_b);
  // Use a short (1-byte) sibling so that when T2 removes it, the
  // Root-I4 goes to 1 child but prefix merge overflows (0+1+7 > 7),
  // preventing collapse.  Root-I4 stays alive with a new version.
  std::array<std::byte, 1> buf_sib{};
  const auto sib = copy_key(make_short_key(enc, 0x20), buf_sib);

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

  bool sib_removed = false;
  auto t2 = unodb::qsbr_thread([&] {
    unodb::detail::thread_syncs[0].wait();
    const unodb::quiescent_state_on_scope_exit q{};
    sib_removed = db.remove(sib);
    unodb::detail::thread_syncs[1].notify();
  });

  bool key_a_removed = false;
  auto t1 = unodb::qsbr_thread([&] {
    const unodb::quiescent_state_on_scope_exit q{};
    key_a_removed = db.remove(key_a);
  });

  t1.join();
  t2.join();
  unodb::this_thread().qsbr_resume();
  unodb::this_thread().quiescent();

  UNODB_ASSERT_TRUE(key_a_removed);
  UNODB_ASSERT_TRUE(sib_removed);
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
  cut_34_byte_chain_during_insert(chain_key_filler_second_byte_changed);
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
  assert_remove_keeps_other_keys<unodb::test::key_view_olc_db>(
      unodb::detail::sync_between_chain_locks, key_a, {}, t2_key);
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
  assert_remove_keeps_other_keys<unodb::test::key_view_olc_db>(
      unodb::detail::sync_between_chain_locks, key_a, {}, t2_key);
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

  assert_remove_keeps_other_keys<unodb::test::key_view_olc_db>(
      unodb::detail::sync_after_chain_locked, key_a, sibs, t2_key,
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
  if (const auto r = db.get(key_a); r.has_value()) {
    UNODB_ASSERT_TRUE(std::ranges::equal(*r, val2));  // LCOV_EXCL_LINE
  }
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
  UNODB_ASSERT_TRUE(db.get(key_a).has_value());
  UNODB_ASSERT_TRUE(db.get(key_b).has_value());
  UNODB_ASSERT_TRUE(db.get(key_c).has_value());
  UNODB_ASSERT_TRUE(db.get(key_d).has_value());
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
  constexpr std::uint64_t val = 42;

  // Seed: two keys with different first bytes → root I4 with 2 children.
  const auto k_seed1 = make_short_key(enc, 0x10);
  std::ignore = db.insert(k_seed1, val);
  const auto k_seed2 = make_short_key(enc2, 0x20);
  std::ignore = db.insert(k_seed2, val);

  // T1 will insert a long key under 0x30 → add_to_nonfull builds a chain.
  unodb::key_encoder enc_t1;
  const auto t1_key = make_key(enc_t1, 0x30, 1);

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
    std::ignore = db.remove(k_seed2);
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
/// key byte.  Every layout value is derived here, so a call site cannot
/// perturb the invariants the static_asserts below pin.
template <unodb::node_type ExpectedType, bool Forward>
void run_torn_read_scan_test() {
  using db_type = unodb::olc_db<unodb::key_view, std::uint64_t>;
  using inode_type = torn_read_inode_type<ExpectedType>;
  // Zero packs to the nullptr bit pattern in value-in-slot mode, so the I256
  // restart scan can detect the filled slots only through the value bitmask
  // (is_value_in_slot()).  I48 occupancy lives in child_indexes, so its
  // scenarios are value-agnostic.
  constexpr std::uint64_t val = 0;

  // Seeding exactly min_size children makes the root ExpectedType.
  constexpr auto count = static_cast<std::uint8_t>(inode_type::min_size);
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
  // the restart lands on a known key byte.
  constexpr auto fill_lo =
      static_cast<std::uint8_t>(Forward ? 0U : 256U - count);
  constexpr auto expected_byte = static_cast<std::byte>(
      Forward ? fill_lo : static_cast<std::uint8_t>(fill_lo + count - 1));

  db_type db;
  unodb::key_encoder enc;
  std::vector<std::array<std::byte, 9>> bufs(static_cast<std::size_t>(count) *
                                             2);
  std::vector<unodb::key_view> seeded;
  std::vector<unodb::key_view> filled;

  for (std::uint8_t i = 0; i < count; ++i) {
    seeded.push_back(make_chain_key(
        enc, static_cast<std::uint8_t>(torn_read_seed_lo + i), 1, bufs[i]));
    UNODB_ASSERT_TRUE(db.insert(seeded.back(), val));
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

  // Only T1's traversal reaches the instrumented scans -- no write path calls
  // I48/I256 begin()/last() -- so a plain counter suffices and the hook can
  // stay armed to keep counting through the restart scan.  Declared before the
  // guard so the guard disarms the hook before the counter it captures dies.
  unsigned hits = 0;
  const sync_point_guard guard{unodb::detail::sync_in_inode_scan};
  unodb::detail::sync_in_inode_scan.arm([&hits]() {
    // Park exactly once: later hits fall through, so the restart does not park.
    if (++hits != torn_read_park_hit) return;
    unodb::detail::thread_syncs[0].notify();
    unodb::detail::thread_syncs[1].wait();
  });

  unodb::this_thread().qsbr_pause();

  // T2: migrate the occupancy set past the parked reader's cursor.
  auto t2 = unodb::qsbr_thread([&] {
    const unodb::quiescent_state_on_scope_exit q{};
    unodb::detail::thread_syncs[0].wait();
    for (const auto k : filled) std::ignore = db.insert(k, val);
    for (const auto k : seeded) std::ignore = db.remove(k);
    unodb::detail::thread_syncs[1].notify();
  });

  // T1: the traversal that observes every slot empty and must restart.
  bool valid = false;
  auto observed_byte = std::byte{0};
  auto t1 = unodb::qsbr_thread([&] {
    const unodb::quiescent_state_on_scope_exit q{};
    auto it = db.test_only_iterator();
    if constexpr (Forward)
      it.first();
    else
      it.last();
    valid = it.valid();
    if (valid) observed_byte = it.get_key().view()[0];
  });

  t1.join();
  t2.join();
  unodb::this_thread().qsbr_resume();
  unodb::this_thread().quiescent();

  const unodb::quiescent_state_on_scope_exit q{};
  // 256 slots of the parked scan, which found none occupied -- that
  // fall-through is what returns torn_read_result rather than an ordinary
  // child -- plus the single restart scan, which finds an occupied slot on its
  // first iteration.  Pinning the sum exactly also catches an unexpected extra
  // restart and an early exit from the parked scan, both of which a >= bound
  // would silently tolerate.
  UNODB_ASSERT_EQ(256U + 1U, hits);
  UNODB_ASSERT_TRUE(valid);
  UNODB_ASSERT_EQ(observed_byte, expected_byte);
  for (const auto k : filled) UNODB_ASSERT_TRUE(db.get(k).has_value());
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

#endif  // NDEBUG

}  // namespace

UNODB_DETAIL_RESTORE_MSVC_WARNINGS()
