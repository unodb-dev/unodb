// Copyright (C) 2019-2025 UnoDB contributors
#ifndef UNODB_DETAIL_QSBR_HPP
#define UNODB_DETAIL_QSBR_HPP

/// \file
/// Quiescent State-based reclamation.
///
/// \ingroup qsbr
///
/// Quiescent state-based reclamation (QSBR) memory reclamation scheme. Instead
/// of freeing memory directly, threads register pending deallocation requests
/// to be executed later. Further, each thread notifies when it's not holding
/// any pointers to the shared data structure (is quiescent with respect to that
/// structure). All the threads having passed through a quiescent state
/// constitute a quiescent period, and an epoch change happens at its boundary.
/// At that point all the pending deallocation requests queued before the start
/// of the just-finished quiescent period can be safely executed.
///
/// For a usage example, see example/example_olc_art.cpp.
///
/// The implementation borrows some of the basic ideas from
/// https://preshing.com/20160726/using-quiescent-states-to-reclaim-memory/

// Should be the first include
#include "global.hpp"

// IWYU pragma: no_include <__ostream/basic_ostream.h>
// IWYU pragma: no_include <__vector/vector.h>
// IWYU pragma: no_include <boost/fusion/algorithm/iteration/for_each.hpp>
// IWYU pragma: no_include <boost/fusion/algorithm/query/find_if.hpp>
// IWYU pragma: no_include <boost/fusion/iterator/next.hpp>
// IWYU pragma: no_include <boost/fusion/sequence/intrinsic/begin.hpp>
// IWYU pragma: no_include <boost/fusion/sequence/intrinsic/end.hpp>
// IWYU pragma: no_include <boost/fusion/iterator/deref.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <memory>
#include <system_error>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>  // IWYU pragma: keep

#ifndef NDEBUG
#include <functional>
#include <optional>
#include <unordered_set>
#endif

#ifdef UNODB_DETAIL_WITH_STATS

#include <mutex>

#include <boost/accumulators/accumulators_fwd.hpp>  // IWYU pragma: keep
#include <boost/accumulators/framework/accumulator_set.hpp>
#include <boost/accumulators/framework/extractor.hpp>
#include <boost/accumulators/statistics/max.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/variance.hpp>

#endif  // UNODB_DETAIL_WITH_STATS

#include "assert.hpp"
#include "heap.hpp"
#include "portability_arch.hpp"

namespace unodb {

namespace detail {

class qsbr_ptr_base;

}  // namespace detail

/// Two-bit wrapping-around QSBR epoch counter. Two epochs can be compared for
/// equality but otherwise are unordered. One bit counter would be enough too,
/// but with two bits we can check more invariants.
class [[nodiscard]] qsbr_epoch final {
 public:
  /// Underlying integer type for the epoch.
  using epoch_type = std::uint8_t;

  /// Maximum epoch value.
  static constexpr epoch_type max = 3U;

  /// Copy-construct the epoch counter.
  qsbr_epoch(const qsbr_epoch&) noexcept = default;

  /// Move-construct the epoch counter.
  qsbr_epoch(qsbr_epoch&&) noexcept = default;

  /// Copy-assign the epoch counter.
  qsbr_epoch& operator=(const qsbr_epoch&) noexcept = default;

  /// Move-assign the epoch counter.
  qsbr_epoch& operator=(qsbr_epoch&&) noexcept = default;

  /// Trivially destruct the epoch counter.
  ~qsbr_epoch() noexcept = default;

  /// Construct an epoch with a specific value \a epoch_val_.
  constexpr explicit qsbr_epoch(epoch_type epoch_val_) : epoch_val{epoch_val_} {
#ifndef NDEBUG
    assert_invariant();
#endif
  }

  /// Return a new epoch advanced by \a by, wrapping around, if necessary.
  [[nodiscard, gnu::pure]] constexpr qsbr_epoch advance(
      unsigned by = 1) const noexcept {
#ifndef NDEBUG
    assert_invariant();
#endif

    return qsbr_epoch{static_cast<epoch_type>((epoch_val + by) % max_count)};
  }

  /// Return the raw epoch value.
  [[nodiscard, gnu::pure]] constexpr epoch_type get_val() const noexcept {
    return epoch_val;
  }

  /// Compare equal to \a other.
  [[nodiscard, gnu::pure]] constexpr bool operator==(
      qsbr_epoch other) const noexcept {
#ifndef NDEBUG
    assert_invariant();
    other.assert_invariant();
#endif

    return epoch_val == other.epoch_val;
  }

  /// Output the epoch to \a os. For debug purposes only.
  [[gnu::cold]] UNODB_DETAIL_NOINLINE void dump(std::ostream& os) const;

  /// Prohibit constructing defaulted epoch values.
  qsbr_epoch() noexcept = delete;

 private:
  /// Number of possible epoch values.
  static constexpr auto max_count = max + 1U;

  static_assert((max_count & (max_count - 1U)) == 0,
                "The number of possible epoch values must be a power of 2");

  /// Epoch value.
  epoch_type epoch_val;

#ifndef NDEBUG
  /// Assert that the epoch value is valid.
  constexpr void assert_invariant() const noexcept {
    UNODB_DETAIL_ASSERT(epoch_val <= max);
  }
#endif
};

// LCOV_EXCL_START
/// Print the epoch \a value to the output stream \a os. For debug purposes
/// only.
[[gnu::cold]] inline std::ostream& operator<<(std::ostream& os
                                              UNODB_DETAIL_LIFETIMEBOUND,
                                              qsbr_epoch value) {
  value.dump(os);
  return os;
}
// LCOV_EXCL_STOP

/// Type for counting QSBR-managed threads.
using qsbr_thread_count_type = std::uint32_t;

/// Maximum number of threads that can participate in QSBR simultaneously. Set
/// to 2^29-1, chosen to allow two thread count fields and an epoch counter to
/// fit in a single machine word. It should be enough for everybody, and its
/// overflow is not checked even in Release builds.
inline constexpr qsbr_thread_count_type max_qsbr_threads = (2UL << 29U) - 1U;

/// Global QSBR state in a single 64-bit machine word.
///
/// Bit allocation:
/// - 0..29: Number of threads in the previous epoch
/// - 30..31: Unused
/// - 32..61: Total number of threads
/// - 62..63: Wrapping-around epoch counter
///
/// Special state transitions:
/// If a thread decrements the number of threads in the previous epoch and
/// observes zero while the total number of threads is greater than zero, then
/// this thread is responsible for the epoch change. The decrement of the last
/// thread in the previous epoch and the epoch bump may happen in a single step,
/// in which case nobody will observe zero threads in the previous epoch.
// TODO(laurynas): move to detail namespace
struct qsbr_state {
  /// Underlying type for the state word.
  using type = std::uint64_t;

 private:
  /// Extract the epoch from \a word.
  [[nodiscard, gnu::const]] static constexpr qsbr_epoch do_get_epoch(
      type word) noexcept {
    return qsbr_epoch{
        static_cast<qsbr_epoch::epoch_type>(word >> epoch_in_word_offset)};
  }

  /// Extract the total thread count from \a word.
  [[nodiscard, gnu::const]] static constexpr qsbr_thread_count_type
  do_get_thread_count(type word) noexcept {
    const auto result = static_cast<qsbr_thread_count_type>(
        (word & thread_count_in_word_mask) >> thread_count_in_word_offset);
    UNODB_DETAIL_ASSERT(result <= max_qsbr_threads);
    return result;
  }

  /// Extract the count of threads in the previous epoch from \a word.
  [[nodiscard, gnu::const]] static constexpr qsbr_thread_count_type
  do_get_threads_in_previous_epoch(type word) noexcept {
    const auto result = static_cast<qsbr_thread_count_type>(
        word & threads_in_previous_epoch_in_word_mask);
    UNODB_DETAIL_ASSERT(result <= max_qsbr_threads);
    return result;
  }

 public:
  /// Get the epoch from \a word.
  [[nodiscard, gnu::const]] static constexpr qsbr_epoch get_epoch(
      type word) noexcept {
    assert_invariants(word);
    return do_get_epoch(word);
  }

  /// Get the total thread count from \a word.
  [[nodiscard, gnu::const]] static constexpr qsbr_thread_count_type
  get_thread_count(type word) noexcept {
    assert_invariants(word);
    return do_get_thread_count(word);
  }

  /// Get the count of threads in the previous epoch from \a word.
  [[nodiscard, gnu::const]] static constexpr qsbr_thread_count_type
  get_threads_in_previous_epoch(type word) noexcept {
    assert_invariants(word);
    return do_get_threads_in_previous_epoch(word);
  }

  /// Check if the count of threads in if \a word is 0 or 1.
  [[nodiscard, gnu::const]] static constexpr bool single_thread_mode(
      type word) noexcept {
    return get_thread_count(word) < 2;
  }

  /// Output QSBR state in \a word to \a os, for debugging only.
  [[gnu::cold]] UNODB_DETAIL_NOINLINE static void dump(std::ostream& os,
                                                       type word);

 private:
  friend class qsbr;

  /// Make a state word from \a epoch.
  [[nodiscard, gnu::const]] static constexpr type make_from_epoch(
      qsbr_epoch epoch) noexcept {
    const auto result = static_cast<type>(epoch.get_val())
                        << epoch_in_word_offset;
    assert_invariants(result);
    return result;
  }

  /// Increment the thread count in \a word by one.
  [[nodiscard, gnu::const]] static constexpr type inc_thread_count(
      type word) noexcept {
    assert_invariants(word);

    const auto result = word + one_thread_in_count;

    assert_invariants(result);
    UNODB_DETAIL_ASSERT(get_epoch(word) == get_epoch(result));
    UNODB_DETAIL_ASSERT(get_threads_in_previous_epoch(word) ==
                        get_threads_in_previous_epoch(result));
    UNODB_DETAIL_ASSERT(get_thread_count(word) + 1 == get_thread_count(result));

    return result;
  }

  /// Decrement the thread count in \a word by one.
  [[nodiscard, gnu::const]] static constexpr type dec_thread_count(
      type word) noexcept {
    assert_invariants(word);
    UNODB_DETAIL_ASSERT(get_thread_count(word) > 0);

    const auto result = word - one_thread_in_count;

    assert_invariants(result);
    UNODB_DETAIL_ASSERT(get_epoch(word) == get_epoch(result));
    UNODB_DETAIL_ASSERT(get_threads_in_previous_epoch(word) ==
                        get_threads_in_previous_epoch(result));
    UNODB_DETAIL_ASSERT(get_thread_count(word) - 1 == get_thread_count(result));

    return result;
  }

  /// Increment both the thread count and threads in previous epoch in \a word.
  [[nodiscard, gnu::const]] static constexpr type
  inc_thread_count_and_threads_in_previous_epoch(type word) noexcept {
    assert_invariants(word);

    const auto result = word + one_thread_and_one_in_previous;

    assert_invariants(result);
    UNODB_DETAIL_ASSERT(get_epoch(word) == get_epoch(result));
    UNODB_DETAIL_ASSERT(get_threads_in_previous_epoch(word) + 1 ==
                        get_threads_in_previous_epoch(result));
    UNODB_DETAIL_ASSERT(get_thread_count(word) + 1 == get_thread_count(result));

    return result;
  }

  /// Decrement both the thread count and threads in previous epoch in \a word.
  [[nodiscard, gnu::const]] static constexpr type
  dec_thread_count_and_threads_in_previous_epoch(type word) noexcept {
    assert_invariants(word);
    UNODB_DETAIL_ASSERT(get_thread_count(word) > 0);
    UNODB_DETAIL_ASSERT(get_threads_in_previous_epoch(word) > 0);

    const auto result = word - one_thread_and_one_in_previous;

    assert_invariants(result);
    UNODB_DETAIL_ASSERT(get_epoch(word) == get_epoch(result));
    UNODB_DETAIL_ASSERT(get_threads_in_previous_epoch(word) - 1 ==
                        get_threads_in_previous_epoch(result));
    UNODB_DETAIL_ASSERT(get_thread_count(word) - 1 == get_thread_count(result));

    return result;
  }

  /// Increment the epoch and reset the threads in previous epoch count to the
  /// total thread count.
  [[nodiscard, gnu::const]] static constexpr type inc_epoch_reset_previous(
      type word) noexcept {
    assert_invariants(word);
    UNODB_DETAIL_ASSERT(get_threads_in_previous_epoch(word) == 0);

    const auto old_epoch = get_epoch(word);
    const auto new_epoch_in_word = make_from_epoch(old_epoch.advance());
    const auto new_thread_count_in_word = word & thread_count_in_word_mask;
    const auto new_threads_in_previous = (word >> thread_count_in_word_offset) &
                                         threads_in_previous_epoch_in_word_mask;
    const auto result =
        new_epoch_in_word | new_thread_count_in_word | new_threads_in_previous;

    UNODB_DETAIL_ASSERT(get_epoch(result) == old_epoch.advance());
    UNODB_DETAIL_ASSERT(get_thread_count(result) == get_thread_count(word));
    UNODB_DETAIL_ASSERT(get_threads_in_previous_epoch(result) ==
                        get_thread_count(result));
    assert_invariants(result);

    return result;
  }

  /// Increment the epoch, decrement the thread count, and reset the threads in
  /// previous epoch to the new thread count.
  [[nodiscard, gnu::const]] static constexpr type
  inc_epoch_dec_thread_count_reset_previous(type word) noexcept {
    assert_invariants(word);
    const auto old_thread_count = get_thread_count(word);
    UNODB_DETAIL_ASSERT(old_thread_count > 0);
    UNODB_DETAIL_ASSERT(get_threads_in_previous_epoch(word) == 1);

    const auto new_word_with_epoch = make_from_epoch(get_epoch(word).advance());
    const auto new_thread_count = old_thread_count - 1;
    const auto new_word_with_thread_count = static_cast<type>(new_thread_count)
                                            << thread_count_in_word_offset;
    const auto new_threads_in_previous = new_thread_count;
    const auto result = new_word_with_epoch | new_word_with_thread_count |
                        new_threads_in_previous;

    UNODB_DETAIL_ASSERT(get_epoch(word).advance() == get_epoch(result));
    UNODB_DETAIL_ASSERT(get_thread_count(word) - 1 == get_thread_count(result));
    UNODB_DETAIL_ASSERT(get_threads_in_previous_epoch(result) ==
                        get_thread_count(result));
    assert_invariants(result);

    return result;
  }

  /// Decrement thread counts in \a word while optionally advancing the epoch if
  /// \a advance_epoch is set.
  [[nodiscard, gnu::const]] static constexpr type
  dec_thread_count_threads_in_previous_epoch_maybe_advance(
      type word, bool advance_epoch) noexcept {
    return UNODB_DETAIL_UNLIKELY(advance_epoch)
               ? inc_epoch_dec_thread_count_reset_previous(word)
               : dec_thread_count_and_threads_in_previous_epoch(word);
  }

  /// Atomically decrement the number of threads in the previous epoch.
  /// \param word atomic QSBR state word
  /// \return old word value
  [[nodiscard]] static qsbr_state::type
  atomic_fetch_dec_threads_in_previous_epoch(std::atomic<type>& word) noexcept;

  /// Assert that all invariants hold for \a word.
  static constexpr void assert_invariants(type word
                                          UNODB_DETAIL_USED_IN_DEBUG) noexcept {
#ifndef NDEBUG
    const auto thread_count = do_get_thread_count(word);
    UNODB_DETAIL_ASSERT(thread_count <= max_qsbr_threads);
    const auto threads_in_previous = do_get_threads_in_previous_epoch(word);
    UNODB_DETAIL_ASSERT(threads_in_previous <= thread_count);
#endif
  }

  /// Mask for the thread count field.
  static constexpr auto thread_count_mask = max_qsbr_threads;

  static_assert((thread_count_mask & (thread_count_mask + 1)) == 0,
                "Thread count field mask should be 2^n-1");

  /// Mask for the threads in previous epoch field.
  /// \hideinitializer
  static constexpr auto threads_in_previous_epoch_in_word_mask =
      static_cast<std::uint64_t>(thread_count_mask);

  /// Bit offset for the total thread count field.
  static constexpr auto thread_count_in_word_offset = 32U;

  /// Mask for the total thread count field.
  /// \hideinitializer
  static constexpr auto thread_count_in_word_mask =
      static_cast<std::uint64_t>(thread_count_mask)
      << thread_count_in_word_offset;

  /// Bit offset for the epoch field.
  static constexpr auto epoch_in_word_offset = 62U;

  /// Value to use for incrementing or decrementing the thread count by one.
  /// \hideinitializer
  static constexpr auto one_thread_in_count = 1ULL
                                              << thread_count_in_word_offset;

  /// Value to use for incrementing or decrementing both thread count and
  /// threads in previous epoch by one.
  /// \hideinitializer
  static constexpr auto one_thread_and_one_in_previous =
      one_thread_in_count | 1U;
};

namespace detail {

/// Pending deallocation request for QSBR-managed memory.
class [[nodiscard]] deallocation_request final {
 public:
#ifndef NDEBUG
  /// Debug build only: function type for callbacks executed during
  /// deallocation.
  using debug_callback = std::function<void(const void*)>;
#endif

  /// Create a new deallocation request for \a pointer_. In debug builds also
  /// pass \a request_epoch_ and \a dealloc_callback_ to be executed during
  /// deallocation.
  explicit deallocation_request(void* pointer_ UNODB_DETAIL_LIFETIMEBOUND
#ifndef NDEBUG
                                ,
                                qsbr_epoch request_epoch_,
                                debug_callback dealloc_callback_
#endif
                                ) noexcept
      : pointer{pointer_}
#ifndef NDEBUG
        ,
        dealloc_callback{std::move(dealloc_callback_)},
        request_epoch{request_epoch_}
#endif
  {
#ifndef NDEBUG
    instance_count.fetch_add(1, std::memory_order_relaxed);
#endif
  }

  /// Move constructor.
  deallocation_request(deallocation_request&&) noexcept = default;

  /// Destructor.
  ~deallocation_request() = default;

  /// Do the deallocation. All parameters are debug build-only.
  ///
#ifndef NDEBUG
  /// \param orphan Whether this request belongs to a terminated thread
  /// \param dealloc_epoch Optional epoch when deallocation happens
  /// \param dealloc_epoch_single_thread_mode Optional flag whether the
  /// deallocation epoch has total zero or one thread only
#endif
  void deallocate(
#ifndef NDEBUG
      bool orphan, std::optional<qsbr_epoch> dealloc_epoch,
      std::optional<bool> dealloc_epoch_single_thread_mode
#endif
  ) const noexcept;

  /// Copy construction is disabled to prevent redundant instances.
  deallocation_request(const deallocation_request&) = delete;

  /// Copy assignment is disabled.
  deallocation_request& operator=(const deallocation_request&) = delete;

  /// Move assignment is disabled.
  deallocation_request& operator=(deallocation_request&&) = delete;

#ifndef NDEBUG
  /// Assert that no instances of this class exist, indicating unhandled
  /// requests.
  static void assert_zero_instances() noexcept {
    UNODB_DETAIL_ASSERT(instance_count.load(std::memory_order_relaxed) == 0);
  }
#endif

 private:
  /// Pointer to memory that should be deallocated.
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  void* const pointer;

#ifndef NDEBUG
  /// Debug build only: callback to execute during deallocation.
  // Non-const to support move
  debug_callback dealloc_callback;

  /// Debug build only: epoch when this deallocation request was created.
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  const qsbr_epoch request_epoch;

  /// Debug build only: global count of active deallocation request instances.
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
  static std::atomic<std::uint64_t> instance_count;
#endif
};

/// Vector of deallocation requests.
using dealloc_request_vector = std::vector<deallocation_request>;

/// A scope guard that executes deallocation requests upon destruction.
class [[nodiscard]] deferred_requests final {
 public:
  /// Create a deferred requests object that takes ownership of \a requests_.
  /// In debug builds, also stores metadata about the requests.
  ///
#ifndef NDEBUG
  /// \param orphaned_requests_ Whether these requests belongs to a terminated
  ///                           thread
  /// \param request_epoch_ The epoch when the requests were created, if known
  /// \param dealloc_epoch_single_thread_mode_ Optional flag whether the
  ///                                          deallocation epoch has total zero
  ///                                          or one thread only
#endif
  UNODB_DETAIL_RELEASE_EXPLICIT deferred_requests(
      dealloc_request_vector&& requests_
#ifndef NDEBUG
      ,
      bool orphaned_requests_, std::optional<qsbr_epoch> request_epoch_,
      std::optional<bool> dealloc_epoch_single_thread_mode_
#endif
      ) noexcept
      : requests{std::move(requests_)}
#ifndef NDEBUG
        ,
        orphaned_requests{orphaned_requests_},
        dealloc_epoch{request_epoch_},
        dealloc_epoch_single_thread_mode{dealloc_epoch_single_thread_mode_}
#endif
  {
  }

  /// Copy construction is disabled to prevent redundant instances.
  deferred_requests(const deferred_requests&) noexcept = delete;

  /// Move construction is disabled, instances are created in-place only.
  deferred_requests(deferred_requests&&) noexcept = delete;

  /// Copy assignment is disabled.
  deferred_requests& operator=(const deferred_requests&) noexcept = delete;

  /// Move assignment is disabled.
  deferred_requests& operator=(deferred_requests&&) noexcept = delete;

  /// Destruct this object and execute all deallocation requests.
  ~deferred_requests() noexcept {
    for (const auto& dealloc_request : requests) {
      dealloc_request.deallocate(
#ifndef NDEBUG
          orphaned_requests, dealloc_epoch, dealloc_epoch_single_thread_mode
#endif
      );
    }
  }

  /// Default construction is disabled.
  deferred_requests() = delete;

 private:
  /// The vector of deallocation requests to be executed.
  const dealloc_request_vector requests;

#ifndef NDEBUG
  /// Whether these requests belong to a terminated thread.
  const bool orphaned_requests;

  /// The epoch when the requests should be deallocated, if known.
  const std::optional<qsbr_epoch> dealloc_epoch;

  /// Whether the deallocation epoch has only 0 or 1 threads, if known.
  const std::optional<bool> dealloc_epoch_single_thread_mode;
#endif
};

UNODB_DETAIL_DISABLE_MSVC_WARNING(26495)
/// Node in a linked list for orphaned (issued by threads that have quit since)
/// deallocation requests.
// TODO(laurynas): rename? orphaned_dealloc_request_list_node?
struct dealloc_vector_list_node {
  /// Orphaned deallocation requests.
  detail::dealloc_request_vector requests;
  /// Next linked list node.
  dealloc_vector_list_node* next;
};
UNODB_DETAIL_RESTORE_MSVC_WARNINGS()

struct set_qsbr_per_thread_in_main_thread;

#ifndef UNODB_DETAIL_MSVC_CLANG

/// Register a function \a fn of type \a Func to be called on thread exit.
template <typename Func>
void on_thread_exit(Func fn) noexcept {
  /// Helper class that executes a function on destruction.
  struct thread_exiter {
    /// Construct with the function to execute on destruction.
    explicit thread_exiter(Func func_ UNODB_DETAIL_LIFETIMEBOUND) noexcept
        : func{func_} {}

    /// Destruct and execute the function on destruction.
    ~thread_exiter() noexcept { func(); }

    /// Copy construction is disabled.
    thread_exiter(const thread_exiter&) = delete;

    /// Move construction is disabled.
    thread_exiter(thread_exiter&&) = delete;

    /// Copy assignment is disabled.
    thread_exiter& operator=(const thread_exiter&) = delete;

    /// Move assignment is disabled.
    thread_exiter& operator=(thread_exiter&&) = delete;

   private:
    /// The function to execute on destruction.
    Func func;
  };

  /// Thread-local instance of the exiter that will run fn on thread exit.
  const thread_local static thread_exiter exiter{fn};
}

#endif  // #ifndef UNODB_DETAIL_MSVC_CLANG

}  // namespace detail

/// Thread-local QSBR data structure.
///
/// Maintains pending deallocation requests for the previous and current epoch,
/// tracks the last seen global epoch by any operation, and the last seen global
/// epoch by a quiescent state for this thread.
///
/// Also owns the preallocated list nodes for orphaning pending previous and
/// current interval deallocation requests on thread exit. Having them
/// preallocated avoids heap allocations on thread exit code path, where
/// handling any allocation failures is hard.
class [[nodiscard]] qsbr_per_thread final {
 public:
  /// Construct a new thread-local QSBR structure and register with global QSBR.
  qsbr_per_thread();

  /// Unregister the current thread from global QSBR on thread exist.
  UNODB_DETAIL_DISABLE_MSVC_WARNING(26447)
  ~qsbr_per_thread() noexcept {
    if (!is_qsbr_paused()) {
#ifdef UNODB_DETAIL_WITH_STATS
      // TODO(laurynas): to avoid try/catch below:
      // - replace std::mutex with noexcept synchronization, realistically only
      // spinlock fits, which might not be good enough;
      // - replace Boost.Accumulator with own noexcept stats.
      try {
        qsbr_pause();
      }
      // The QSBR destructor can only throw std::system_error from the stats
      // mutex lock. Eat this exception, and eat any other unexpected exceptions
      // too, except for the debug build.
      // LCOV_EXCL_START
      catch (const std::system_error& e) {
        std::cerr << "Failed to register QSBR stats for the quitting thread: "
                  << e.what() << '\n';
      } catch (const std::exception& e) {
        std::cerr << "Unknown exception in the QSBR thread destructor: "
                  << e.what() << '\n';
        UNODB_DETAIL_DEBUG_CRASH();
      } catch (...) {
        std::cerr << "Unknown exception in the QSBR thread destructor";
        UNODB_DETAIL_DEBUG_CRASH();
      }
#else
      qsbr_pause();
#endif
      // LCOV_EXCL_STOP
    }
  }
  UNODB_DETAIL_RESTORE_MSVC_WARNINGS()

  /// Request deallocation of \a pointer when it is safe to do so.
  ///
  /// \param pointer Memory to deallocate
#ifdef UNODB_DETAIL_WITH_STATS
  /// \param size Size of the memory, for statistics
#endif
#ifndef NDEBUG
  /// \param dealloc_callback Debug callback to execute during deallocation
#endif
  void on_next_epoch_deallocate(
      void* pointer
#ifdef UNODB_DETAIL_WITH_STATS
      ,
      std::size_t size
#endif
#ifndef NDEBUG
      ,
      detail::deallocation_request::debug_callback dealloc_callback
#endif
  );

  /// Signal that this thread is quiescent.
  /// \pre No active pointers to QSBR-managed data must be held.
  void quiescent()
#ifndef UNODB_DETAIL_WITH_STATS
      noexcept
#endif
      ;

  /// Pause QSBR for this thread, unregistering from global QSBR.
  /// \pre No active pointers to QSBR-managed data must be held.
  /// \pre QSBR must not be paused for this thread.
  void qsbr_pause()
#ifndef UNODB_DETAIL_WITH_STATS
      noexcept
#endif
      ;

  /// Resume QSBR for this thread, registering with global QSBR.
  /// \pre QSBR must be paused for this thread.
  void qsbr_resume();

  /// Check if QSBR is paused for this thread.
  [[nodiscard, gnu::pure]] bool is_qsbr_paused() const noexcept {
    return paused;
  }

  /// Check if there are no deallocation requests for the previous epoch.
  [[nodiscard]] bool previous_interval_requests_empty() const noexcept {
    return previous_interval_dealloc_requests.empty();
  }

  /// Check if there are no deallocation requests for the current epoch.
  [[nodiscard]] bool current_interval_requests_empty() const noexcept {
    return current_interval_dealloc_requests.empty();
  }

#ifdef UNODB_DETAIL_WITH_STATS

  /// Get the total size of memory submitted for deallocation in the current
  /// epoch.
  [[nodiscard, gnu::pure]] std::size_t get_current_interval_total_dealloc_size()
      const noexcept {
    return current_interval_total_dealloc_size;
  }

#endif  // UNODB_DETAIL_WITH_STATS

  /// Copy construction is disabled for a per-thread singleton.
  qsbr_per_thread(const qsbr_per_thread&) = delete;

  /// Move construction is disabled for a per-thread singleton.
  qsbr_per_thread(qsbr_per_thread&&) = delete;

  /// Copy assignment is disabled for a per-thread singleton.
  qsbr_per_thread& operator=(const qsbr_per_thread&) = delete;

  /// Move assignment is disabled for a per-thread singleton.
  qsbr_per_thread& operator=(qsbr_per_thread&&) = delete;

 private:
  friend class detail::qsbr_ptr_base;
  friend class qsbr;
  friend class qsbr_thread;
  friend qsbr_per_thread& this_thread() noexcept;
  friend struct detail::set_qsbr_per_thread_in_main_thread;

#ifndef NDEBUG
  /// Register an active pointer \a ptr to QSBR-managed data in this thread.
  ///
  /// \note Do not call directly: use unodb::qsbr_ptr instead.
  void register_active_ptr(const void* ptr);

  /// Unregister an active pointer \a ptr to QSBR-managed data in this thread.
  ///
  /// \note Do not call directly: use unodb::qsbr_ptr instead.
  void unregister_active_ptr(const void* ptr);
#endif

  /// Get the thread-local QSBR data structure instance.
  [[nodiscard]] static qsbr_per_thread& get_instance() noexcept {
    return *current_thread_instance;
  }

  /// Make \a new_instance to be the thread-local QSBR data structure instance.
  static void set_instance(
      std::unique_ptr<qsbr_per_thread> new_instance) noexcept {
    current_thread_instance = std::move(new_instance);
#ifndef UNODB_DETAIL_MSVC_CLANG
    // Force qsbr_per_thread destructor to run on thread exit. It already runs
    // without this kludge on most configurations, except with gcc --coverage on
    // macOS, and it seems it is not guaranteed -
    // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=61991
    detail::on_thread_exit([]() noexcept { current_thread_instance.reset(); });
#endif
  }

  /// The thread-local QSBR instance.
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
  thread_local static std::unique_ptr<qsbr_per_thread> current_thread_instance;

  /// Preallocated list node for orphaning pending previous epoch requests on
  /// thread exit.
  /// \hideinitializer
  std::unique_ptr<detail::dealloc_vector_list_node>
      previous_interval_orphan_list_node{
          std::make_unique<detail::dealloc_vector_list_node>()};

  /// Preallocated list node for orphaning pending current epoch requests on
  /// thread exit.
  /// \hideinitializer
  std::unique_ptr<detail::dealloc_vector_list_node>
      current_interval_orphan_list_node{
          std::make_unique<detail::dealloc_vector_list_node>()};

  /// Number of quiescent states since the last epoch change.
  std::uint64_t quiescent_states_since_epoch_change{0};

  /// Last seen global epoch by quiescent state for this thread.
  qsbr_epoch last_seen_quiescent_state_epoch;

  /// Last seen global epoch by any operation for this thread.
  qsbr_epoch last_seen_epoch;

  /// Pending deallocation requests for the previous epoch.
  detail::dealloc_request_vector previous_interval_dealloc_requests;

  /// Pending deallocation requests for the current epoch.
  detail::dealloc_request_vector current_interval_dealloc_requests;

  /// Whether QSBR participation is currently paused for this thread.
  bool paused{false};

#ifdef UNODB_DETAIL_WITH_STATS
  /// Total size of memory that was submitted for deallocation in the current
  /// epoch.
  std::size_t current_interval_total_dealloc_size{0};
#endif  // UNODB_DETAIL_WITH_STATS

  /// Update this thread's view of the current epoch and execute the previous
  /// interval deallocation requests if a newer epoch is seen.
  ///
  /// \param single_thread_mode Whether QSBR is in single-thread mode
  /// \param new_seen_epoch The new epoch to advance to
  /// \param new_current_requests Deallocation requests for the current epoch
  void advance_last_seen_epoch(
      bool single_thread_mode, qsbr_epoch new_seen_epoch,
      detail::dealloc_request_vector new_current_requests = {})
#ifndef UNODB_DETAIL_WITH_STATS
      noexcept
#endif
      ;

  /// Execute the previous interval deallocation requests and move the
  /// current interval ones to the previous interval.
  ///
  /// \param single_thread_mode Whether QSBR is in single-thread mode
  /// \param dealloc_epoch Deallocation epoch
  /// \param new_current_requests New deallocation requests for the current
  ///                             epoch
  void execute_previous_requests(
      bool single_thread_mode, qsbr_epoch dealloc_epoch,
      detail::dealloc_request_vector new_current_requests = {})
#ifndef UNODB_DETAIL_WITH_STATS
      noexcept
#endif
      ;

  /// Move any pending deallocation requests to the global QSBR structure when
  /// the thread exits.
  void orphan_pending_requests() noexcept;

#ifndef NDEBUG
  /// Set of active pointers to QSBR-managed data held by this thread.
  std::unordered_multiset<const void*> active_ptrs;
#endif
};

/// Get the thread-local QSBR instance.
[[nodiscard]] inline qsbr_per_thread& this_thread() noexcept {
  return qsbr_per_thread::get_instance();
}

#ifdef UNODB_DETAIL_WITH_STATS

namespace boost_acc = boost::accumulators;

#endif  // UNODB_DETAIL_WITH_STATS

// If C++ standardisation proposal by A. D. Robison "Policy-based design for
// safe destruction in concurrent containers" ever gets anywhere, consider
// changing to its interface, like Stamp-it paper does.

/// Global QSBR state manager for memory reclamation.
///
/// Tracks threads participating in QSBR, manages epoch transitions, and handles
/// orphaned deallocation requests from terminated threads.
class qsbr final {
 public:
  /// Get the global QSBR singleton instance.
  [[nodiscard]] static qsbr& instance() noexcept {
    static qsbr instance;
    return instance;
  }

  /// Process quiescent state for a thread, potentially triggering an epoch
  /// change.
  /// \param current_global_epoch Current global epoch
#ifndef NDEBUG
  /// \param thread_epoch Current thread epoch (debug builds only)
#endif
  /// \return Potentially updated global epoch
  [[nodiscard]] qsbr_epoch remove_thread_from_previous_epoch(
      qsbr_epoch current_global_epoch
#ifndef NDEBUG
      ,
      qsbr_epoch thread_epoch
#endif
      ) noexcept;

  /// Register a new thread with QSBR.
  /// \return The current global epoch
  [[nodiscard]] qsbr_epoch register_thread() noexcept;

  /// Unregister a thread from QSBR.
  /// \param quiescent_states_since_epoch_change Count of quiescent states by
  ///                                            this thread since last epoch
  ///                                            change
  /// \param thread_epoch Last seen epoch by this thread
  /// \param qsbr_thread The thread's QSBR instance to unregister
  void unregister_thread(std::uint64_t quiescent_states_since_epoch_change,
                         qsbr_epoch thread_epoch, qsbr_per_thread& qsbr_thread)
#ifndef UNODB_DETAIL_WITH_STATS
      noexcept
#endif
      ;

#ifdef UNODB_DETAIL_WITH_STATS
  /// Reset all statistics counters to zero.
  void reset_stats();

  /// Register thread's count of quiescent \a states between thread epoch
  /// changes.
  void register_quiescent_states_per_thread_between_epoch_changes(
      std::uint64_t states) {
    const std::lock_guard guard{quiescent_state_stats_lock};
    quiescent_states_per_thread_between_epoch_change_stats(states);
    publish_quiescent_states_per_thread_between_epoch_change_stats();
  }

  /// Register thread's deallocation request stats between thread epoch changes.
  ///
  /// \param total_size Total size in bytes of deallocation requests
  /// \param count Count of deallocation requests
  void register_dealloc_stats_per_thread_between_epoch_changes(
      std::size_t total_size, std::size_t count) {
    const std::lock_guard guard{dealloc_stats_lock};
    deallocation_size_per_thread_stats(total_size);
    publish_deallocation_size_stats();
    epoch_dealloc_per_thread_count_stats(count);
    publish_epoch_callback_stats();
  }

  /// Get maximum number of threads' deallocation request counts between epoch
  /// changes.
  [[nodiscard]] std::size_t get_epoch_callback_count_max() const noexcept {
    return epoch_dealloc_per_thread_count_max.load(std::memory_order_acquire);
  }

  /// Get variance of threads' deallocation request counts between epoch
  /// changes.
  [[nodiscard]] double get_epoch_callback_count_variance() const noexcept {
    return epoch_dealloc_per_thread_count_variance.load(
        std::memory_order_acquire);
  }

  /// Get mean of threads' deallocation request counts between epoch
  /// changes.
  [[nodiscard]] double
  get_mean_quiescent_states_per_thread_between_epoch_changes() const noexcept {
    return quiescent_states_per_thread_between_epoch_change_mean.load(
        std::memory_order_acquire);
  }

  /// Get total number of epoch changes.
  [[nodiscard]] std::uint64_t get_epoch_change_count() const noexcept {
    return epoch_change_count.load(std::memory_order_acquire);
  }

  /// Get maximum size of memory waiting for deallocation in bytes.
  [[nodiscard]] std::uint64_t get_max_backlog_bytes() const noexcept {
    return deallocation_size_per_thread_max.load(std::memory_order_acquire);
  }

  /// Get mean backlog of memory waiting for deallocation in bytes.
  [[nodiscard]] double get_mean_backlog_bytes() const noexcept {
    return deallocation_size_per_thread_mean.load(std::memory_order_acquire);
  }

#endif  // UNODB_DETAIL_WITH_STATS

  /// Get the current QSBR state word.
  /// \note Made public for tests and asserts, do not call from the user code.
  [[nodiscard]] qsbr_state::type get_state() const noexcept {
    return state.load(std::memory_order_acquire);
  }

  /// Check if there are no orphaned (issued by threads that have quit
  /// since) deallocation requests for the previous epoch.
  [[nodiscard]] bool previous_interval_orphaned_requests_empty()
      const noexcept {
    return orphaned_previous_interval_dealloc_requests.load(
               std::memory_order_acquire) == nullptr;
  }

  /// Check if there are no orphaned (issued by threads that have quit
  /// since) deallocation requests for the current epoch.
  [[nodiscard]] bool current_interval_orphaned_requests_empty() const noexcept {
    return orphaned_current_interval_dealloc_requests.load(
               std::memory_order_acquire) == nullptr;
  }

  /// Assert that QSBR is idle (at most one thread, no orphaned requests).
  void assert_idle() const noexcept {
#ifndef NDEBUG
    const auto current_state = get_state();
    qsbr_state::assert_invariants(current_state);
    UNODB_DETAIL_ASSERT(qsbr_state::get_thread_count(current_state) <= 1);
    // Quitting threads may race with epoch changes by design, resulting in
    // previous epoch orphaned requests not being executed until epoch
    // changes one more time. If that does not happen, this assert may fire at
    // the process exit time. While it is possible to handle this silently, by
    // executing the requests then, this may also result in some memory being
    // held too long. Thus users are advised to pass through Q state in the
    // last thread a couple more times at the end.
    UNODB_DETAIL_ASSERT(previous_interval_orphaned_requests_empty());
    UNODB_DETAIL_ASSERT(current_interval_orphaned_requests_empty());
    detail::deallocation_request::assert_zero_instances();
#endif
  }

  /// Output QSBR state to \a out for debugging.
  [[gnu::cold]] UNODB_DETAIL_NOINLINE void dump(std::ostream& out) const;

  /// Copy construction is disabled for a singleton.
  qsbr(const qsbr&) = delete;

  /// Move construction is disabled for a singleton.
  qsbr(qsbr&&) = delete;

  /// Copy assignment is disabled for a singleton.
  qsbr& operator=(const qsbr&) = delete;

  /// Move assignment is disabled for a singleton.
  qsbr& operator=(qsbr&&) = delete;

 private:
  friend class detail::deallocation_request;
  friend class qsbr_per_thread;

  /// Default constructor for singleton.
  qsbr() noexcept = default;

  /// Destructor. In debug builds asserts that QSBR is idle.
  ~qsbr() noexcept { assert_idle(); }

  /// Free memory at \a pointer using detail::free_aligned.
  ///
  /// In debug builds, call \a debug_callback at the actual deallocation time.
  UNODB_DETAIL_DISABLE_MSVC_WARNING(26447)
  static void deallocate(
      void* pointer
#ifndef NDEBUG
      ,
      const detail::deallocation_request::debug_callback& debug_callback
#endif
      ) noexcept {
#ifndef NDEBUG
    if (debug_callback != nullptr) debug_callback(pointer);
#endif
    detail::free_aligned(pointer);
  }
  UNODB_DETAIL_RESTORE_MSVC_WARNINGS()

  /// Synchronization barrier for thread epoch changes.
  ///
  /// Synchronizes with qsbr::epoch_change_barrier_and_handle_orphans.
  static void thread_epoch_change_barrier() noexcept;

  /// Synchronize threads and handle orphaned requests.
  ///
  /// Synchronizes with qsbr::thread_epoch_change_barrier.
  ///
  /// \param single_thread_mode Whether there is at most one QSBR thread.
  void epoch_change_barrier_and_handle_orphans(
      bool single_thread_mode) noexcept;

  /// Advance to the next epoch by synchronizing threads, handling orphaned
  /// requests, and publishing the new global QSBR state.
  ///
  /// \param current_global_epoch Current global epoch
  /// \param single_thread_mode Whether there is at most one QSBR thread.
  /// \return New global epoch
  qsbr_epoch change_epoch(qsbr_epoch current_global_epoch,
                          bool single_thread_mode) noexcept;

#ifdef UNODB_DETAIL_WITH_STATS
  /// Increment the epoch change counter.
  void bump_epoch_change_count() noexcept;

  /// Make updated deallocation size statistics visible for readers.
  void publish_deallocation_size_stats() {
    deallocation_size_per_thread_max.store(
        boost_acc::max(deallocation_size_per_thread_stats),
        std::memory_order_relaxed);
    deallocation_size_per_thread_mean.store(
        boost_acc::mean(deallocation_size_per_thread_stats),
        std::memory_order_relaxed);
    deallocation_size_per_thread_variance.store(
        boost_acc::variance(deallocation_size_per_thread_stats),
        std::memory_order_relaxed);
  }

  /// Make updated deallocation count statistics visible for readers.
  void publish_epoch_callback_stats() {
    epoch_dealloc_per_thread_count_max.store(
        boost_acc::max(epoch_dealloc_per_thread_count_stats),
        std::memory_order_relaxed);
    epoch_dealloc_per_thread_count_variance.store(
        boost_acc::variance(epoch_dealloc_per_thread_count_stats),
        std::memory_order_relaxed);
  }

  /// Make updated Q state between thread epoch change statistics visible for
  /// readers.
  void publish_quiescent_states_per_thread_between_epoch_change_stats() {
    quiescent_states_per_thread_between_epoch_change_mean.store(
        boost_acc::mean(quiescent_states_per_thread_between_epoch_change_stats),
        std::memory_order_relaxed);
  }

#endif  // UNODB_DETAIL_WITH_STATS

  /// The global QSBR state.
  alignas(detail::hardware_destructive_interference_size)
      std::atomic<qsbr_state::type> state;

  /// Orphaned deallocation requests for the previous epoch from terminated
  /// threads.
  std::atomic<detail::dealloc_vector_list_node*>
      orphaned_previous_interval_dealloc_requests;

  /// Orphaned deallocation requests for the current epoch from terminated
  /// threads.
  std::atomic<detail::dealloc_vector_list_node*>
      orphaned_current_interval_dealloc_requests;

  static_assert(sizeof(state) +
                        sizeof(orphaned_previous_interval_dealloc_requests) +
                        sizeof(orphaned_current_interval_dealloc_requests) <=
                    detail::hardware_constructive_interference_size,
                "Global QSBR fields must fit into a single cache line");

#ifdef UNODB_DETAIL_WITH_STATS

  /// Epoch change counter.
  alignas(detail::hardware_destructive_interference_size)
      std::atomic<std::uint64_t> epoch_change_count;

  /// Mutex protecting deallocation statistics.
  alignas(detail::hardware_destructive_interference_size) std::mutex
      dealloc_stats_lock;

  // TODO(laurynas): more interesting callback stats?

  /// Statistics for deallocation counts per thread between epoch changes.
  boost_acc::accumulator_set<
      std::size_t,
      boost_acc::stats<boost_acc::tag::max, boost_acc::tag::variance>>
      epoch_dealloc_per_thread_count_stats;

  /// Maximum deallocation count per thread between epoch changes.
  std::atomic<std::size_t> epoch_dealloc_per_thread_count_max;

  /// Variance of deallocation count per thread between epoch changes..
  std::atomic<double> epoch_dealloc_per_thread_count_variance;

  /// Statistics for deallocation sizes per thread between epoch changes.
  boost_acc::accumulator_set<
      std::uint64_t,
      boost_acc::stats<boost_acc::tag::max, boost_acc::tag::variance>>
      deallocation_size_per_thread_stats;

  /// Maximum total deallocation size per thread between epoch changes.
  std::atomic<std::uint64_t> deallocation_size_per_thread_max;

  /// Mean total deallocation size per thread between epoch changes.
  std::atomic<double> deallocation_size_per_thread_mean;

  /// Variance of total deallocation size per thread between epoch changes.
  std::atomic<double> deallocation_size_per_thread_variance;

  /// Mutex protecting quiescent state statistics.
  alignas(detail::hardware_destructive_interference_size) std::mutex
      quiescent_state_stats_lock;

  /// Statistics for quiescent states per thread between epoch changes.
  boost_acc::accumulator_set<std::uint64_t,
                             boost_acc::stats<boost_acc::tag::mean>>
      quiescent_states_per_thread_between_epoch_change_stats;
  /// Mean number of quiescent states between epoch changes per thread.
  std::atomic<double> quiescent_states_per_thread_between_epoch_change_mean;

#endif  // UNODB_DETAIL_WITH_STATS
};

static_assert(std::atomic<std::size_t>::is_always_lock_free);
static_assert(std::atomic<double>::is_always_lock_free);

UNODB_DETAIL_DISABLE_MSVC_WARNING(26455)
inline qsbr_per_thread::qsbr_per_thread()
    : last_seen_quiescent_state_epoch{qsbr::instance().register_thread()},
      last_seen_epoch{last_seen_quiescent_state_epoch} {}
UNODB_DETAIL_RESTORE_MSVC_WARNINGS()

inline void qsbr_per_thread::on_next_epoch_deallocate(
    void* pointer
#ifdef UNODB_DETAIL_WITH_STATS
    ,
    std::size_t size
#endif
#ifndef NDEBUG
    ,
    detail::deallocation_request::debug_callback dealloc_callback
#endif
) {
  UNODB_DETAIL_ASSERT(!is_qsbr_paused());

  const auto current_qsbr_state = qsbr::instance().get_state();
  const auto current_global_epoch = qsbr_state::get_epoch(current_qsbr_state);
  const auto single_thread_mode =
      qsbr_state::single_thread_mode(current_qsbr_state);

  if (UNODB_DETAIL_UNLIKELY(single_thread_mode)) {
    advance_last_seen_epoch(single_thread_mode, current_global_epoch);
    qsbr::deallocate(pointer
#ifndef NDEBUG
                     ,
                     dealloc_callback
#endif
    );
    return;
  }

  if (last_seen_epoch != current_global_epoch) {
    detail::dealloc_request_vector new_current_requests;
    new_current_requests.emplace_back(pointer
#ifndef NDEBUG
                                      ,
                                      current_global_epoch,
                                      std::move(dealloc_callback)
#endif
    );
    advance_last_seen_epoch(single_thread_mode, current_global_epoch,
                            std::move(new_current_requests));
    UNODB_DETAIL_ASSERT(current_interval_dealloc_requests.size() == 1);
#ifdef UNODB_DETAIL_WITH_STATS
    UNODB_DETAIL_ASSERT(current_interval_total_dealloc_size == 0);
    current_interval_total_dealloc_size = size;
#endif  // UNODB_DETAIL_WITH_STATS
    return;
  }

  current_interval_dealloc_requests.emplace_back(pointer
#ifndef NDEBUG
                                                 ,
                                                 last_seen_epoch,
                                                 std::move(dealloc_callback)
#endif
  );
#ifdef UNODB_DETAIL_WITH_STATS
  current_interval_total_dealloc_size += size;
#endif  // UNODB_DETAIL_WITH_STATS
}

inline void qsbr_per_thread::advance_last_seen_epoch(
    bool single_thread_mode, qsbr_epoch new_seen_epoch,
    detail::dealloc_request_vector new_current_requests)
#ifndef UNODB_DETAIL_WITH_STATS
    noexcept
#endif
{
  if (new_seen_epoch == last_seen_epoch) return;

  // NOLINTNEXTLINE(readability-simplify-boolean-expr)
  UNODB_DETAIL_ASSERT(
      new_seen_epoch == last_seen_epoch.advance()
      // The current thread is 1) quitting; 2) not having seen the current epoch
      // yet; 3) it quitting will cause an epoch advance
      || (!single_thread_mode && new_seen_epoch == last_seen_epoch.advance(2)));

  execute_previous_requests(single_thread_mode, new_seen_epoch,
                            std::move(new_current_requests));
}

inline void qsbr_per_thread::execute_previous_requests(
    bool single_thread_mode, qsbr_epoch dealloc_epoch,
    detail::dealloc_request_vector new_current_requests)
#ifndef UNODB_DETAIL_WITH_STATS
    noexcept
#endif
{
  last_seen_epoch = dealloc_epoch;

  const detail::deferred_requests requests_to_deallocate{
      std::move(previous_interval_dealloc_requests)
#ifndef NDEBUG
          ,
      false, dealloc_epoch, single_thread_mode
#endif
  };

#ifdef UNODB_DETAIL_WITH_STATS
  qsbr::instance().register_dealloc_stats_per_thread_between_epoch_changes(
      current_interval_total_dealloc_size,
      current_interval_dealloc_requests.size());
  current_interval_total_dealloc_size = 0;
#endif  // UNODB_DETAIL_WITH_STATS

  if (UNODB_DETAIL_LIKELY(!single_thread_mode)) {
    previous_interval_dealloc_requests =
        std::move(current_interval_dealloc_requests);
  } else {
    previous_interval_dealloc_requests.clear();
    const detail::deferred_requests additional_requests_to_deallocate{
        std::move(current_interval_dealloc_requests)
#ifndef NDEBUG
            ,
        false, dealloc_epoch, single_thread_mode
#endif
    };
  }
  current_interval_dealloc_requests = std::move(new_current_requests);
}

inline void qsbr_per_thread::quiescent()
#ifndef UNODB_DETAIL_WITH_STATS
    noexcept
#endif
{
  UNODB_DETAIL_ASSERT(!paused);
  UNODB_DETAIL_ASSERT(active_ptrs.empty());

  const auto state = qsbr::instance().get_state();
  const auto current_global_epoch = qsbr_state::get_epoch(state);
  const auto single_thread_mode = qsbr_state::single_thread_mode(state);

  advance_last_seen_epoch(single_thread_mode, current_global_epoch);

  if (current_global_epoch != last_seen_quiescent_state_epoch) {
    UNODB_DETAIL_ASSERT(current_global_epoch ==
                        last_seen_quiescent_state_epoch.advance());

    last_seen_quiescent_state_epoch = current_global_epoch;
#ifdef UNODB_DETAIL_WITH_STATS
    qsbr::instance().register_quiescent_states_per_thread_between_epoch_changes(
        quiescent_states_since_epoch_change);
#endif  // UNODB_DETAIL_WITH_STATS
    quiescent_states_since_epoch_change = 0;
  }

  UNODB_DETAIL_ASSERT(current_global_epoch == last_seen_quiescent_state_epoch);
  if (quiescent_states_since_epoch_change == 0) {
    const auto new_global_epoch =
        qsbr::instance().remove_thread_from_previous_epoch(
            current_global_epoch
#ifndef NDEBUG
            ,
            last_seen_quiescent_state_epoch
#endif
        );
    UNODB_DETAIL_ASSERT(new_global_epoch == last_seen_quiescent_state_epoch ||
                        new_global_epoch ==
                            last_seen_quiescent_state_epoch.advance());

    if (new_global_epoch != last_seen_quiescent_state_epoch) {
      last_seen_quiescent_state_epoch = new_global_epoch;

      UNODB_DETAIL_ASSERT(last_seen_epoch.advance() == new_global_epoch);
      execute_previous_requests(single_thread_mode, new_global_epoch);

#ifdef UNODB_DETAIL_WITH_STATS
      qsbr::instance()
          .register_quiescent_states_per_thread_between_epoch_changes(1);
#endif  // UNODB_DETAIL_WITH_STATS
      return;
    }
  }
  ++quiescent_states_since_epoch_change;
}

namespace detail {

inline void deallocation_request::deallocate(
#ifndef NDEBUG
    bool orphan, std::optional<qsbr_epoch> dealloc_epoch,
    std::optional<bool> dealloc_epoch_single_thread_mode
#endif
) const noexcept {
  // NOLINTNEXTLINE(readability-simplify-boolean-expr)
  UNODB_DETAIL_ASSERT((orphan && !dealloc_epoch.has_value() &&
                       !dealloc_epoch_single_thread_mode.has_value()) ||
                      (!orphan && dealloc_epoch.has_value() &&
                       dealloc_epoch_single_thread_mode.has_value() &&
                       ((*dealloc_epoch_single_thread_mode &&
                         *dealloc_epoch == request_epoch.advance()) ||
                        *dealloc_epoch == request_epoch.advance(2))));

  qsbr::deallocate(pointer
#ifndef NDEBUG
                   ,
                   dealloc_callback
#endif
  );

#ifndef NDEBUG
  instance_count.fetch_sub(1, std::memory_order_relaxed);
#endif
}

}  // namespace detail

inline void qsbr_per_thread::qsbr_pause()
#ifndef UNODB_DETAIL_WITH_STATS
    noexcept
#endif
{
  UNODB_DETAIL_ASSERT(!paused);
  UNODB_DETAIL_ASSERT(active_ptrs.empty());

  qsbr::instance().unregister_thread(quiescent_states_since_epoch_change,
                                     last_seen_quiescent_state_epoch, *this);
  paused = true;

  UNODB_DETAIL_ASSERT(previous_interval_requests_empty());
  UNODB_DETAIL_ASSERT(previous_interval_orphan_list_node == nullptr);
  UNODB_DETAIL_ASSERT(current_interval_requests_empty());
  UNODB_DETAIL_ASSERT(current_interval_orphan_list_node == nullptr);
}

inline void qsbr_per_thread::qsbr_resume() {
  UNODB_DETAIL_ASSERT(paused);
  UNODB_DETAIL_ASSERT(active_ptrs.empty());
  UNODB_DETAIL_ASSERT(previous_interval_requests_empty());
  UNODB_DETAIL_ASSERT(current_interval_requests_empty());

  previous_interval_orphan_list_node =
      std::make_unique<detail::dealloc_vector_list_node>();
  current_interval_orphan_list_node =
      std::make_unique<detail::dealloc_vector_list_node>();

  last_seen_quiescent_state_epoch = qsbr::instance().register_thread();
  last_seen_epoch = last_seen_quiescent_state_epoch;
  quiescent_states_since_epoch_change = 0;
  paused = false;
}

/// RAII guard that signals quiescent state for this thread on destruction.
struct quiescent_state_on_scope_exit final {
  /// Default constructor.
  quiescent_state_on_scope_exit() = default;

  /// Destructor, that signals quiescent state for this thread.
  ~quiescent_state_on_scope_exit()
#ifdef UNODB_DETAIL_WITH_STATS
      noexcept(false)
#else
      noexcept
#endif
  {
#ifdef UNODB_DETAIL_WITH_STATS
    // TODO(laurynas): to avoid try/catch, see the TODO at the previous
    // try/catch.
    try {
      this_thread().quiescent();
    }
    // LCOV_EXCL_START
    catch (const std::system_error& e) {
      if (exceptions_at_ctor == std::uncaught_exceptions()) throw;
      std::cerr << "QSBR quiescent state failed to register QSBR stats: "
                << e.what() << '\n';
    } catch (const std::exception& e) {
      if (exceptions_at_ctor == std::uncaught_exceptions()) throw;
      std::cerr << "QSBR quiescent state exception: " << e.what() << '\n';
      UNODB_DETAIL_DEBUG_CRASH();
    } catch (...) {
      if (exceptions_at_ctor == std::uncaught_exceptions()) throw;
      std::cerr << "QSBR quiescent state unknown exception\n";
      UNODB_DETAIL_DEBUG_CRASH();
    }
    // LCOV_EXCL_STOP
#else
    this_thread().quiescent();
#endif
  }

  /// Copy construction is disabled.
  quiescent_state_on_scope_exit(const quiescent_state_on_scope_exit&) = delete;

  /// Move construction is disabled.
  quiescent_state_on_scope_exit(quiescent_state_on_scope_exit&&) = delete;

  /// Copy assignment is disabled.
  quiescent_state_on_scope_exit& operator=(
      const quiescent_state_on_scope_exit&) = delete;

  /// Move assignment is disabled.
  quiescent_state_on_scope_exit& operator=(quiescent_state_on_scope_exit&&) =
      delete;

#ifdef UNODB_DETAIL_WITH_STATS

 private:
  /// Count of exceptions at constructor time for proper rethrow handling.
  const int exceptions_at_ctor{std::uncaught_exceptions()};
#endif
};

/// An `std::thread`-like thread that participates in QSBR.
///
/// Ensures that a thread-local unodb::qsbr_per_thread instance gets properly
/// constructed.
class [[nodiscard]] qsbr_thread : public std::thread {
 public:
  using thread::thread;

  /// Create a new thread running function \a f with arguments \a args.
  template <typename Function, typename... Args>
  explicit qsbr_thread(Function&& f, Args&&... args)
    requires(!std::is_same_v<std::remove_cvref_t<Function>, qsbr_thread>)
      : std::thread{make_qsbr_thread(std::forward<Function>(f),
                                     std::forward<Args>(args)...)} {}

 private:
  /// Create a thread with a unodb::qsbr_per_thread instance set.
  template <typename Function, typename... Args>
  [[nodiscard]] static std::thread make_qsbr_thread(Function&& f,
                                                    Args&&... args) {
    auto new_qsbr_per_thread = std::make_unique<qsbr_per_thread>();
    return std::thread{
        [inner_new_qsbr_per_thread = std::move(new_qsbr_per_thread)](
            auto&& f2,
            // cppcheck-suppress constStatement
            auto&&... a2) mutable noexcept(noexcept(f2(std::
                                                           forward<
                                                               decltype(a2)>(
                                                               a2)...))) {
          qsbr_per_thread::set_instance(std::move(inner_new_qsbr_per_thread));
          f2(std::forward<decltype(a2)>(a2)...);
        },
        std::forward<Function>(f), std::forward<Args>(args)...};
  }
};

}  // namespace unodb

#endif  // UNODB_DETAIL_QSBR_HPP
