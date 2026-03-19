// Copyright (C) 2019-2025 UnoDB contributors

/// \file
/// QSBR implementation details.
///
/// \ingroup qsbr
///
/// Implementation of non-inline symbols from qsbr.hpp, main thread QSBR
/// registration and orphaned deallocation request list management helpers.

// Should be the first include
#include "global.hpp"

// IWYU pragma: no_include <__new/exceptions.h>
// IWYU pragma: no_include <__hash_table>

#include <atomic>
#include <cstdint>
#include <exception>
#include <iostream>
#include <memory>
#include <new>  // IWYU pragma: keep
#include <utility>

#include "qsbr.hpp"

#ifdef UNODB_DETAIL_WITH_STATS
#include <mutex>
#endif  // UNODB_DETAIL_WITH_STATS

#ifdef UNODB_DETAIL_THREAD_SANITIZER
#include <sanitizer/tsan_interface.h>
#endif

#include "assert.hpp"

namespace unodb::detail {

/// Helper for registering the main thread with QSBR.
struct set_qsbr_per_thread_in_main_thread {
  UNODB_DETAIL_DISABLE_MSVC_WARNING(26447)
  /// Register main thread with QSBR during static initialization.
  ///
  /// Creates and registers a QSBR instance for the main thread before main code
  /// runs. This ensures the main thread starts out participating in QSBR.
  ///
  /// \note Fails fatally if unable to allocate the QSBR instance.
  set_qsbr_per_thread_in_main_thread() noexcept {
    try {
      auto main_thread_qsbr_reclamator_instance =
          std::make_unique<unodb::qsbr_per_thread>();
      unodb::qsbr_per_thread::set_instance(
          std::move(main_thread_qsbr_reclamator_instance));
    }
    // LCOV_EXCL_START
    catch (const std::bad_alloc& e) {
      std::cerr << "Allocation failure: " << e.what() << '\n';
      UNODB_DETAIL_CRASH();
    } catch (const std::exception& e) {
      std::cerr << "Unexpected exception: " << e.what() << '\n';
      UNODB_DETAIL_CRASH();
    } catch (...) {
      std::cerr << "Unexpected exception\n";
      UNODB_DETAIL_CRASH();
    }
    // LCOV_EXCL_STOP
  }
  UNODB_DETAIL_RESTORE_MSVC_WARNINGS()
};

}  // namespace unodb::detail

namespace {

/// Global instance to register the thread with QSBR during program startup.
// NOLINTNEXTLINE(fuchsia-statically-constructed-objects)
const unodb::detail::set_qsbr_per_thread_in_main_thread do_it;

}  // namespace

namespace unodb {

// Try converting to constinit at C++23
thread_local std::unique_ptr<qsbr_per_thread>
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
    qsbr_per_thread::current_thread_instance;

// LCOV_EXCL_START
[[gnu::cold]] UNODB_DETAIL_NOINLINE void qsbr_epoch::dump(
    std::ostream& os) const {
  os << "epoch = " << static_cast<std::uint64_t>(epoch_val);
#ifndef NDEBUG
  assert_invariant();
#endif
}
// LCOV_EXCL_STOP

[[nodiscard]] qsbr_state::type
qsbr_state::atomic_fetch_dec_threads_in_previous_epoch(
    std::atomic<qsbr_state::type>& word) noexcept {
  const auto old_word = word.fetch_sub(1, std::memory_order_acq_rel);

  UNODB_DETAIL_ASSERT(get_threads_in_previous_epoch(old_word) > 0);
  assert_invariants(old_word);

  return old_word;
}

// Some GCC versions suggest cold attribute on already cold-marked functions
UNODB_DETAIL_DISABLE_GCC_WARNING("-Wsuggest-attribute=cold")

// LCOV_EXCL_START
[[gnu::cold]] UNODB_DETAIL_NOINLINE void qsbr_state::dump(std::ostream& os,
                                                          type word) {
  os << "QSBR state: " << do_get_epoch(word)
     << ", threads = " << do_get_thread_count(word)
     << ", threads in the previous epoch = "
     << do_get_threads_in_previous_epoch(word);
  assert_invariants(word);
}
// LCOV_EXCL_STOP

UNODB_DETAIL_RESTORE_GCC_WARNINGS()

#ifndef NDEBUG

namespace detail {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
constinit std::atomic<std::uint64_t> deallocation_request::instance_count{0};

}  // namespace detail

void qsbr_per_thread::register_active_ptr(const void* ptr) {
  UNODB_DETAIL_ASSERT(ptr != nullptr);
  UNODB_DETAIL_ASSERT(!is_qsbr_paused());

  active_ptrs.insert(ptr);
}

void qsbr_per_thread::unregister_active_ptr(const void* ptr) {
  UNODB_DETAIL_ASSERT(ptr != nullptr);
  UNODB_DETAIL_ASSERT(!is_qsbr_paused());

  const auto itr = active_ptrs.find(ptr);
  UNODB_DETAIL_ASSERT(itr != active_ptrs.end());
  active_ptrs.erase(itr);
}

#endif  // !NDEBUG

namespace {

/// Add vector of pending deallocation requests to orphan list.
///
/// This function is used when a thread quits with pending deallocation
/// requests. These requests become "orphaned" and are added to a global list
/// for later processing. A preallocated list node is passed to avoid a
/// potential OOM failure at a thread exit time.
///
/// \param orphan_list Global orphan list
/// \param requests Deallocation requests to be added to the orphan list
/// \param orphan_list_node Preallocated list node that will hold the requests
void add_to_orphan_list(
    std::atomic<detail::dealloc_vector_list_node*>& orphan_list,
    detail::dealloc_request_vector&& requests,
    std::unique_ptr<detail::dealloc_vector_list_node>
        orphan_list_node) noexcept {
  if (requests.empty()) return;

  auto* const list_node_ptr = orphan_list_node.release();

  list_node_ptr->requests = std::move(requests);
  list_node_ptr->next = orphan_list.load(std::memory_order_acquire);

  while (true) {
    if (UNODB_DETAIL_LIKELY(orphan_list.compare_exchange_weak(
            list_node_ptr->next, list_node_ptr, std::memory_order_acq_rel,
            std::memory_order_acquire)))
      return;
  }
}

/// Atomically take ownership of entire orphaned request list.
///
/// Replaces the list with `nullptr` and returns the previous head node.
///
/// \param orphan_list Global orphaned request list
/// \return Taken orphaned request list
[[nodiscard]] detail::dealloc_vector_list_node* take_orphan_list(
    std::atomic<detail::dealloc_vector_list_node*>& orphan_list
    UNODB_DETAIL_LIFETIMEBOUND) noexcept {
  return orphan_list.exchange(nullptr, std::memory_order_acq_rel);
}

/// Free pending requests and orphan \a list itself.
void free_orphan_list(detail::dealloc_vector_list_node* list) noexcept {
  while (list != nullptr) {
    const std::unique_ptr<detail::dealloc_vector_list_node> list_ptr{list};
    const detail::deferred_requests requests_to_deallocate{
        std::move(list_ptr->requests)
#ifndef NDEBUG
            ,
        true,
        {},
        {}
#endif
    };
    list = list_ptr->next;
  }
}

}  // namespace

void qsbr_per_thread::orphan_pending_requests() noexcept {
  add_to_orphan_list(
      qsbr::instance().orphaned_previous_interval_dealloc_requests,
      std::move(previous_interval_dealloc_requests),
      std::move(previous_interval_orphan_list_node));
  add_to_orphan_list(
      qsbr::instance().orphaned_current_interval_dealloc_requests,
      std::move(current_interval_dealloc_requests),
      std::move(current_interval_orphan_list_node));

  previous_interval_dealloc_requests.clear();
  current_interval_dealloc_requests.clear();

  // cppcheck-suppress accessMoved
  UNODB_DETAIL_ASSERT(previous_interval_orphan_list_node == nullptr);
  // cppcheck-suppress accessMoved
  UNODB_DETAIL_ASSERT(current_interval_orphan_list_node == nullptr);
}

UNODB_DETAIL_DISABLE_CLANG_21_WARNING("-Wnrvo")
qsbr_epoch qsbr::register_thread() noexcept {
  auto old_state = get_state();

  while (true) {
    const auto old_epoch = qsbr_state::get_epoch(old_state);
    const auto old_thread_count = qsbr_state::get_thread_count(old_state);
    const auto old_threads_in_previous_epoch =
        qsbr_state::get_threads_in_previous_epoch(old_state);

    if (UNODB_DETAIL_LIKELY(old_threads_in_previous_epoch > 0 ||
                            old_thread_count == 0)) {
      const auto new_state =
          qsbr_state::inc_thread_count_and_threads_in_previous_epoch(old_state);

      if (UNODB_DETAIL_LIKELY(state.compare_exchange_weak(
              old_state, new_state, std::memory_order_acq_rel,
              std::memory_order_acquire)))
        return old_epoch;

      // LCOV_EXCL_START
      continue;
    }

    // TODO(laurynas): this and the rest of coverage exclusions would require
    // thread schedules that are impossible to get deterministically. Try to
    // device some thread pause-wait-continue facility for debug builds.
    UNODB_DETAIL_ASSERT(old_threads_in_previous_epoch == 0);
    UNODB_DETAIL_ASSERT(old_thread_count > 0);

    // Epoch change in progress - try to bump the thread count only
    const auto new_state = qsbr_state::inc_thread_count(old_state);

    if (UNODB_DETAIL_LIKELY(state.compare_exchange_weak(
            old_state, new_state, std::memory_order_acq_rel,
            std::memory_order_acquire))) {
      // Spin until the epoch change completes. An alternative would be to
      // return the new epoch early, and then handle seeing it in quiescent
      // state as a no-op, but that trades spinning here for more work in a
      // hotter path.
      while (true) {
        old_state = get_state();
        const auto new_epoch = qsbr_state::get_epoch(old_state);
        if (new_epoch != old_epoch) return new_epoch;
      }
    }
  }
  // LCOV_EXCL_STOP
}
UNODB_DETAIL_RESTORE_CLANG_21_WARNINGS()

void qsbr::unregister_thread(std::uint64_t quiescent_states_since_epoch_change,
                             qsbr_epoch thread_epoch,
                             qsbr_per_thread& qsbr_thread)
#ifndef UNODB_DETAIL_WITH_STATS
    noexcept
#endif
{
  bool epoch_change_prepared = false;
  auto old_state = state.load(std::memory_order_acquire);

  while (true) {
    const auto old_threads_in_previous_epoch =
        qsbr_state::get_threads_in_previous_epoch(old_state);

    if (UNODB_DETAIL_UNLIKELY(old_threads_in_previous_epoch == 0)) {
      // LCOV_EXCL_START
      UNODB_DETAIL_ASSERT(thread_epoch == qsbr_state::get_epoch(old_state));

      // Epoch change in progress - try to decrement the thread count only
      const auto new_state = qsbr_state::dec_thread_count(old_state);
      if (UNODB_DETAIL_LIKELY(state.compare_exchange_weak(
              old_state, new_state, std::memory_order_acq_rel,
              std::memory_order_acquire))) {
        qsbr_thread.orphan_pending_requests();
        return;
      }
      continue;
      // LCOV_EXCL_STOP
    }

    UNODB_DETAIL_ASSERT(old_threads_in_previous_epoch > 0);
    const auto old_epoch = qsbr_state::get_epoch(old_state);
    const auto old_single_thread_mode =
        qsbr_state::single_thread_mode(old_state);

    const auto remove_thread_from_old_epoch =
        (thread_epoch != old_epoch) ||
        (quiescent_states_since_epoch_change == 0);
    const auto advance_epoch =
        remove_thread_from_old_epoch && (old_threads_in_previous_epoch == 1);

    const auto new_state =
        UNODB_DETAIL_UNLIKELY(remove_thread_from_old_epoch)
            ? qsbr_state::
                  dec_thread_count_threads_in_previous_epoch_maybe_advance(
                      old_state, advance_epoch)
            : qsbr_state::dec_thread_count(old_state);

    if (UNODB_DETAIL_UNLIKELY(remove_thread_from_old_epoch)) {
      thread_epoch_change_barrier();

      if (UNODB_DETAIL_UNLIKELY(advance_epoch) &&
          UNODB_DETAIL_LIKELY(!epoch_change_prepared)) {
        // Handle global orphans only once for one epoch change. We cannot do
        // this after setting the new state as then other threads may proceed
        // with subsequent epoch changes.
        epoch_change_barrier_and_handle_orphans(old_single_thread_mode);
        epoch_change_prepared = true;
      }
    }

    if (UNODB_DETAIL_LIKELY(state.compare_exchange_weak(
            old_state, new_state, std::memory_order_acq_rel,
            std::memory_order_acquire))) {
      // Might be the first time the quitting thread saw the old epoch too, if a
      // second-to-last thread quit before, advancing the epoch.
      qsbr_thread.advance_last_seen_epoch(old_single_thread_mode, old_epoch);
      if (UNODB_DETAIL_UNLIKELY(advance_epoch)) {
#ifdef UNODB_DETAIL_WITH_STATS
        bump_epoch_change_count();
#endif  // UNODB_DETAIL_WITH_STATS
        qsbr_thread.execute_previous_requests(old_single_thread_mode,
                                              old_epoch.advance());
      }
      qsbr_thread.orphan_pending_requests();

#ifdef UNODB_DETAIL_WITH_STATS
      if (UNODB_DETAIL_UNLIKELY(thread_epoch != old_epoch)) {
        register_quiescent_states_per_thread_between_epoch_changes(
            quiescent_states_since_epoch_change);
      }
#endif  // UNODB_DETAIL_WITH_STATS

      return;
    }
  }
}

#ifdef UNODB_DETAIL_WITH_STATS

void qsbr::reset_stats() {
  // Stats can only be reset on idle QSBR - best-effort check as nothing
  // prevents to leaving idle state at any time
  assert_idle();

  {
    const std::lock_guard guard{dealloc_stats_lock};

    epoch_dealloc_per_thread_count_stats = {};
    publish_epoch_callback_stats();

    deallocation_size_per_thread_stats = {};
    publish_deallocation_size_stats();
  }

  {
    const std::lock_guard guard{quiescent_state_stats_lock};

    quiescent_states_per_thread_between_epoch_change_stats = {};
    publish_quiescent_states_per_thread_between_epoch_change_stats();
  }
}

#endif  // UNODB_DETAIL_WITH_STATS

// Some GCC versions suggest cold attribute on already cold-marked functions
UNODB_DETAIL_DISABLE_GCC_WARNING("-Wsuggest-attribute=cold")

[[gnu::cold]] UNODB_DETAIL_NOINLINE void qsbr::dump(std::ostream& out) const {
  out << state.load(std::memory_order_acquire) << '\n';
}

UNODB_DETAIL_RESTORE_GCC_WARNINGS()

void qsbr::thread_epoch_change_barrier() noexcept {
#ifndef UNODB_DETAIL_THREAD_SANITIZER
  // No loads and stores can be reordered past this point, or the quiescent
  // state contract would be violated
  std::atomic_thread_fence(std::memory_order_release);
#else
  // I have no idea what I am doing
  __tsan_release(&instance());
#endif
}

qsbr_epoch qsbr::remove_thread_from_previous_epoch(
    qsbr_epoch current_global_epoch
#ifndef NDEBUG
    ,
    qsbr_epoch thread_epoch
#endif
    ) noexcept {
  thread_epoch_change_barrier();

  const auto old_state =
      qsbr_state::atomic_fetch_dec_threads_in_previous_epoch(state);

  const auto old_threads_in_previous_epoch =
      qsbr_state::get_threads_in_previous_epoch(old_state);
  const auto old_single_thread_mode = qsbr_state::single_thread_mode(old_state);

  // The global epoch could not have advanced since the passed in value was
  // read because this thread is passing through the quiescent state for the
  // first time in this epoch.
  UNODB_DETAIL_ASSERT(current_global_epoch == qsbr_state::get_epoch(old_state));
  UNODB_DETAIL_ASSERT(thread_epoch == current_global_epoch ||
                      thread_epoch.advance() == current_global_epoch);

  if (old_threads_in_previous_epoch > 1) return current_global_epoch;

  const auto new_epoch =
      change_epoch(current_global_epoch, old_single_thread_mode);

  UNODB_DETAIL_ASSERT(current_global_epoch.advance() == new_epoch);

  return new_epoch;
}

#ifdef UNODB_DETAIL_WITH_STATS

void qsbr::bump_epoch_change_count() noexcept {
  const auto new_epoch_change_count =
      epoch_change_count.load(std::memory_order_relaxed) + 1;
  epoch_change_count.store(new_epoch_change_count, std::memory_order_relaxed);
}

#endif  // UNODB_DETAIL_WITH_STATS

void qsbr::epoch_change_barrier_and_handle_orphans(
    bool single_thread_mode) noexcept {
#ifndef UNODB_DETAIL_THREAD_SANITIZER
  // Acquire synchronizes-with atomic_thread_fence(std::memory_order_release)
  // in thread_epoch_change_barrier
  std::atomic_thread_fence(std::memory_order_acquire);
#else
  __tsan_acquire(&instance());
#endif

  auto* orphaned_previous_requests =
      take_orphan_list(orphaned_previous_interval_dealloc_requests);
  auto* orphaned_current_requests =
      take_orphan_list(orphaned_current_interval_dealloc_requests);

  free_orphan_list(orphaned_previous_requests);

  if (UNODB_DETAIL_LIKELY(!single_thread_mode)) {
    detail::dealloc_vector_list_node* new_previous_requests = nullptr;
    if (UNODB_DETAIL_UNLIKELY(
            !orphaned_previous_interval_dealloc_requests
                 .compare_exchange_strong(
                     new_previous_requests, orphaned_current_requests,
                     std::memory_order_acq_rel, std::memory_order_acquire))) {
      // Someone added new previous requests since we took the previous batch
      // above. Append ours at the tail then, only one thread can do this, as
      // everybody else add at the list head. The list should be short in
      // general case as not too many threads could have quit since we took the
      // previous batch.
      while (new_previous_requests->next != nullptr)
        new_previous_requests = new_previous_requests->next;
      new_previous_requests->next = orphaned_current_requests;
    }
  } else {
    free_orphan_list(orphaned_current_requests);
  }
}

qsbr_epoch qsbr::change_epoch(qsbr_epoch current_global_epoch,
                              bool single_thread_mode) noexcept {
  epoch_change_barrier_and_handle_orphans(single_thread_mode);

  auto old_state = state.load(std::memory_order_acquire);
  while (true) {
    UNODB_DETAIL_ASSERT(current_global_epoch ==
                        qsbr_state::get_epoch(old_state));

    const auto new_state = qsbr_state::inc_epoch_reset_previous(old_state);
    if (UNODB_DETAIL_LIKELY(state.compare_exchange_weak(
            old_state, new_state, std::memory_order_acq_rel,
            std::memory_order_acquire))) {
      UNODB_DETAIL_ASSERT(current_global_epoch.advance() ==
                          qsbr_state::get_epoch(new_state));

#ifdef UNODB_DETAIL_WITH_STATS
      bump_epoch_change_count();
#endif  // UNODB_DETAIL_WITH_STATS
      return current_global_epoch.advance();
    }

    // Nobody else can change epoch nor threads in the previous epoch, only
    // allowed failures are thread count change and spurious. The next loop
    // iteration will assert this.
  }
}

}  // namespace unodb
