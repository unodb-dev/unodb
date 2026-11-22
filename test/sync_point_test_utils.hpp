// Copyright 2026 UnoDB contributors
#ifndef UNODB_DETAIL_SYNC_POINT_TEST_UTILS_HPP
#define UNODB_DETAIL_SYNC_POINT_TEST_UTILS_HPP

/// \file
/// Sync point test helpers: an RAII guard that disarms a sync point, and the
/// two-thread protocol that parks one thread at it.
///
/// \ingroup test-internals

// Should be the first include
#include "global.hpp"  // IWYU pragma: keep

#ifndef NDEBUG

#include <gtest/gtest.h>

#include "qsbr.hpp"
#include "sync.hpp"
#include "thread_sync.hpp"

namespace unodb::test {

/// RAII guard that disarms a sync_point on destruction.
struct sync_point_guard {
  /// The sync point being guarded.
  unodb::detail::sync_point& pt_;

  /// Construct from a sync point reference.
  explicit sync_point_guard(unodb::detail::sync_point& pt) noexcept : pt_{pt} {}

  /// Disarm the sync point on destruction.
  ~sync_point_guard() { pt_.disarm(); }

  sync_point_guard(const sync_point_guard&) = delete;
  sync_point_guard& operator=(const sync_point_guard&) = delete;
  sync_point_guard(sync_point_guard&&) = delete;
  sync_point_guard& operator=(sync_point_guard&&) = delete;
};

/// Release the interleaved thread of a parked run whose parked operation
/// returned without parking, once the parked thread has joined.  The
/// interleaved thread must still be waiting on \a park_sync, and must notify
/// \a resume_sync once, after its operation returns.  Each of \a points is
/// disarmed first, since the interleaved operation may pass it and would then
/// park on itself or wait for a release that never comes.  The final wait
/// consumes that notify, so that both syncs end reset, as ~thread_sync asserts.
/// Report the miss only once the interleaved thread has joined too: under
/// --gtest_throw_on_failure the report throws, and unwinding past a joinable
/// thread would terminate the run before the exception could reach a handler.
template <class... Points>
void release_after_missed_park(unodb::detail::thread_sync& park_sync,
                               unodb::detail::thread_sync& resume_sync,
                               Points&... points) {
  (points.disarm(), ...);
  park_sync.notify();
  resume_sync.wait();
}

/// Run \a parked on one QSBR thread, stopped at the \a ParkHit-th hit of
/// \a point, while \a interleaved runs to completion on another.
///
/// Each callable runs inside a `unodb::quiescent_state_on_scope_exit` scope.
/// \a parked must hit \a point at least \a ParkHit times, since \a interleaved
/// waits for that park: if \a parked returns first, the test fails instead of
/// hanging, and \a interleaved then runs without a park.  \a parked resumes
/// only once \a interleaved returns, so \a interleaved must not need anything
/// \a parked holds at \a point: it spins on a node \a parked has write-locked,
/// even one it only descends through, and the call never returns.  The hook
/// counts both threads' hits and parks only on the \a ParkHit-th, so neither a
/// restart of \a parked nor \a interleaved passing \a point parks again.
///
/// The calling thread is QSBR-paused until both threads have joined and
/// passes a quiescent state before returning, so it must not be paused
/// already and must not hold a QSBR-protected pointer or a view into a tree
/// across the call.
///
/// \return The number of times either thread hit \a point, or only \a parked's
/// hits if the park was missed.  Each call site's schedule fixes it, so callers
/// pin it exactly: that is what shows an unexpected extra restart, or a write
/// path that starts passing \a point.
template <unsigned ParkHit = 1, class Parked, class Interleaved>
[[nodiscard]] unsigned run_parked_interleaving(unodb::detail::sync_point& point,
                                               const Parked& parked,
                                               const Interleaved& interleaved) {
  static_assert(ParkHit > 0, "hits are 1-based; 0 would never park");
  // Declared before the guard, so that the guard disarms the hook before the
  // counter and the handshake it captures die.  The handshake is local, so no
  // earlier call or callable can leave it signalled, and ~thread_sync checks
  // on return that this call left it reset.
  unsigned hits = 0;
  unodb::detail::thread_sync park_sync;
  unodb::detail::thread_sync resume_sync;
  const sync_point_guard guard{point};
  point.arm([&hits, &park_sync, &resume_sync] {
    if (++hits != ParkHit) return;
    park_sync.notify();
    resume_sync.wait();
  });

  unodb::this_thread().qsbr_pause();
  // Spawned first, so that it is registered with QSBR before parked's thread
  // exists: otherwise parked could start as the only registered thread, and
  // its deallocations before the park would be immediate instead of deferred.
  auto interleaved_thread =
      unodb::qsbr_thread([&interleaved, &park_sync, &resume_sync] {
        const unodb::quiescent_state_on_scope_exit q{};
        park_sync.wait();
        interleaved();
        resume_sync.notify();
      });
  // noexcept when its body is, as GCC's -Wnoexcept wants: parked may be, and
  // without stats so is ~quiescent_state_on_scope_exit.
  auto parked_thread = unodb::qsbr_thread(
      [&parked]() noexcept(noexcept(
          unodb::quiescent_state_on_scope_exit{}) && noexcept(parked())) {
        const unodb::quiescent_state_on_scope_exit q{};
        parked();
      });
  parked_thread.join();
  // interleaved has not run yet if parked never parked, and otherwise made all
  // its hits before releasing parked, so this read races with nothing.
  const bool missed_park = hits < ParkHit;
  // Some interleaved operations pass point too.
  if (missed_park) release_after_missed_park(park_sync, resume_sync, point);
  interleaved_thread.join();
  unodb::this_thread().qsbr_resume();
  unodb::this_thread().quiescent();
  if (missed_park) {
    ADD_FAILURE() << "parked returned before hit " << ParkHit
                  << " of its sync point";
  }
  return hits;
}

}  // namespace unodb::test

#endif  // NDEBUG

#endif  // UNODB_DETAIL_SYNC_POINT_TEST_UTILS_HPP
