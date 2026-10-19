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

/// Run \a parked on one QSBR thread, stopped at the \a ParkHit-th hit of
/// \a point, while \a interleaved runs to completion on another.
///
/// Each callable runs inside a `unodb::quiescent_state_on_scope_exit` scope.
/// \a interleaved starts only once \a parked has parked, so \a parked must hit
/// \a point at least \a ParkHit times.  \a parked resumes only once
/// \a interleaved returns, so \a interleaved must not need anything \a parked
/// holds at \a point: it spins on a node \a parked has write-locked, even one
/// it only descends through, and the call never returns.  The hook counts both
/// threads' hits and parks only on the \a ParkHit-th, so neither a restart of
/// \a parked nor \a interleaved passing \a point parks again.
///
/// The calling thread is QSBR-paused until both threads have joined and
/// passes a quiescent state before returning, so it must not be paused
/// already and must not hold a QSBR-protected pointer or a view into a tree
/// across the call.
///
/// \return The number of times either thread hit \a point.  Each call site's
/// schedule fixes it, so callers pin it exactly: that is what shows an
/// unexpected extra restart, or a write path that starts passing \a point.
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
  auto parked_thread = unodb::qsbr_thread([&parked] {
    const unodb::quiescent_state_on_scope_exit q{};
    parked();
  });
  parked_thread.join();
  interleaved_thread.join();
  unodb::this_thread().qsbr_resume();
  unodb::this_thread().quiescent();
  return hits;
}

}  // namespace unodb::test

#endif  // NDEBUG

#endif  // UNODB_DETAIL_SYNC_POINT_TEST_UTILS_HPP
