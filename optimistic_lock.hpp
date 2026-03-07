// Copyright (C) 2019-2025 UnoDB contributors
#ifndef UNODB_DETAIL_OPTIMISTIC_LOCK_HPP
#define UNODB_DETAIL_OPTIMISTIC_LOCK_HPP

/// \file
/// Optimistic lock.
///
/// \ingroup optimistic-lock

/// \addtogroup optimistic-lock
/// \{
/// # Overview
///
/// A version-based optimistic lock that supports single-writer/multiple-readers
/// concurrency without shared memory writes during read operations. Writers
/// bump the version counter and readers detect concurrent writes by comparing
/// the version counter before and after the reads.
///
/// ## Examples
///
/// Protected data declaration and access API:
/// \code{.cpp}
/// // Multiple data fields protected by the same optimistic lock:
/// unodb::in_critical_section<std::uint64_t> val;
/// unodb::in_critical_section<std::uint64_t> val2;
/// // Transparent operations using the underlying type:
/// const auto bar = val + 5;
/// ++val; // etc.
/// // Explicit loads and stores when needed:
/// const auto baz = val2.load();
/// val2.store(10);
/// \endcode
///
/// The simplest read locking example:
/// \code{.cpp}
/// // Spin until lock is not write-locked nor obsolete
/// auto foo_read_critical_section = lock.try_read_lock();
/// if (foo_read_critical_section.must_restart()) {
///   // Obsolete, restart
///   return false;
/// }
/// // Read
/// const auto read_foo = foo.data;
/// // Try unlock
/// if (!foo_read_critical_section.try_read_unlock()) {
///    // The lock was write-locked while we were accessing data. Do not act on
///    // the read data, restart.
///    return false;
/// }
/// // Act on read_foo and return success
/// // ...
/// return true;
/// \endcode
///
/// An example of read locking with interim checks:
/// \code{.cpp}
/// auto foo_rcs = lock.try_read_lock();
/// if (foo_rcs.must_restart()) return false;
/// const auto read_foo_1 = foo.data_1;
/// // Check whether read_foo_1 was read consistently but do not end the read
/// // critical section
/// if (!foo_rcs.check()) {
///   // The check failed because the lock was write-locked while we were
///   // accessing data. Do not act on it, restart.
///   return false;
/// }
/// // Act on read_foo_1
/// // ...
/// const auto read_foo_2 = foo.data_2;
/// if (!foo_rcs.try_read_unlock()) return false;
/// // Both read_foo_1 and read_foo_2 were read consistently together, act on
/// // them.
/// // ...
/// return true;
/// \endcode
///
/// An example of write locking:
/// \code{.cpp}
/// // Write lock critical sections always start out as read lock ones.
/// auto foo_rcs = lock.try_read_lock();
/// if (foo_rcs.must_restart()) return false;
/// // Read current data state if needed
/// // ...
/// // Try upgrading the lock
/// const auto foo_write_guard =
///   optimistic_lock::write_guard{std::move(foo_rcs)};
/// if (foo_write_guard.must_restart()) {
///   // The lock upgrade failed because somebody else write-locked it
///   // first. Restart, also don't act on the read data in the read critical
///   // section since the last check.
///   return false;
/// }
/// // We have the exclusive write lock, freely write the data. The lock will be
/// // released on scope exit.
/// // ...
/// return true;
/// \endcode
///
/// An example of write locking that ends with data deletion:
/// \code{.cpp}
/// auto foo_rcs = lock.try_read_lock();
/// if (foo_rcs.must_restart()) return false;
/// auto foo_wg = optimistic_lock::write_guard{std::move(foo_rcs)};
/// if (foo_wg.must_restart()) return false;
/// // Act on write-locked data before marking it for deletion
/// // ...
/// foo_wg.unlock_and_obsolete();
/// // Mark data to be reclaimed when it is safe to do so
/// // ...
/// \endcode
///
/// ## API conventions
///
/// All `bool`-returning `try_` methods return true on success and false when
/// a concurrent write lock requires the operation to be restarted.
///
/// ## Protected data declaration
///
/// All data fields or variables to be protected by an optimistic lock must be
/// wrapped in unodb::in_critical_section template. Effectively it converts the
/// data accesses to relaxed atomic accesses, which is required by the
/// optimistic lock memory model.
///
/// ## Read protocol
/// \anchor olc-read-protocol
///
/// A read critical section (RCS) is created by
/// unodb::optimistic_lock::try_read_lock(), which will either spin until the
/// lock is not write-locked, or will return immediately if the lock goes to the
/// obsolete state.
///
/// The obsolete state must be checked for by calling
/// unodb::optimistic_lock::read_critical_section::must_restart() immediately
/// after creating the RCS.
///
/// No pointers may be dereferenced in an RCS before a successful read unlock
/// (unodb::optimistic_lock::try_read_unlock()) or an interim check
/// (unodb::optimistic_lock::check()) call. Similarly, no non-pointer data may
/// be accessed in any fault-causing way if it's illegal.
///
/// To follow the above rules, first copy the data of interest, then verify
/// consistency via unlock or version check call. Only use the copied data if
/// these operations succeeded. Otherwise an algorithm restart is necessary.
///
/// In the current implementation, it is possible for a reader to be starved
/// indefinitely.
///
/// ## Write protocol
///
/// After a successful write lock acquisition by
/// unodb::optimistic_lock::write_guard(), the protected data may be accessed
/// freely, as if under a regular write lock, with the exception of data
/// deletion, discussed below. The write lock object is a C++ scope guard which
/// will unlock on leaving the scope.
///
/// Since read locking does not write to the shared memory, readers can have
/// active pointers to the data without the writer knowing about them.
/// Therefore, lock-protected heap data cannot be deallocated immediately.
/// Instead of immediate deallocation, the data is marked as obsolete
/// (unodb::optimistic_lock::write_guard::write_unlock_and_obsolete) and
/// reclaimed later when it is safe to do so. This is implemented by \ref qsbr.
///
/// ## Internals
///
/// A lock is a single machine word, that encodes locked-unlocked state,
/// obsolete state, and version number.
///
/// Locking for write (unodb::optimistic_lock::write_guard())
/// atomically sets the locked state and bumps the version number.
///
/// Locking for read (unodb::optimistic_lock::try_read_lock()) saves the version
/// number at the time, and unlocking for read
/// (unodb::optimistic_lock::read_critical_section::try_read_unlock()) checks
/// whether the lock version did not advance since the read lock. It is also
/// possible to check this in a middle of an RCS
/// (unodb::optimistic_lock::read_critical_section::check()), which has exactly
/// the same semantics under a different name for descriptive code.
///
/// A lock in obsolete state marks data which is on the deallocation backlog to
/// be freed once all the thread epochs have advanced. All algorithms must
/// immediately stop retrying read locking such data and restart.
///
/// ## Memory model
///
/// The data races are prevented by implementing the Figure 6 method from
/// Boehm's paper (see Literature section below for the reference):
/// \code{.cpp}
/// const auto ver0 = lock_version.load(std::memory_order_acquire);
/// const auto data0 = data0.load(std::memory_order_relaxed);
/// const auto data1 = data1.load(std::memory_order_relaxed);
/// std::atomic_thread_fence(std::memory_order_acquire);
/// const auto ver1 = lock_version.load(std::memory_order_relaxed);
/// if (ver0 == ver1 && is_free(ver1)) {
///   // OK to act on data0 and data1
/// } else {
///   // Restart
/// }
/// \endcode
///
/// ## Literature
///
/// Based on the design from:
/// - V. Leis et al., "The ART of Practical Synchronization," DaMoN 2016, for
///   the algorithms.
/// - H. Boehm, "Can seqlocks get along with programming language memory
///   models?", MSPC 2012, for the critical section data access memory ordering
///   rules.
///
/// The optimistic lock is also similar to Linux kernel sequential locks with
/// the addition of an obsolete state for data marked for reclamation.
/// \}

// Should be the first include
#include "global.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <thread>
#include <tuple>
#include <type_traits>

#ifdef UNODB_DETAIL_X86_64
#include <emmintrin.h>
#endif

#include "assert.hpp"

namespace unodb {

/// Optimistic spinlock wait loop algorithm implementation. The implementation
/// is selected by #UNODB_DETAIL_SPINLOCK_LOOP_VALUE, set by CMake, and can be
/// either #UNODB_DETAIL_SPINLOCK_LOOP_PAUSE or
/// #UNODB_DETAIL_SPINLOCK_LOOP_EMPTY
// TODO(laurynas): move to unodb::detail namespace
// LCOV_EXCL_START
inline void spin_wait_loop_body() noexcept {
#if UNODB_SPINLOCK_LOOP_VALUE == UNODB_DETAIL_SPINLOCK_LOOP_PAUSE

#if defined(UNODB_DETAIL_X86_64)
  _mm_pause();
#elif defined(__aarch64__)
  __asm__ __volatile__("yield\n");
#else
#error Needs porting
#endif

#elif UNODB_SPINLOCK_LOOP_VALUE == UNODB_DETAIL_SPINLOCK_LOOP_EMPTY

  // Empty

#else  // UNODB_SPINLOCK_LOOP_VALUE

#error Unknown SPINLOCK_LOOP value in CMake

#endif  // UNODB_SPINLOCK_LOOP_VALUE
}
// LCOV_EXCL_STOP

/// Underlying integer type used to store optimistic lock word, including its
/// version and lock state information.
//
// TODO(laurynas) can we use optimistic_lock::version_type instead?
using version_tag_type = std::uint64_t;

/// Version-based optimistic lock that supports single-writer/multiple-readers
/// concurrency without shared memory writes during read operations.
///
/// Writers bump the version counter and readers detect concurrent writes by
/// comparing the version counter before and after the reads. Instances are
/// non-copyable and non-moveable.
///
/// See \ref optimistic-lock for usage examples and protocols.
///
/// To support reusing the same code for single-threaded context too, there is a
/// no-op counterpart: unodb::fake_optimistic_lock, enabling templatizing on the
/// lock type and passing either class as needed.
class [[nodiscard]] optimistic_lock final {
 public:
  /// Non-atomic lock word representation. Used for copying and manipulating
  /// snapshots of the atomic lock word.
  /// The lock word consists of:
  /// - Bit 0: obsolete state. If set, all other bits are zero.
  /// - Bit 1: write lock
  /// - Bits 2-63: version counter
  // TODO(laurynas): rename to lock_word
  class [[nodiscard]] version_type final {
   public:
    /// Lock word value constant in the obsolete state.
    static constexpr version_tag_type obsolete_lock_word = 1U;

    /// Create a new lock word from a raw \a version_val value.
    explicit constexpr version_type(version_tag_type version_val) noexcept
        : version{version_val} {}

    /// Return whether the lock word has the write lock bit set.
    // TODO(laurynas): introduce the precondition of not being obsolete
    [[nodiscard, gnu::const]] constexpr bool is_write_locked() const noexcept {
      return (version & 2U) != 0U;
    }

    /// Return whether the lock word indicates a free lock that is available for
    /// acquisition - neither write-locked nor obsolete.
    [[nodiscard, gnu::const]] constexpr bool is_free() const noexcept {
      return (version & 3U) == 0U;
    }

    /// Return whether the lock word is in the obsolete state.
    // Force inline because LLVM 14-17 and possibly later versions generate a
    // call to outline version from optimistic_lock::try_lock in release build
    // with UBSan. That same method is apparently miscompiled in that its loop
    // only checks whether the lock is free but never if it's obsolete,
    // resulting in hangs. Forcing to inline seems to make that issue to go away
    // too.
    [[nodiscard, gnu::const]] UNODB_DETAIL_FORCE_INLINE constexpr bool
    is_obsolete() const noexcept {
      return version == obsolete_lock_word;
    }

    /// Return a lock word with the current version and lock bit set.
    /// \pre the lock word must be free.
    [[nodiscard, gnu::const]] constexpr version_type set_locked_bit()
        const noexcept {
      UNODB_DETAIL_ASSERT(is_free());
      return version_type{version + 2};
    }

    /// Return the version_tag_type (just the data, including both the version
    /// and the write lock / obsolete bits).
    // TODO(laurynas): will go away once this class is used directly instead of
    // version_tag_type?
    [[nodiscard]] constexpr version_tag_type get() const noexcept {
      return version;
    }

    /// Compare two lock words for equality, including all the version and lock
    /// / obsolete bits.
    [[nodiscard]] constexpr bool operator==(version_type other) const noexcept {
      return version == other.version;
    }

    /// Output the lock word to \a os output stream. Should only be used
    /// for debug dumping.
    [[gnu::cold]] UNODB_DETAIL_NOINLINE void dump(std::ostream& os) const {
      os << "version = 0x" << std::hex << std::setfill('0') << std::setw(8)
         << version << std::dec;
      if (is_write_locked()) os << " (write locked)";
      if (is_obsolete()) os << " (obsoleted)";
    }

   private:
    /// Raw lock word value.
    version_tag_type version{0};
  };  // class version_type

 private:
  /// Atomic lock word and its operations.
  class [[nodiscard]] atomic_version_type final {
   public:
    /// Atomically load the lock word with acquire memory ordering.
    [[nodiscard]] version_type load_acquire() const noexcept {
      return version_type{version.load(std::memory_order_acquire)};
    }

    /// Atomically load the lock word with relaxed memory ordering.
    [[nodiscard]] version_type load_relaxed() const noexcept {
      return version_type{version.load(std::memory_order_relaxed)};
    }

    /// Atomically compare-and-exchange the lock word with acquire ordering on
    /// success. May not fail spuriously.
    /// \param[in] expected The expected current lock word value
    /// \param[in] new_val The new lock word value to set
    /// \return true if the exchange was successful
    [[nodiscard]] bool cas_acquire(version_type expected,
                                   version_type new_val) noexcept {
      auto expected_val = expected.get();
      return UNODB_DETAIL_LIKELY(version.compare_exchange_strong(
          expected_val, new_val.get(), std::memory_order_acquire,
          std::memory_order_relaxed));
    }

    /// Atomically clear the write lock bit with release memory ordering.
    /// The version number is preserved.
    /// \pre The write lock bit must be set.
    void write_unlock() noexcept {
      // This thread has written the previous lock word value, and no other
      // thread may write it before the unlock, thus we can read it without
      // ordering.
      const auto old_lock_word = load_relaxed();
      UNODB_DETAIL_ASSERT(old_lock_word.is_write_locked());

      const auto new_lock_word = old_lock_word.get() + 2;
      version.store(new_lock_word, std::memory_order_release);
    }

    /// Atomically clear the set write lock bit and set the obsolete bit with
    /// release memory ordering.
    /// \pre The obsolete bit must be clear
    /// \pre The write lock bit must be set
    void write_unlock_and_obsolete() noexcept {
#ifndef NDEBUG
      const auto old_lock_word{load_relaxed()};
      UNODB_DETAIL_ASSERT(!old_lock_word.is_obsolete());
      UNODB_DETAIL_ASSERT(old_lock_word.is_write_locked());
#endif

      version.store(version_type::obsolete_lock_word,
                    std::memory_order_release);

      UNODB_DETAIL_ASSERT(load_relaxed().is_obsolete());
    }

   private:
    /// Raw atomic lock word.
    std::atomic<std::uint64_t> version;

    static_assert(decltype(version)::is_always_lock_free,
                  "Must use always lock-free atomics");
  };  // class atomic_version_type

 public:
  class write_guard;

  /// Read critical section (RCS) that stores the lock version at the read lock
  /// time and checks it against the current version for consistent reads.
  /// Instances are non-copyable and only movable with the move constructor.
  ///
  /// There are three different states for an RCS:
  /// 1. The lock was in obsolete state when the RCS was returned by
  ///    optimistic_lock::try_read_lock(). This must always be checked for after
  ///    the RCS has been created with a read_critical_section::must_restart()
  ///    call.
  /// 2. The RCS was acquired and no newer write-locking has been detected for
  ///    the underlying lock.
  /// 3. The RCS was unlocked or the underlying lock has been write-locked since
  ///    the RCS was created, and this has been detected by a try_read_unlock()
  ///    or check() call. The RCS is no longer valid.
  ///
  /// To support reusing the same code for single-threaded contexts too, there
  /// is a no-op counterpart: unodb::fake_read_critical_section, enabling
  /// templatizing on the RCS type and passing the either class as needed.
  ///
  /// Internally the obsolete state (and in the debug builds, the unlocked /
  /// underlying lock write locked state too) is represented by
  /// read_critical_section::lock == nullptr.
  class [[nodiscard]] read_critical_section final {
   public:
    /// Default-construct an invalid RCS. The resulting RCS may only be
    /// destructed or another RCS may be move-assigned to it. Typically used as
    /// a destination for move assignment.
    read_critical_section() noexcept = default;

    /// Construct an RCS for \a lock_ read-locked at specific \a version_. Users
    /// should not call this directly. Use optimistic_lock::try_read_lock() or
    /// optimistic_lock::rehydrate_read_lock() instead.
    // TODO(laurynas): hide this constructor from users with C++ access rules.
    read_critical_section(optimistic_lock& lock_,
                          version_type version_) noexcept
        : lock{&lock_}, version{version_} {}

    /// Destruct an RCS.
    ~read_critical_section() noexcept {
      // TODO(laurynas): figure out why not all the paths have
      // called try_read_unlock first, if possible assert that lock is nullptr.

      // If the destructor ever starts doing something in the release build,
      // reset moved-from lock fields in the move and write_guard constructors.
#ifndef NDEBUG
      if (lock != nullptr) std::ignore = lock->try_read_unlock(version);
#endif
    }

    /// Move \a other RCS into this one.
    read_critical_section& operator=(read_critical_section&& other) noexcept {
#ifndef NDEBUG
      // Release the old RCS if it was still valid.
      if (lock != nullptr) std::ignore = lock->try_read_unlock(version);
#endif
      lock = other.lock;
      // The current implementation does not need lock == nullptr in the
      // destructor, thus only reset other.lock in debug builds
#ifndef NDEBUG
      other.lock = nullptr;
#endif
      version = other.version;

      return *this;
    }

    /// Check whether this RCS was not constructed on an obsolete lock, must
    /// be called first thing after creating the RCS. In the case of failed
    /// check this RCS may only be destructed or another RCS may be
    /// move-assigned to it.
    ///
    /// \retval true if the lock was obsolete at the time that the RCS was
    /// obtained.
    [[nodiscard]] bool must_restart() const noexcept {
      return UNODB_DETAIL_UNLIKELY(lock == nullptr);
    }

    /// Check whether this RCS is still valid. If the RCS is found to be
    /// invalid, it may only be destructed or another RCS may be move-assigned
    /// to it.
    ///
    /// \pre read_critical_section::must_restart must have returned false on
    /// this RCS to check whether it was not created on a lock in obsolete
    /// state.
    ///
    /// \retval true The underlying lock is at the same version it was at the
    /// RCS creation time, all read protected data is consistent.
    /// \retval false The underlying lock has advanced since the RCS creation
    /// or last check time, indicating a write lock, any data read since then
    /// must be discarded.
    ///
    /// The return value is determined by comparing
    /// read_critical_section::version with the current lock version. If the
    /// versions don't match and the RCS is no longer valid, a debug build will
    /// reset read_critical_section::lock pointer to `nullptr`, causing
    /// subsequent use attempts of the RCS to fault.
    [[nodiscard]] bool check() const noexcept {
      const auto result = lock->check(version);
#ifndef NDEBUG
      if (UNODB_DETAIL_UNLIKELY(!result)) lock = nullptr;
#endif
      return UNODB_DETAIL_LIKELY(result);
    }

    /// Check one last time whether this RCS is still valid and unlock it.
    /// The RCS is no longer valid after this call and may only be destructed or
    /// another RCS may be move-assigned to it.
    ///
    /// \pre read_critical_section::must_restart must have returned false on
    /// this RCS to check whether it was not created on a lock in obsolete
    /// state.
    ///
    /// \retval true The underlying lock is at the same version it was at the
    /// RCS creation time, all read protected data is consistent.
    /// \retval false The underlying lock has advanced since the RCS creation
    /// or last check time, indicating a write lock, any data read since then
    /// must be discarded.
    ///
    /// The return value is determined by comparing
    /// read_critical_section::version with the current lock version. In a debug
    /// build, read_critical_section::lock pointer is reset to `nullptr`,
    /// causing subsequent use attempts of the RCS to fault.
    [[nodiscard, gnu::flatten]] UNODB_DETAIL_FORCE_INLINE bool try_read_unlock()
        const noexcept {
      const auto result = lock->try_read_unlock(version);
#ifndef NDEBUG
      lock = nullptr;
#endif
      return UNODB_DETAIL_LIKELY(result);
    }

    /// Return the lock version when this RCS was created.
    [[nodiscard]] constexpr version_tag_type get() const noexcept {
      return version.get();
    }

    read_critical_section(const read_critical_section&) = delete;
    read_critical_section(read_critical_section&&) = delete;
    read_critical_section& operator=(const read_critical_section&) = delete;

   private:
    /// Lock backing this RCS.
#ifndef NDEBUG
    mutable
#endif
        optimistic_lock* lock{nullptr};

    /// Lock version at the RCS creation time. Immutable throughout the RCS
    /// lifetime.
    version_type version{0};

    friend class write_guard;
  };  // class read_critical_section

  /// Write guard (WG) for exclusive access protection. Functions as a scope
  /// guard if needed. Can only be created by attempting to upgrade a
  /// optimistic_lock::read_critical_section. Instances are non-copyable and
  /// non-movable.
  ///
  /// There are two different states for a WG:
  /// 1. Active: the lock version at upgrade time matched the RCS version. The
  ///    WG holds the write lock.
  /// 2. Inactive: either the upgrade failed due to concurrent write lock, or
  ///    one of the write unlock methods has already been called. An inactive WG
  ///    may only be destructed.
  ///
  /// Internally the active and inactive states are represented by
  /// write_guard::lock pointing to a lock or being `nullptr` respectively.
  class [[nodiscard]] write_guard final {
   public:
    /// Create a write guard by attempting to upgrade a read \a
    /// critical_section, which is consumed in process. The upgrade succeeds if
    /// the RCS lock version equals the current lock version.
    /// \note write_guard::must_restart must be called on the created instance
    /// to check for success.
    explicit write_guard(read_critical_section&& critical_section) noexcept
        : lock{try_lock_upgrade(std::move(critical_section))} {}

    /// Create an inactive write guard (no lock held).
    write_guard() noexcept : lock{nullptr} {}

    /// Unlock if needed and destruct the write guard.
    ~write_guard() noexcept {
      if (lock == nullptr) return;
      lock->write_unlock();
    }

    /// Check whether this write guard failed to acquire the write lock. Must be
    /// called after construction and before the first protected data access.
    /// \retval true if the lock upgrade failed and this WG is inactive.
    [[nodiscard]] bool must_restart() const noexcept {
      return UNODB_DETAIL_UNLIKELY(lock == nullptr);
    }

    /// Write unlock and make obsolete the underlying lock, deactivating this
    /// write guard. Only destruction is legal after this call.
    /// \pre The write guard must be active (write_guard::must_restart returned
    /// false).
    void unlock_and_obsolete() noexcept {
      lock->write_unlock_and_obsolete();
      lock = nullptr;
    }

    /// Write unlock the underlying lock, deactivating this write guard. Only
    /// destruction is legal after this call.
    /// \pre The write guard must be active (write_guard::must_restart returned
    /// false).
    void unlock() noexcept {
      lock->write_unlock();
      lock = nullptr;
    }

#ifndef NDEBUG
    /// Check whether this write guard is active.
    [[nodiscard]] bool active() const noexcept { return lock != nullptr; }

    /// Check whether this write guard holds a write lock on \a lock_
    [[nodiscard]] bool guards(const optimistic_lock& lock_) const noexcept {
      return lock == &lock_;
    }
#endif

    write_guard(const write_guard&) = delete;
    write_guard(write_guard&& other) noexcept : lock{other.lock} {
      other.lock = nullptr;
    }
    write_guard& operator=(const write_guard&) = delete;
    write_guard& operator=(write_guard&& other) noexcept {
      if (this != &other) {
        if (lock != nullptr) lock->write_unlock();
        lock = other.lock;
        other.lock = nullptr;
      }
      return *this;
    }

   private:
    /// Attempt to upgrade the underlying lock of \a critical_section to
    /// write-locked state. The RCS is consumed in the process.
    /// \return the write-locked lock if upgrade succeeded, `nullptr` if the RCS
    /// version did not match the current lock version
    [[nodiscard]] static optimistic_lock* try_lock_upgrade(
        read_critical_section&& critical_section) noexcept {
      const auto upgrade_success =
          critical_section.lock->try_upgrade_to_write_lock(
              critical_section.version);
      auto* const result = UNODB_DETAIL_LIKELY(upgrade_success)
                               ? critical_section.lock
                               : nullptr;
#ifndef NDEBUG
      critical_section.lock = nullptr;
#endif
      return result;
    }

    /// Underlying lock. If `nullptr`, this WG is inactive.
    optimistic_lock* lock{nullptr};
  };  // class write_guard

  /// Construct a new optimistic lock.
  optimistic_lock() noexcept = default;

  /// Destruct the lock, trivially.
  ~optimistic_lock() noexcept = default;

  /// Acquire and return an optimistic_lock::read_critical_section for this
  /// lock. This is done without writing anything on the lock, but it will spin
  /// if the lock is write-locked. It will return immediately if the lock is in
  /// obsolete state. In debug builds, this will maintain the open RCS counter.
  ///
  /// \note read_critical_section::must_restart must be called before the first
  /// protected data access to check for obsolete state.
  [[nodiscard]] read_critical_section try_read_lock() noexcept {
    while (true) {
      const auto current_version = version.load_acquire();
      if (UNODB_DETAIL_LIKELY(current_version.is_free())) {
        inc_read_lock_count();
        return read_critical_section{*this, current_version};
      }
      // LCOV_EXCL_START
      if (UNODB_DETAIL_UNLIKELY(current_version.is_obsolete()))
        return read_critical_section{};
      UNODB_DETAIL_ASSERT(current_version.is_write_locked());
      spin_wait_loop_body();
      // LCOV_EXCL_STOP
    }
  }

  /// Create an optimistic_lock::read_critical_section using a previously saved
  /// \a version_tag. Used for restoring OLC iterator state. It does not do any
  /// spin waits or even look at the current lock version. When the caller calls
  /// read_critical_section::check() on the returned lock they will figure out
  /// whether or not the version is still valid. In debug builds, this will
  /// maintain the open RCS counter.
  [[nodiscard]] read_critical_section rehydrate_read_lock(
      version_tag_type version_tag) noexcept {
    // TODO(laurynas) The inc_read_lock_count call should be refactored to a
    // RCS-creating factory method in optimistic_lock, removing the need for
    // this comment and cleaning up usage.
    inc_read_lock_count();
    return read_critical_section{*this, version_type(version_tag)};
  }

#ifndef NDEBUG
  /// Assert that this lock has no open optimistic_lock::read_critical_section
  /// instances. Used in debug builds at lock heap deallocation time.
  void check_on_dealloc() const noexcept {
    UNODB_DETAIL_ASSERT(read_lock_count.load(std::memory_order_acquire) == 0);
  }

  /// In debug builds, check whether this lock is in obsolete state and that it
  /// was this thread that obsoleted it.
  [[nodiscard]] bool is_obsoleted_by_this_thread() const noexcept {
    return version.load_acquire().is_obsolete() &&
           std::this_thread::get_id() == obsoleter_thread;
  }

  /// In debug builds, check whether this lock is write locked.
  [[nodiscard]] bool is_write_locked() const noexcept {
    return version.load_acquire().is_write_locked();
  }
#endif

  /// Output the lock representation to \a os output stream. Should only be used
  /// for debug dumping.
  [[gnu::cold]] UNODB_DETAIL_NOINLINE void dump(std::ostream& os) const {
    const auto dump_version = version.load_acquire();
    os << "lock: ";
    dump_version.dump(os);
#ifndef NDEBUG
    os << " current read lock count = "
       << read_lock_count.load(std::memory_order_acquire);
#endif
  }

  optimistic_lock(const optimistic_lock&) = delete;
  optimistic_lock(optimistic_lock&&) = delete;
  optimistic_lock& operator=(const optimistic_lock&) = delete;
  optimistic_lock& operator=(optimistic_lock&&) = delete;

 private:
  /// Check if the current lock version has not changed since \a
  /// locked_version. Act as a read unlock if the check fails.
  /// \pre At least one read lock must exist
  [[nodiscard]] bool check(version_type locked_version) const noexcept {
    UNODB_DETAIL_ASSERT(read_lock_count.load(std::memory_order_acquire) > 0);

#ifndef UNODB_DETAIL_THREAD_SANITIZER
    std::atomic_thread_fence(std::memory_order_acquire);
#endif
    const auto result{locked_version == version.load_relaxed()};
#ifndef NDEBUG
    if (UNODB_DETAIL_UNLIKELY(!result)) dec_read_lock_count();
#endif
    return UNODB_DETAIL_LIKELY(result);
  }

  /// Try to read unlock this lock by comparing the current version with
  /// \a locked_version. Since read locking and unlocking does not affect the
  /// shared lock state, this only checks whether the lock version is equal to
  /// \a locked_version.
  /// \retval true if the read unlock succeeded
  /// \retval false if the current lock version has advanced since the read lock
  /// was taken
  [[nodiscard, gnu::flatten]] UNODB_DETAIL_FORCE_INLINE bool try_read_unlock(
      version_type locked_version) const noexcept {
    const auto result{check(locked_version)};
#ifndef NDEBUG
    if (UNODB_DETAIL_LIKELY(result)) dec_read_lock_count();
#endif
    return UNODB_DETAIL_LIKELY(result);
  }

  /// Try to write lock by atomically setting the lock bit while verifying the
  /// version matches \a locked_version. Acts as a read unlock if unsuccessful.
  /// \retval true if the write lock succeeded
  /// \retval false if the current lock version has advanced since the read lock
  /// was taken
  [[nodiscard]] bool try_upgrade_to_write_lock(
      version_type locked_version) noexcept {
    const auto result{
        version.cas_acquire(locked_version, locked_version.set_locked_bit())};
    dec_read_lock_count();
    return UNODB_DETAIL_LIKELY(result);
  }

  /// Write unlock this lock.
  /// \pre The lock must be write-locked.
  void write_unlock() noexcept { version.write_unlock(); }

  /// Atomically write unlock and obsolete this lock.
  /// \pre The lock must be write-locked.
  void write_unlock_and_obsolete() noexcept {
    version.write_unlock_and_obsolete();
#ifndef NDEBUG
    obsoleter_thread = std::this_thread::get_id();
#endif
  }

  /// Atomic lock word.
  atomic_version_type version{};

#ifndef NDEBUG
  /// In debug builds, the counter of currently-active read locks.
  mutable std::atomic<std::int64_t> read_lock_count{0};

  /// In debug builds, the ID of the thread which obsoleted this lock.
  std::thread::id obsoleter_thread{};
#endif

  /// In debug builds, increment the read lock counter, no-op in release builds.
  void inc_read_lock_count() const noexcept {
#ifndef NDEBUG
    read_lock_count.fetch_add(1, std::memory_order_release);
#endif
  }

  /// In debug builds, decrement the read lock counter, no-op in release builds.
  void dec_read_lock_count() const noexcept {
#ifndef NDEBUG
    const auto old_value =
        read_lock_count.fetch_sub(1, std::memory_order_release);
    UNODB_DETAIL_ASSERT(old_value > 0);
#endif
  }
};  // class optimistic_lock

static_assert(std::is_standard_layout_v<optimistic_lock>);
static_assert(std::is_trivially_destructible_v<optimistic_lock>);
static_assert(std::is_nothrow_destructible_v<optimistic_lock>);

#ifdef NDEBUG
static_assert(sizeof(optimistic_lock) == 8);
#else
static_assert(sizeof(optimistic_lock) == 24);
#endif

/// Gloss for the atomic semantics used to guard loads and stores. Wraps the
/// protected data fields. The loads and stores become relaxed atomic operations
/// as required by the optimistic lock memory model. The instances are
/// non-moveable and non-copy-constructable but the assignments both from the
/// wrapped values and plain value type are supported.
///
/// To support reusing the same code for single-threaded context too, there is a
/// no-op counterpart: unodb::in_fake_critical_section, enabling templatizing on
/// the wrapper type and passing either class as needed.
///
/// Implements the required set of transparent operations, extend as necessary.
template <typename T>
class [[nodiscard]] in_critical_section final {
 public:
  /// Default construct the wrapped \a T value.
  constexpr in_critical_section() noexcept = default;

  /// Construct the wrapped value from the passed \a value_.
  // cppcheck-suppress noExplicitConstructor
  // NOLINTNEXTLINE(google-explicit-constructor,hicpp-explicit-conversions)
  constexpr in_critical_section(T value_) noexcept : value{value_} {}

  /// Destruct the wrapped value.
  ~in_critical_section() noexcept = default;

  /// Copy-assign another wrapped value.
  // NOLINTNEXTLINE(cert-oop54-cpp)
  in_critical_section& operator=(
      const in_critical_section& new_value) noexcept {
    store(new_value.load());
    return *this;
  }

  /// Assign \a new_value to the wrapped value.
  in_critical_section& operator=(T new_value) noexcept {
    store(new_value);
    return *this;
  }

  /// Pre-increment the wrapped value.
  void operator++() noexcept { store(load() + 1); }

  /// Pre-decrement the wrapped value.
  void operator--() noexcept {
    // The cast silences MSVC diagnostics about signed/unsigned mismatch.
    store((static_cast<T>(load() - 1)));
  }

  /// Post-decrement the wrapped value. Returns the old unwrapped value.
  // NOLINTNEXTLINE(cert-dcl21-cpp)
  T operator--(int) noexcept {
    const auto result = load();
    store(result - 1);
    return result;
  }

  /// Checks whether the wrapped pointer is `nullptr`.
  [[nodiscard]] bool operator==(std::nullptr_t) const noexcept {
    return load() == nullptr;
  }

  /// Convert to the wrapped value, implicitly if needed.
  // NOLINTNEXTLINE(google-explicit-constructor,hicpp-explicit-conversions)
  operator T() const noexcept { return load(); }

  /// Explicitly read the wrapped value.
  [[nodiscard]] T load() const noexcept {
    return value.load(std::memory_order_relaxed);
  }

  /// Explicitly assign the wrapped value from \a new_value.
  void store(T new_value) noexcept {
    value.store(new_value, std::memory_order_relaxed);
  }

  in_critical_section(const in_critical_section&) = delete;
  in_critical_section(in_critical_section&&) = delete;
  void operator=(in_critical_section&&) = delete;

 private:
  /// Wrapped value.
  std::atomic<T> value;

  static_assert(std::atomic<T>::is_always_lock_free,
                "Must use always lock-free atomics");
};  // class in_critical_section

}  // namespace unodb

#endif  // UNODB_DETAIL_OPTIMISTIC_LOCK_HPP
