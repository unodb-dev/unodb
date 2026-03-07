<!--- -*- gfm -*- -->

# unodb

[![Github Actions Build
Status](https://github.com/laurynas-biveinis/unodb/workflows/build/badge.svg)](https://github.com/laurynas-biveinis/unodb/actions?query=workflow%3Abuild)
[![codecov](https://codecov.io/gh/laurynas-biveinis/unodb/branch/master/graph/badge.svg)](https://codecov.io/gh/laurynas-biveinis/unodb)
[![GitHub
Super-Linter](https://github.com/laurynas-biveinis/unodb/workflows/Super-Linter/badge.svg)](https://github.com/marketplace/actions/super-linter)
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=laurynas-biveinis_unodb&metric=alert_status)](https://sonarcloud.io/dashboard?id=laurynas-biveinis_unodb)

## Introduction

UnoDB is a library that implements of Adaptive Radix Tree (ART) data structure,
designed for efficient indexing in main-memory databases. The ART is a
trie-based data structure that dynamically adapts to the distribution of keys,
ensuring good search performance and memory usage. UnoDB offers two variants of
ART: a regular one and a concurrent one based on Optimistic Lock Coupling (OLC)
with Quiescent State Based Reclamation (QSBR).

This library serves as a playground for experimenting with various C++ tools and
ideas. I am describing some of the things I learned at my [blog](https://of-code.blogspot.com/search/label/art).

## Requirements

UnoDB source code is written in C++20, and relies on the following
platform-specific features:

- On Intel platforms, it requires SSE4.1 intrinsics (Nehalem and higher), AVX
  for MSVC builds, and optional AVX2 support, when available. This differs from
  the original ART paper, which only required SSE2.
- On ARM, it uses NEON intrinsics.

### Build dependencies

- Earliest versions of supported compilers: GCC 10, LLVM 11, Xcode 16.1,
  MSVC 2022. Open an issue if you require support for an older version.
- CMake, at least 3.16.
- Boost library. It is a mandatory dependency for Boost.Container
  (small_vector). If building with statistics counters, Boost.Accumulator is
  also required. Boost.Stacktrace is an optional dependency.

### Optional vendored dependencies, bundled as Git submodules

- Google Test for tests.
- Google Benchmark for microbenchmarks.
- [DeepState][deepstate] for fuzzing tests.

These dependencies need not be present if the build is configured to skip the
corresponding part. For example, if the CMake option `-DTESTS=OFF` is given,
then Google Test and DeepState submodules don't have to be populated.

## Building

Unless you configure your build otherwise, first you need to populate the Git
submodules for test and benchmark dependencies:

```bash
# --recursive is not strictly required at the moment, but a good habit to have
git submodule update --init --recursive
```

Out-of-source builds are recommended, for example

```bash
mkdir build
cd build
cmake .. <other options, see below>
```

There are some CMake options for users:

- `-DSPINLOCK_LOOP=PAUSE|EMPTY` to choose the spinlock wait loop body
  implementation for the optimistic lock. `EMPTY` may benchmark better as long
  as there are fewer threads than available CPU cores. `PAUSE` will use that
  instruction on x86_64, and something similar on ARM. The default is `PAUSE`.
- `-DSTATS=OFF` if you want to compile away all the statistics counters. The
  result will scale better in benchmarks. The current stats implementation is
  global cache line-padded shared atomic counters. This option might be removed
  in the future if the stats are reimplemented with less overhead.
- `-DWITH_AVX2=OFF` to disable AVX2 intrinsics to use SSE4.1/AVX only.
- `-DTESTS=OFF` to skip building the tests.
- `-DBENCHMARKS=ON` to build the benchmarks.

There are other CMake options that are mainly intended for UnoDB development
itself and are discussed in [CONTRIBUTING.md](CONTRIBUTING.md).

## Platform-Specific Notes

### Ubuntu 22.04

```bash
# libc6-dev-i386 is for DeepState
sudo apt-get install -y libboost-dev libc6-dev-i386
```

### Amazon Linux 2023

```bash
sudo dnf install git gcc g++ cmake boost-devel
# Optional, if you want to use Boost.Stacktrace:
git clone https://github.com/ianlancetaylor/libbacktrace
(cd libbacktrace && ./configure && make && sudo make install)
# Build as usual
```

### Amazon Linux 2

```bash
sudo yum install git gcc10 gcc10-c++ cmake3 boost-devel
# Pass -DCMAKE_C_COMPILER=gcc10-gcc -DCMAKE_CXX_COMPILER=gcc10-c++ to cmake3
#
# Benchmarking. For jemalloc, consider building a newer version from source.
sudo amazon-linux-extras install -y epel
sudo yum install jemalloc perf
```

## Usage

See `examples/` directory for simple usage examples. This directory is a
top-level CMake project.

All the declarations live in the `unodb` namespace, which is omitted in the
descriptions below.

The key type is a template argument for the `unodb::db` classes. In general,
the library supports both integral keys (though only `std::uint64_t` is tested
at this time) and variable length keys (using `unodb::key_view` as the key
type). Variable length keys are supported using the `unodb::key_encoder` and
`unodb::key_decoder`. The `unodb::key_encoder` is responsible for making
encoded keys which obey lexicographic ordering and handles various signed and
unsigned types, floating point, types, and text. The `unodb::key_encoder` MUST
be used for text data (including Unicode sort keys) and for compound keys (keys
consisting of multiple components). The ART data structure has a restriction
that no full length key may be a prefix of another key. This restriction is
trivially satisfied for any fixed width key. Also per the ART paper, the
`unodb::key_encoder` maintains this contract for text keys by truncating them to
not more than `unodb::key_encoder::maxlen` bytes and then logically padding them
out (with a run length counter) to `unodb::key_encoder::maxlen`. Unicode data
SHOULD be converted by the caller using a quality library (e.g., ICU) to Unicode
sort keys which capture the desired collation order, and those sort keys then
passed to the `unodb::key_encoder`. Finally, the `unodb::key_decoder` may be
used to decode signed and unsigned integral types and floating point types. Note
that all NaNs are mapped to a canonical NaN during encoding, so decode of NaN
always returns that canonical value. Decode of Unicode sort keys is not
possible due to the transformation to generate the sort key. Decode of other
text can run into the truncation and run length padding artifacts and is not
supported. When ART is used as a secondary index, the caller stores the record
identifier (or record pointer) as the values in the tree. The original text can
then be recovered from the source record.

Values are treated opaquely. For `unodb::db`, they are passed as non-owning
objects of `value_view` (a `std::span<std::byte>`), and insertion copies them
internally. The same applies for `get`, which returns a non-owning `value_view`.
For `unodb::olc_db`, `get` returns a `qsbr_value_view`, a `span` guaranteed to
remain valid until the current thread passes through a quiescent state.

All ART classes share the same API:

- constructor.
- `get(key k)` returns `get_result` (a `std::optional<value_view>`).
- `bool insert(key k, value_view v)` returns whether the insert was successful
  (i.e. the key was not already present).
- `bool remove(key k)` returns whether delete was successful (i.e. the key was
  found in the tree).
- `clear()` empties the tree. For `olc_db`, it must be called from a
  single-threaded context.
- `bool empty()` returns whether the tree is empty.
- `template <FN> scan(), scan_from(...), scan_range()` a family of
  scan methods, including forward and reverse scans and scans with a
  from_key and an exclusive upper bound to_key where fn is a lambda
  accepting a visitor and returning a bool indicating whether the scan
  should halt (bool halt). See `examples/examples_art.cpp`.
- `void dump(std::ostream &)` outputs the tree representation.
- Several getters provide tree info, such as current memory use, and internal
  operation counters (e.g. number of times Node4 grew to Node16, key prefix was
  split, etc - see the source code for details).

Three ART classes available:

- `db`: unsynchronized ART tree, for single-thread contexts or with
  external synchronization
- `mutex_db`: ART tree with single global mutex synchronization
- `olc_db`: a concurrent ART tree, implementing Optimistic Lock Coupling as
  described in "The ART of Practical Synchronization" paper by Leis et al.;
  nodes are versioned, writers lock per-node optimistic locks, readers don't
  lock but check node versions and restart if they change.

The full source code reference, including public API, is available at
[https://docs.unodb.dev](https://docs.unodb.dev) (work in progress).

Do not use macros starting with `UNODB_DETAIL_` or declarations in
`unodb::detail` and `unob::test` namespaces as they are internal and may change
at any time.

## Technical Details

### Adaptive Radix Tree

The implementation follows the paper description closely, with the following
differences and design choices:

- The paper algorithms are specified in SSE2 intrinsics. This implementation has
  SSE4.1 as the minimal level, and AVX2 as the default one on Intel. On ARM,
  NEON is used.
- Different ways to implement leaf nodes are discussed in the paper
  (single-value leaves, multi-value leaves, and combined pointer/value slots).
  Here single-value leaves are implemented.
- The paper discusses different choices in implementing search key path
  compression. Here the pessimistic path compression is implemented, with up to
  7 bytes of key data per internal node.

### Sequential Lock

The optimistic lock concept seems to be nearly identical to that of [sequential
locks][seqlock] as used in the Linux kernel, with the addition of "obsolete"
state. The lock implementation uses the seqlock memory model implementation as
described by Boehm in "Can seqlocks get along with programming language memory
models?" 2012 paper.

### Quiescent State-Based Reclamation (QSBR)

The OLC ART implementation necessitates a memory reclamation scheme as required
by lock-free data structures (even though `olc_db` is not lock-free in the
general sense, the readers do not take locks), and for that a Quiescent State
Based Reclamation (QSBR) was chosen. In QSBR, each thread periodically announces
that it is not using any pointers to a shared data structures. After all the
threads have done so, a new epoch starts, and the memory reclamation requests
from two epochs ago are executed. To participate in QSBR, a thread must an
instance of `unodb::qsbr_thread`, which derives from `std::thread`. A thread may
temporarily or permanently stop its participation in QSBR by calling
`unodb::this_thread().qsbr_pause()` and resume by calling
`unodb::this_thread().qsbr_resume()`.

The registered threads must periodically signal their quiescent states. They can
do this by using the `unodb::quiescent_state_on_scope_exit` scope guard, which
automatically reports the quiescent state when the scope is exited.

## Related Projects

[art_map](https://github.com/justinasvd/art_map) is a C++14 template library
providing `std::`-like interface over the ART data structure. It shares some
code with UnoDB.

## Contributing

Please see [CONTRIBUTING.md](CONTRIBUTING.md).

## Literature

_ART Trie_: V. Leis, A. Kemper and T. Neumann, "The adaptive radix tree: ARTful
indexing for main-memory databases," 2013 29th IEEE International Conference on
Data Engineering (ICDE 2013)(ICDE), Brisbane, QLD, 2013, pp. 38-49.
doi:10.1109/ICDE.2013.6544812

_ART Sync_: V. Leis, F. Schneiber, A. Kemper and T. Neumann, "The ART of
Practical Synchronization," 2016 Proceedings of the 12th International Workshop
on Data Management on New Hardware (DaMoN), pages 3:1--3:8, 2016.

_qsbr_: P. E. McKenney, J. D. Slingwine, "Read-copy update: using execution
history to solve concurrency problems," Parallel and Distributed Computing and
Systems, 1998, pages 509--518.

_seqlock sync_: H-J. Boehm, "Can seqlocks get along with programming language
memory models?," Proceedings of the 2012 ACM SIGPLAN Workshop on Memory Systems
Performance and Correctness, June 2012, pages 12--21, 2012.

[deepstate]: https://github.com/trailofbits/deepstate "DeepState on GitHub"
[seqlock]: https://en.wikipedia.org/wiki/Seqlock
