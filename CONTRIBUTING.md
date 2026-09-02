<!--- -*- gfm -*- -->

# Contributing to UnoDB

## Optional development dependencies

- clang-format
- lcov
- clang-tidy
- clangd
- cppcheck
- cpplint
- include-what-you-use
- libfuzzer

## General workflow

Clone the repository, create your feature or fix branches, work on them, commit,
open a pull request to this repository.

## Development CMake options

The regular CMake option `-DCMAKE_BUILD_TYPE` is recognized. Setting it to
`Debug` will enable a lot of assertions, which are useful during development.

There also other development-specific options. All of them are `OFF` by default.

- `-DSTANDALONE=ON` always should be given when working on UnoDB itself, whereas
  for users with UnoDB as a part of another project it should be `OFF`. When
  turned on, it will build benchmarks by default and enable extra global debug
  checks that require entire programs to be compiled with them. Currently, this
  consists of the libstdc++ debug mode.
- `-DMAINTAINER_MODE=ON` to enable maintainer diagnostics. This makes
  compilation warnings fatal.
- `-DSANITIZE_ADDRESS=ON` to enable AddressSanitizer (asan) and, if available,
  LeakSanitizer. It is incompatible with the `-DSANITIZE_THREAD=ON` option.
- `-DSANITIZE_THREAD=ON` to enable ThreadSanitizer (tsan). It is incompatible
  with the `-DSANITIZE_ADDRESS=ON` option, not available under MSVC, and will
  disable libfuzzer support if it would be enabled otherwise.
- `-DSANITIZE_UB=ON` to enable UndefinedBehaviorSanitizer (ubsan). It is
  compatible with other sanitizer options, although some [false
  positive][sanitizer-combination-bug] might occur. Not available under MSVC.
- `-DSTATIC_ANALYSIS=ON` for GCC or MSVC compiler static analysis. LLVM analyzer
  is used without any CMake option, see the "Linting and static analysis"
  section below.
- `-DIWYU=ON` to use include-what-you-use. It will take effect if building with
  clang.
- `-DCPPCHECK_AGGRESSIVE=ON` to enable inconclusive cppcheck diagnostics. They
  will not fail a build.
- `-DCOVERAGE=ON` to generate coverage reports on tests, excluding fuzzers,
  using lcov.

## Code organization

Header files can be public and internal. The internal ones have "internal"
somewhere in their name.

Public API must be introduced in `unodb` namespace, and any such declarations
should appear in a public header. It is OK for the implementation of a public
declaration to be in a private header, but see below. All the private
declarations for implementation details should live in `unodb::detail`
namespace. Thus, private headers may contain both `unodb` and `unodb::detail`
code, but rethink the header split if the former becomes a majority in a header.

Macros should not be a part of public API, and may be used internally only when
unavoidable. If a macro has to be introduced, its name must be prefixed with
`UNODB_DETAIL_`.

## Code style guide

- The code should follow existing conventions, formatted with
  [Google C++ style][gc++style]. This is enforced by GitHub Actions SuperLinter
  running clang-format, currently version 21.
- Each source file must have a `// Copyright <file-intro-year>-<last-edit-year>
UnoDB contributors` as the first line.
- Each source file must `#include "global.hpp"` first thing.
- Identifiers should be `snake_case`.
- Type names should either have no suffix, or `_type` when declaring types from
  template parameters and other identifiers. It cannot be `_t` as this suffix is
  reserved by POSIX.
- The code is `noexcept`-maximalist. Every function and method that cannot throw
  should be marked as `noexcept`. If the code does not throw in release build
  but may throw in debug one, it should ignore the latter and be `noexcept`.
- The code is exception-safe, with strong exception guarantee. This is actually
  tested by [test/test_art_oom.cpp](test/test_art_oom.cpp) injecting
  `std::bad_alloc` into heap allocations. New code should exception-safe too,
  with OOM tests expanded as necessary.
- The code is `[[nodiscard]]`-maximalist. Every value-returning function and
  method starts out as `[[nodiscard]]` by default, and is only changed if there
  is a clear need to both handle and ignore the return value.
- The code follows the Almost Always Auto guideline.
- `const` should be used everywhere it is possible to do so, with the exception
  of by-value function parameters and class fields that support moving from.
- `constexpr` (and `consteval`) should be applied everywhere it is legal to do
  so. Perhaps one day we will have a compile-time Adaptive Radix Tree.
- All C++ standard library symbols must be namespace-qualified, and this
  includes symbols shared with C. For example `std::size_t`.
- Automatic code formatting can be configured through Git clean/fuzz filters. To
  enable this feature, do `git config --local include.path ../.gitconfig`. If
  you need to temporarily disable it, run `git config --local --unset include.path`.
- The code that cannot be possibly tested by short deterministic tests, for
  example, because it handles rarely-occurring non-deterministic concurrency
  conditions, should be excluded from coverage testing with `// LCOV_EXCL_LINE`
  comment for a single line or with `// LCOV_EXCL_START`, `// LCOV_EXCL_STOP`
  start and end markers for a block.

## Documentation style guide

- The code should be commented, but without comments repeating already obvious
  code.
- `TODO` comments may be used for future tasks. `FIXME` comments should be used
  for things that must be fixed before the code lands in the master branch,
  however sometimes they land there. In both cases they should have a username
  in parentheses, i.e. `TODO(alice)`, `FIXME(bob)`. It indicates the comment
  author, not necessarily who should address it.
- Doxygen is used to produce source code documentation. To build the local HTML
  docs, run `doxygen Doxyfile` from the root source directory.
- Doxygen commands should use `\foo` (and not `@foo`) syntax.
- The preferred location of the comments is next to the declarations. A
  declaration with multiple conditionally compiled definitions may be
  documented in place, either above the whole conditional or, when the other
  branches are no-ops that need no documentation of their own, inside
  the branch Doxygen selects. Hoist it instead into a Doxygen comment with a
  `\def`, `\var`, or another suitable tag at the top of its section or source
  file when the definitions are scattered. Closely related declarations may be
  hoisted alongside it even when they have a single definition.
- Use a `\hideinitializer` tag when the initializer Doxygen would otherwise
  render is not worth showing, wherever the comment is placed. It is required
  when the declaration has more than one definition and the one Doxygen
  selects has a non-empty value, so that one configuration's value is not
  presented as the declaration's only value. Otherwise it is a judgement call:
  tag when the rendered value would only restate the brief, and leave it
  untagged when the value tells the reader something the brief does not. An
  initializer already longer than `MAX_INITIALIZER_LINES` may still carry the
  tag: the judgement is about the value, not about the current threshold, and
  tagging states the suppression in the source rather than resting it on a
  global setting that nothing in the declaration mentions. Where the selected
  definition is empty (no replacement text, so Doxygen renders no initializer)
  there is nothing to hide; `((void)0)` renders one and counts as non-empty.
- Which definition Doxygen selects is fixed by the documentation build's own
  configuration, not by the compiler you build with. Its inputs are
  `Doxyfile`'s `PREDEFINED` list together with the
  `#ifdef UNODB_DETAIL_DOXYGEN` sections in the sources, chiefly `global.hpp`'s
  CMake macro block. The build also executes every `#define` in the regions
  those inputs select, and the definitions propagate through `#include`:
  `global.hpp`'s macro-derivation cascade derives `UNODB_DETAIL_X86_64`, which
  appears in neither input, from `PREDEFINED`'s `_MSC_VER` and `_M_X64`,
  selecting its regions across the tree, and likewise derives
  `UNODB_DETAIL_MSVC_CLANG`, deselecting the regions its `#ifndef` guards.
  Together the inputs and the cascade currently describe a standalone,
  MSVC-with-clang-frontend, x86-64, statistics-enabled debug build. Run
  `doxygen Doxyfile` and look at the declaration's page to see which
  definition it documents. Do not read the answer off a `PREDEFINED` entry
  that names the macro itself: an entry defines the macro for the
  documentation build the way `-D` would, satisfying the conditionals that
  test it and controlling how it expands in the signatures that use it, but
  it never supplies the value shown on the macro's own page, which always
  comes from the source `#define` in the selected branch.
- A guard neither listed in the inputs nor derived by the cascade is simply
  undefined for the documentation build, and a documented declaration confined
  to a region such a guard deselects vanishes silently. Resolve every macro
  that conditionally compiles documented declarations deliberately: add it to
  `PREDEFINED`, define it in the block, let the cascade derive it, or
  knowingly leave it undefined (`NDEBUG`).
- When changing the inputs or the cascade, or introducing such a conditional,
  re-audit the `\hideinitializer` tags, re-read comments placed outside a
  conditional for claims that hold only for the definition previously
  selected, and compare the generated docs before and after for declarations
  that disappeared: Doxygen warns when a comment inside a deselected branch is
  dropped only if a selected branch still defines the declaration, and it
  never warns when a comment placed outside a conditional re-anchors. Keep the
  build description above current when the selected configuration changes.
- The comparison above is differential: it cannot surface a declaration that
  was already missing, and documentation added inside an already deselected
  region is absent from every baseline and fires no re-audit trigger. When
  adding documentation inside a conditional, and periodically, sweep the
  regions the documentation build deselects, including those a defined macro
  deselects through `#ifndef`, for Doxygen comment blocks; make each hit
  render or record it as a knowing omission. To make a declaration render,
  define and document its guard in the `UNODB_DETAIL_DOXYGEN` block, hoist a
  `\def` block anchored by an immediately-`#undef`ed `#define` in an
  `#ifdef UNODB_DETAIL_DOXYGEN` region, keeping the macro undefined for the
  selection of later conditionals, or extend the region's guard with
  `|| defined(UNODB_DETAIL_DOXYGEN)`. A hit whose declaration the
  selected branch of the same conditional already documents is a knowing
  omission recorded by the build description above; record any other knowing
  omission at the site with a `// not in the generated docs: <reason>`
  comment, so that a sweep can tell recorded omissions from new hits.
- Trailing Doxygen comments (`///<`, `/**<`) should be avoided; place
  documentation before declarations instead.
- Doxygen automatic brief description detection is enabled, thus explicit
  `\brief` tags are not required, rather the first sentence in the Doxygen
  comment block will be interpreted as the brief description. It should use a
  headline-like style without articles.
- Markdown markup is preferred, i.e. `` `foo` `` instead of `\c foo`.
- Private class members should be documented too.
- Backticking a name never costs its autolink, and usually creates one.
  Measured against this project's `Doxyfile`: Doxygen resolves a documented
  entity inside a backtick code span, single- or multi-word, but it resolves
  every name relative to the scope the comment is written in and the scopes
  enclosing it. That scope-relativity governs the function rule below as well.
  Over-qualifying costs nothing, so spell `unodb::detail` entities
  `detail::class::member` throughout: `detail::basic_inode_48::child_indexes`
  links from comments in `unodb` and in `unodb::detail` alike, while the
  shorter `basic_inode_48::child_indexes` links only from inside
  `unodb::detail` and renders as plain text in, say, `unodb::db::iterator`'s
  documentation. Within one comment block, qualify the mention that introduces
  an entity and every mention where one specific class is meant; an established
  repeat, or a mention that generalizes over several node types, may go bare,
  at the price of its link.
- A name rooted at a template parameter never resolves, backticks and
  qualification notwithstanding: `ArtPolicy` is a template parameter, not a
  scope Doxygen resolves members through. Measured in the same run,
  `` `ArtPolicy::can_eliminate_leaf` `` renders as plain text while
  `detail::basic_art_policy::can_eliminate_leaf` links. `art_internal_impl.hpp`
  keeps the `ArtPolicy::` spelling knowingly, because that is how the code
  beside those comments names the flag, so the plain text there is a recorded
  decision rather than a defect for a later sweep to fix.
- A bare, unbackticked name is context-dependent, so backtick it rather than
  relying on it. In running comment text Doxygen looks up only a name carrying
  `::`, `#` or `()`, or one naming a documented class or type alias: a bare
  `basic_inode_48` links, and so does a bare member alias such as
  `iter_result_opt`, while a bare data member or namespace-scope variable —
  `end_result`, `sync_in_inode_scan` — stays plain text. In a `\sa` paragraph,
  Doxygen attempts lookup for every word, including names in explanatory text:
  bare `torn_read_result`, `sync_in_inode_scan` and `basic_inode` all link when
  documented and reachable. Ordinary running-text lookup resumes after the
  paragraph ends. So a bare `\sa` argument is not by itself evidence of a dead
  reference — when a reference really is dead, the undocumented-target rule
  below is the usual cause.
- A reference to a function resolves with empty parentheses, a `#` prefix, or
  scope qualification — `is_value_in_slot()`, `#is_value_in_slot`,
  `detail::basic_inode_256::is_value_in_slot()` — and also with an argument
  list, but only when that list is the member's complete signature, trailing
  `const` included. Each spelling below was measured both inline and as a `\sa`
  argument:
  `is_value_in_slot(std::uint8_t)` renders as plain text while
  `is_value_in_slot(std::uint8_t) const` links, because that member is const,
  and `get_child(node_type, std::uint8_t)` links because that one is not. None
  of the resolving spellings names `noexcept`, which both members carry. An
  argument name never matches: `is_value_in_slot(index)` is plain text. An
  argument list is worth spelling when the name is overloaded, but it is
  signature-coupled: adding `const` to `get_child` would silently unlink the
  five `\sa get_child(node_type, std::uint8_t)` references in
  `art_internal_impl.hpp`.
- With the active Doxygen configuration (`EXTRACT_ALL = NO`,
  `WARN_AS_ERROR = NO`), an undocumented entity is not a link target at all, so
  a reference to it renders as plain text however it is spelled. Document the
  target in the same commit that introduces the reference: the failure is
  silent, emitting no warning of its own, and the target's own
  undocumented-member warning will not fail a build either.

## Linting and static analysis

clang-tidy, cppcheck, and cpplint will be invoked automatically during the build
if found. The current diagnostic level for them, as well as for compiler
warnings, is set very high and can be relaxed if needed.

For GCC or MSVC static analysis, add `-DSTATIC_ANALYSIS=ON` CMake option as
listed in the "CMake options" section above. For LLVM, no special CMake option
is needed; instead prepend `scan-build` to `make`.

## Testing

Google Test and DeepState fuzzer are used for testing. For DeepState, both LLVM
libfuzzer and built-in fuzzer are supported.

To run the tests, do `ctest`. It is also useful to run tests in parallel with
e.g. `ctest -j10`. For verbose test output add `-V`.

Benchmarks can also serve as tests, especially under debug build and under
sanitizers or Valgrind. For that, there is a CMake target `quick_benchmarks`,
i.e. `make -j10 -k quick_benchmarks`. CI runs this target too.

To run everything (tests and quick benchmarks) under Valgrind, use `valgrind`
CMake target.

## Fuzzing

Fuzzer tests for ART and QSBR components are located in the `fuzz_deepstate`
subdirectory. The tests use DeepState with either a brute force or
libfuzzer-based backend. However, the only supported platforms are Linux (GCC &
clang) and macOS (clang only, no AddressSanitizer and ThreadSanitizer) under
x86_64 only.

Several Make targets are available for fuzzing. For time-based brute-force
fuzzing of all components, use one of the following: `deepstate_2s`,
`deepstate_1m`, `deepstate_20m`, or `deepstate_8h`. To use individual fuzzers,
insert `art` or `qsbr`, for example: `deepstate_qsbr_20m` or `deepstate_art_8h`.
Running fuzzer under Valgrind is available through `valgrind_deepstate` for
everything or `valgrind_{art|qsbr}_deepstate` for individual fuzzers.

Fuzzers that use libfuzzer mirror the above by adding `_lf` before the time
suffix, such as `deepstate_lf_8h`, `deepstate_qsbr_lf_20m`,
`valgrind_deepstate_lf`, and so on.

## Commit messages

- Keep the first line under 72 characters and don't finish it with a full stop.
- The second line should be empty.
- Use imperative mood ("Fix bug" not "fixes bug", nor "fixed bug").
- Reference fixed issues, i.e. "fixes: #123"

## Pull Requests

- Create one PR per feature or fix. If it is possible to split PR into
  independent smaller parts, do so.
- Include documentation updates for the changes, use Doxygen as needed.
- Code changes should be covered by small deterministic tests. The coverage
  target is 100% (to the achievable extent), after non-deterministic code has
  been annotated to be excluded from coverage as described in the "Style Guide"
  section above.
- A clean CI run is a prerequisite for merging the PR.
- In the case of merge conflicts, rebase.
- All commits should be squashed into logical units. If the PR has only one
  feature or fix, as it should, there should be only one commit in it.
- Very obvious changes may be pushed directly.

## Benchmarking

Benchmarking is hard: it is easy to do it incorrectly with no obvious warning
signs. Here are some suggestions how to set it up. They will not guarantee valid
results but should exclude some classes of invalid ones:

- Set up the benchmark machine for the best CPU counter observability and run
  repeatability. This configuration is at odds with some of the Linux security
  features, so don't do that on machines close to production:

  ```bash
  sudo sh -c "echo -1 > /proc/sys/kernel/perf_event_paranoid"
  sudo sysctl -w kernel.kptr_restrict=0
  sudo sh -c "echo 0 > /proc/sys/kernel/yama/ptrace_scope"
  sudo sysctl -w vm.swappiness=0
  ```

- To make the above settings survive reboots, edit `/etc/sysctl.conf` for:

  ```conf
  kernel.perf_event_paranoid = -1
  kernel.kptr_restrict = 0
  vm.swappiness = 0
  ```

  and `/etc/sysctl.d/10-ptrace.conf` for `kernel.yama.ptrace_scope = 0`.

- On x86_64, make sure to use the performance governor for the CPU frequency
  management: edit `/etc/default/cpufrequtils` for `GOVERNOR="performance"`
  followed by `sudo /etc/init.d/cpufrequtils restart`. You can also do
  `sudo cpupower frequency-set --governor performance` for transient setting.
- Disable hyperthreading:
  `sudo sh -c "echo off > /sys/devices/system/cpu/smt/control"`
- Pick a CPU and shield it from other threads running there:
  `sudo cset shield --cpu=2 --kthread=on`
- Start a shell for benchmarking in the shield:
  `sudo cset shield --exec zsh`
- Use jemalloc for heap:
  `export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2`
- Now run your desired benchmarks. For the supported flags for i.e. benchmark
  filtering see [Google Benchmark
  documentation](https://github.com/google/benchmark/blob/main/docs/user_guide.md).
- To measure the effect of a patch, Google Benchmark provides
  [U-Test](https://github.com/google/benchmark/blob/main/docs/tools.md#u-test)
  feature. Let's say you changed N16 node fields, but only for the OLC case.
  Then you can run N16-specific tests, filtered for the OLC. The next command is
  in the directory with benchmark executables, and the baseline executables are
  at `/foo/bar/baseline`:

  ```bash
  python3 ../../../3rd_party/benchmark/tools/compare.py benchmarks \
     /foo/bar/baseline/micro_benchmark_olc \
     ./micro_benchmark_olc --benchmark_repetitions=9 --benchmark_filter='olc_db'
  ```

## License

By contributing, you agree that your contributions will be licensed under the
[LICENSE](LICENSE) terms.

[gc++style]: https://google.github.io/styleguide/cppguide.html "Google C++ Style Guide"
[sanitizer-combination-bug]: https://github.com/google/sanitizers/issues/1106
