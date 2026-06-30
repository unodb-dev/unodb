# Agent Bootstrap — unodb

## Scope

- Repository: `unodb-dev/unodb` (upstream), `thompsonbry/unodb` (fork)
- Fork remote: `origin` (SSH push target — updates upstream PRs automatically)
- Upstream remote: `upstream` (read-only reference)

## Repository Orientation

At the start of every session, read:

1. This file
2. `CONTRIBUTING.md` — code quality standards and conventions

## CI Process

All CI tooling lives in `.unodb-work/`. Scripts are executable.

### Before pushing (MANDATORY — no exceptions)

```bash
bash .unodb-work/pre-push-checks.sh        # FULL: format, build, test, sanitizers
```

**Do NOT use `--quick` before pushing to origin.** The full run catches issues
that waste 12+ hours of CI time if missed. `--quick` is only for local iteration.

### Fork CI (MSVC + coverage) — run BEFORE pushing to upstream

```bash
bash .unodb-work/trigger-msvc-ci.sh         # Merges branch into msvc-check, pushes
bash .unodb-work/trigger-coverage-ci.sh     # Merges branch into coverage-check, pushes
bash .unodb-work/check-msvc-ci.sh           # Poll MSVC status
bash .unodb-work/check-coverage-ci.sh       # Poll coverage status
bash .unodb-work/check-coverage-ci.sh --download  # Download lcov artifacts
bash .unodb-work/parse-lcov-patch.sh        # Patch coverage analysis (target ≥ 98.63%)
```

### Upstream CI — poll after pushing

```bash
bash .unodb-work/check-ci.sh https://github.com/unodb-dev/unodb/pull/<N>
```

Known infra failures (not our code): SonarCloud, claude-review.
**Read the actual CI failure logs** — do not assume failures are FPs without checking.

### Formatting

```bash
docker run --rm -v "$(pwd):/src" -w /src unodb-clang21 \
  clang-format-21 -i <files>                    # Fix
docker run --rm -v "$(pwd):/src" -w /src unodb-clang21 \
  clang-format-21 --dry-run -Werror <files>     # Check
```

## Git Rules

- **Never amend commits** — always make new ones.
- **Never `git add -A`** — stage specific files.
- **Push to `origin`** (fork) via SSH — this updates the upstream PR.
- **clang-format-21 via docker** (`unodb-clang21`) is canonical.
- **`gh` CLI token**: Run `unset GITHUB_TOKEN` before `gh pr create` so
  `gh` falls back to `~/.config/gh/hosts.yml` which has `repo` scope.

## Push Workflow (complete sequence)

1. Run FULL pre-push checks: `bash .unodb-work/pre-push-checks.sh`
2. Trigger fork CI: MSVC + coverage (scripts above)
3. Wait for MSVC to pass (9/9 jobs)
4. Wait for coverage CI to pass, then download and verify patch coverage:
   ```bash
   bash .unodb-work/check-coverage-ci.sh --download
   bash .unodb-work/parse-lcov-patch.sh   # MUST show ≥ 98.63%
   ```
   If patch coverage is below target, fix uncovered lines before pushing.
5. Push: `git push origin <branch>`
6. Poll upstream CI and READ failure logs — do not assume FPs

## Code Quality Rules (from CONTRIBUTING.md)

- `const` everywhere possible (except by-value params and movable fields).
- `constexpr` everywhere it is legal.
- `[[nodiscard]]` on all value-returning functions by default.
- `noexcept` on everything that cannot throw.
- Pass by const reference for non-trivial types.
- All testable paths must be tested; LCOV_EXCL only for genuinely dead code.
- Coverage target: ≥ 98.63% patch coverage.
- Doxygen comments on all declarations including private members.
- A clean CI run is a prerequisite for merging.

## Build Rules

### TSan and OLC Fields

**Never use `DISABLE_TSAN` or `__attribute__((no_sanitize("thread")))`.** The
pre-push check enforces this mechanically.

OLC uses optimistic reads (read without lock, validate version after). TSan
cannot model this protocol and reports races on unprotected fields. The correct
fix is to wrap the field in the existing `in_critical_section<T>` template
(relaxed atomics that TSan understands) — NOT to suppress TSan.

Pattern:

- `in_critical_section<T>` (from `optimistic_lock.hpp`) — for OLC/concurrent use
- `in_fake_critical_section<T>` (from `in_fake_critical_section.hpp`) — for single-threaded `db`
- Both provide `load()` and `operator=(T)` with identical interfaces
- Template parameterize via `ArtPolicy::template critical_section_policy`

Example (value_bitmask_field):

```cpp
template <bool Enabled, class Storage, template <class> class CritSec>
struct value_bitmask_field {
  CritSec<Storage> bits{};  // NOT plain Storage
  void set(std::uint8_t i) noexcept {
    bits = static_cast<Storage>(bits.load() | (Storage{1} << i));
  }
};
```

For array-based fields, wrap each element: `std::array<CritSec<T>, N> bits{}`.
