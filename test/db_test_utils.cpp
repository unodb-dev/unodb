// Copyright 2019-2025 UnoDB contributors

// Should be the first include
#include "global.hpp"

#include "db_test_utils.hpp"  // IWYU pragma: keep

#include <cstdint>
#include <iostream>  // IWYU pragma: keep

#include "art.hpp"
#include "art_common.hpp"
#include "art_internal.hpp"
#include "mutex_art.hpp"
#include "olc_art.hpp"

namespace unodb {

//
// Instantiate the various classes under test.
//

template class unodb::db<std::uint64_t, unodb::value_view>;
template class unodb::mutex_db<std::uint64_t, unodb::value_view>;
template class unodb::olc_db<std::uint64_t, unodb::value_view>;

template class unodb::db<unodb::key_view, unodb::value_view>;
template class unodb::mutex_db<unodb::key_view, unodb::value_view>;
template class unodb::olc_db<unodb::key_view, unodb::value_view>;

template class unodb::db<unodb::key_view, std::uint64_t>;
template class unodb::mutex_db<unodb::key_view, std::uint64_t>;
template class unodb::olc_db<unodb::key_view, std::uint64_t>;

}  // namespace unodb

namespace unodb::test {

//
// Instantiate tree_verifier variants for each class under test.
//

template class tree_verifier<u64_db>;
template class tree_verifier<u64_mutex_db>;
template class tree_verifier<u64_olc_db>;

template class tree_verifier<key_view_db>;
template class tree_verifier<key_view_mutex_db>;
template class tree_verifier<key_view_olc_db>;

template class tree_verifier<key_view_u64val_db>;
template class tree_verifier<key_view_u64val_mutex_db>;
template class tree_verifier<key_view_u64val_olc_db>;

}  // namespace unodb::test

namespace unodb {

// Ensure unrolled in .cpp and therefore available to debugger.
template <>
[[gnu::cold]] UNODB_DETAIL_NOINLINE void
detail::basic_art_key<std::uint64_t>::dump() const {
  dump(std::cerr);
}
template <>
[[gnu::cold]] UNODB_DETAIL_NOINLINE void
detail::basic_art_key<unodb::key_view>::dump() const {
  dump(std::cerr);
}

}  // namespace unodb
