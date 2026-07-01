// Copyright 2026 UnoDB contributors
//
// Shared implementation of the bulk_load algorithm for all ART variants.
// Included at the bottom of art.hpp and olc_art.hpp after class definitions.
//
// Template parameter Db: must expose art_key_type, art_policy, tree_depth_type,
// inode_4, inode_16, inode_48, inode_256, build_result, bulk_subtree_guard,
// build_chain(), account_growing_inode<>(), and root member.

#ifndef UNODB_DETAIL_ART_BULK_LOAD_DETAIL_HPP
#define UNODB_DETAIL_ART_BULK_LOAD_DETAIL_HPP

// MSVC static analysis false positive: claims pointers dangle after
// smart-pointer release() + reassignment. Suppressed file-wide.
UNODB_DETAIL_DISABLE_MSVC_WARNING(26815)
// C26496: vmask variables are mutated in if-constexpr branches that
// MSVC SA doesn't track; cannot be const in the general case.
UNODB_DETAIL_DISABLE_MSVC_WARNING(26496)

/// \cond UNODB_DETAIL_INTERNAL
namespace unodb::detail {

/// Core bulk_load algorithm parameterized on database type.
///
/// \pre Tree must be empty (caller validates).
/// \pre [first, last) is sorted and non-empty (caller validates).
template <typename Db, typename ExecutionPolicy, typename RandomIt>
void bulk_load_impl(Db& self, ExecutionPolicy&&, RandomIt first,
                    RandomIt last) {
  using art_key_type = typename Db::art_key_type;
  using art_policy = typename Db::art_policy;
  using tree_depth_type = typename Db::tree_depth_type;
  using inode_4 = typename Db::inode_4;
  using inode_16 = typename Db::inode_16;
  using inode_48 = typename Db::inode_48;
  using inode_256 = typename Db::inode_256;
  using build_result_t = typename Db::build_result;
  using guard_t = typename Db::bulk_subtree_guard;
  using node_ptr_t = decltype(build_result_t{}.ptr);
  using bulk_child_t = bulk_child<node_ptr_t>;
  constexpr std::size_t prefix_cap = key_prefix_capacity;

  auto common_prefix_length = [](RandomIt f, RandomIt l,
                                 tree_depth_type depth) -> std::size_t {
    const art_key_type first_ak{f->first};
    const art_key_type last_ak{std::prev(l)->first};
    const auto fk = first_ak.get_key_view();
    const auto lk = last_ak.get_key_view();
    const auto d = static_cast<std::size_t>(depth);
    const auto max_len = std::min(fk.size(), lk.size());
    std::size_t len = 0;
    while (d + len < max_len && fk[d + len] == lk[d + len]) ++len;
    return len;
  };

  struct partition_entry {
    RandomIt begin;
    std::byte key_byte;
  };

  auto partition_by_byte = [](RandomIt f, RandomIt l,
                              tree_depth_type dispatch_depth) {
    boost::container::small_vector<partition_entry, 16> parts;
    const auto dd = static_cast<std::size_t>(dispatch_depth);
    auto cur = f;
    while (cur != l) {
      const art_key_type ak{cur->first};
      const auto kv = ak.get_key_view();
      UNODB_DETAIL_ASSERT(dd < kv.size());
      const auto byte = kv[dd];
      parts.push_back({cur, byte});
      ++cur;
      while (cur != l) {
        const art_key_type ak2{cur->first};
        const auto kv2 = ak2.get_key_view();
        UNODB_DETAIL_ASSERT(dd < kv2.size());
        if (kv2[dd] != byte) break;
        ++cur;
      }
    }
    UNODB_DETAIL_DISABLE_MSVC_WARNING(26479)
    return std::move(parts);  // NOLINT(performance-move-const-arg)
    UNODB_DETAIL_RESTORE_MSVC_WARNINGS()
  };

  auto build_prefix_chain = [&self](art_key_type k, node_ptr_t child_inode,
                                    tree_depth_type start_depth,
                                    std::size_t end_depth) -> node_ptr_t {
    const auto full_key = k.get_key_view();
    const auto start = static_cast<std::size_t>(start_depth);
    auto current = child_inode;
    std::size_t pos = end_depth;
    while (pos > start + prefix_cap) {
      const auto depth = pos - prefix_cap - 1;
      const auto dispatch = full_key[pos - 1];
      auto remaining = k;
      remaining.shift_right(depth);
      auto chain{
          inode_4::create(self, full_key, remaining,
                          tree_depth_type{static_cast<std::uint32_t>(depth)},
                          dispatch, current)};
      current = node_ptr_t{chain.release(), node_type::I4};
#ifdef UNODB_DETAIL_WITH_STATS
      self.template account_growing_inode<node_type::I4>();
#endif
      pos = depth;
    }
    if (pos > start) {
      const auto dispatch = full_key[pos - 1];
      auto chain{inode_4::create(
          self, full_key, tree_depth_type{static_cast<std::uint32_t>(start)},
          static_cast<key_prefix_size>(pos - start - 1), dispatch, current)};
      current = node_ptr_t{chain.release(), node_type::I4};
#ifdef UNODB_DETAIL_WITH_STATS
      self.template account_growing_inode<node_type::I4>();
#endif
    }
    return current;
  };

  auto build_single_leaf = [&self](RandomIt it,
                                   tree_depth_type depth) -> build_result_t {
    const art_key_type ak{it->first};
    if constexpr (art_policy::can_eliminate_leaf) {
      const auto packed = art_policy::pack_value(it->second);
      if (static_cast<std::size_t>(depth) >= ak.size()) {
        return {packed, true};
      }
      return {self.build_chain(ak, packed, depth), false};
    } else if constexpr (art_policy::can_eliminate_key_in_leaf) {
      auto leaf = art_policy::make_db_leaf_ptr(ak, it->second, self);
      auto leaf_ptr = node_ptr_t{leaf.release(), node_type::LEAF};
      if (static_cast<std::size_t>(depth) >= ak.size()) {
        return {leaf_ptr, false};
      }
      return {self.build_chain(ak, leaf_ptr, depth), false};
    } else {
      static_cast<void>(depth);
      auto leaf = art_policy::make_db_leaf_ptr(ak, it->second, self);
      const auto leaf_ptr = node_ptr_t{leaf.release(), node_type::LEAF};
      return {leaf_ptr, false};
    }
  };

  // ─── Shared inode factory ────────────────────────────────────────────

  auto make_bulk_inode =
      [&self](std::span<const bulk_child_t> cs,
              const boost::container::small_vector<guard_t, 16>& guards,
              const boost::container::small_vector<bulk_child_t, 16>& children,
              key_prefix_size inode_prefix_len, key_view prefix_kv,
              tree_depth_type inode_depth) -> node_ptr_t {
    if constexpr (!art_policy::can_eliminate_leaf) {
      static_cast<void>(guards);
      static_cast<void>(children);
    }
    const auto child_count = cs.size();
    if (child_count <= 4) {
      std::uint8_t vmask = 0;
      if constexpr (art_policy::can_eliminate_leaf) {
        for (std::size_t i = 0; i < child_count; ++i) {
          if (guards[i].is_packed_value)
            vmask |= static_cast<std::uint8_t>(1U << i);
        }
      }
      auto ptr = inode_4::create_bulk(self, inode_prefix_len, prefix_kv,
                                      inode_depth, cs, vmask);
#ifdef UNODB_DETAIL_WITH_STATS
      self.template account_growing_inode<node_type::I4>();
#endif
      return node_ptr_t{ptr.release(), node_type::I4};
    }
    if (child_count <= 16) {
      std::uint16_t vmask = 0;
      if constexpr (art_policy::can_eliminate_leaf) {
        for (std::size_t i = 0; i < child_count; ++i) {
          if (guards[i].is_packed_value)
            vmask |= static_cast<std::uint16_t>(1U << i);
        }
      }
      auto ptr = inode_16::create_bulk(self, inode_prefix_len, prefix_kv,
                                       inode_depth, cs, vmask);
#ifdef UNODB_DETAIL_WITH_STATS
      self.template account_growing_inode<node_type::I16>();
#endif
      return node_ptr_t{ptr.release(), node_type::I16};
    }
    if (child_count <= 48) {
      std::array<std::uint8_t, 6> vmask{};
      if constexpr (art_policy::can_eliminate_leaf) {
        for (std::size_t i = 0; i < child_count; ++i) {
          if (guards[i].is_packed_value)
            vmask[i / 8] |= static_cast<std::uint8_t>(1U << (i % 8));
        }
      }
      auto ptr = inode_48::create_bulk(self, inode_prefix_len, prefix_kv,
                                       inode_depth, cs, vmask);
#ifdef UNODB_DETAIL_WITH_STATS
      self.template account_growing_inode<node_type::I48>();
#endif
      return node_ptr_t{ptr.release(), node_type::I48};
    }
    std::array<std::uint8_t, 32> vmask{};
    if constexpr (art_policy::can_eliminate_leaf) {
      for (std::size_t i = 0; i < child_count; ++i) {
        if (guards[i].is_packed_value) {
          const auto kb = static_cast<std::uint8_t>(children[i].key_byte);
          vmask[static_cast<std::size_t>(kb / 8)] |=
              static_cast<std::uint8_t>(1U << (kb % 8));
        }
      }
    }
    auto ptr = inode_256::create_bulk(self, inode_prefix_len, prefix_kv,
                                      inode_depth, cs, vmask);
#ifdef UNODB_DETAIL_WITH_STATS
    self.template account_growing_inode<node_type::I256>();
#endif
    return node_ptr_t{ptr.release(), node_type::I256};
  };

  // ─── Recursive subtree builder ─────────────────────────────────────────

  struct subtree_builder {
    Db& self;
    decltype(common_prefix_length)& cpl;
    decltype(partition_by_byte)& pbb;
    decltype(build_prefix_chain)& bpc;
    decltype(build_single_leaf)& bsl;
    decltype(make_bulk_inode)& mbi;

    build_result_t operator()(RandomIt f, RandomIt l,
                              tree_depth_type depth) const {
      const auto n = std::distance(f, l);
      if (n == 0) return {node_ptr_t{nullptr}, false};
      if (n == 1) return bsl(f, depth);

      const auto prefix_len = cpl(f, l, depth);
      const auto dispatch_depth = tree_depth_type{static_cast<std::uint32_t>(
          static_cast<std::size_t>(depth) + prefix_len)};
      auto parts = pbb(f, l, dispatch_depth);
      const auto child_count = parts.size();

      boost::container::small_vector<guard_t, 16> guards;
      guards.reserve(child_count);
      boost::container::small_vector<bulk_child_t, 16> children;
      children.reserve(child_count);

      const auto next_depth = tree_depth_type{static_cast<std::uint32_t>(
          static_cast<std::size_t>(dispatch_depth) + 1)};

      for (std::size_t i = 0; i < child_count; ++i) {
        const auto part_begin = parts[i].begin;
        const auto part_end = (i + 1 < child_count) ? parts[i + 1].begin : l;
        auto result = (*this)(part_begin, part_end, next_depth);
        guards.emplace_back(self);
        guards.back().ptr = result.ptr;
        guards.back().is_packed_value = result.is_packed_value;
        children.push_back({parts[i].key_byte, result.ptr});
      }

      const std::size_t chain_consumed =
          (prefix_len > prefix_cap)
              ? (prefix_len / (prefix_cap + 1)) * (prefix_cap + 1)
              : 0;
      const auto inode_prefix_len =
          static_cast<key_prefix_size>(prefix_len - chain_consumed);
      const auto inode_depth = tree_depth_type{static_cast<std::uint32_t>(
          static_cast<std::size_t>(depth) + chain_consumed)};

      const art_key_type prefix_key{f->first};
      const auto prefix_kv = prefix_key.get_key_view();

      const auto cs =
          std::span<const bulk_child_t>{children.data(), children.size()};
      auto inode_ptr =
          mbi(cs, guards, children, inode_prefix_len, prefix_kv, inode_depth);
      for (auto& g : guards) g.release();

      if (chain_consumed > 0) {
        return {bpc(prefix_key, inode_ptr, depth,
                    chain_consumed + static_cast<std::size_t>(depth)),
                false};
      }
      return {inode_ptr, false};
    }
  };

  subtree_builder builder{self,
                          common_prefix_length,
                          partition_by_byte,
                          build_prefix_chain,
                          build_single_leaf,
                          make_bulk_inode};

  using policy_t = std::remove_cvref_t<ExecutionPolicy>;
  constexpr bool is_parallel =
      std::is_same_v<policy_t, std::execution::parallel_policy> ||
      std::is_same_v<policy_t, std::execution::parallel_unsequenced_policy>;

  if constexpr (!is_parallel) {
    auto result = builder(first, last, tree_depth_type{0});
    self.root = result.ptr;
  } else {
    const auto n = std::distance(first, last);
    if (n <= 1) {
      auto result = builder(first, last, tree_depth_type{0});
      self.root = result.ptr;
      return;
    }

    const auto prefix_len =
        common_prefix_length(first, last, tree_depth_type{0});
    const auto dispatch_depth =
        tree_depth_type{static_cast<std::uint32_t>(prefix_len)};
    auto parts = partition_by_byte(first, last, dispatch_depth);
    const auto child_count = parts.size();

    const auto next_depth = tree_depth_type{static_cast<std::uint32_t>(
        static_cast<std::size_t>(dispatch_depth) + 1)};

    std::vector<std::future<build_result_t>> futures;
    futures.reserve(child_count);
    for (std::size_t i = 0; i < child_count; ++i) {
      const auto part_begin = parts[i].begin;
      const auto part_end = (i + 1 < child_count) ? parts[i + 1].begin : last;
      futures.push_back(std::async(
          std::launch::async, [&builder, part_begin, part_end, next_depth] {
            return builder(part_begin, part_end, next_depth);
          }));
    }

    boost::container::small_vector<guard_t, 16> guards;
    guards.reserve(child_count);
    boost::container::small_vector<bulk_child_t, 16> children;
    children.reserve(child_count);

    for (std::size_t i = 0; i < child_count; ++i) {
      try {
        auto result = futures[i].get();
        guards.emplace_back(self);
        guards.back().ptr = result.ptr;
        guards.back().is_packed_value = result.is_packed_value;
        children.push_back({parts[i].key_byte, result.ptr});
      } catch (...) {
        // Drain remaining futures into guards to prevent leaks.
        for (std::size_t j = i + 1; j < child_count; ++j) {
          try {
            auto r = futures[j].get();
            guards.emplace_back(self);
            guards.back().ptr = r.ptr;
            guards.back().is_packed_value = r.is_packed_value;
          } catch (...) {
          }  // LCOV_EXCL_LINE
        }
        throw;
      }
    }

    const std::size_t chain_consumed =
        (prefix_len > prefix_cap)
            ? (prefix_len / (prefix_cap + 1)) * (prefix_cap + 1)
            : 0;
    const auto inode_prefix_len =
        static_cast<key_prefix_size>(prefix_len - chain_consumed);
    const auto inode_depth =
        tree_depth_type{static_cast<std::uint32_t>(chain_consumed)};

    const art_key_type prefix_key{first->first};
    const auto prefix_kv = prefix_key.get_key_view();

    const auto cs =
        std::span<const bulk_child_t>{children.data(), children.size()};
    auto inode_ptr = make_bulk_inode(cs, guards, children, inode_prefix_len,
                                     prefix_kv, inode_depth);
    for (auto& g : guards) g.release();

    if (chain_consumed > 0) {
      self.root = build_prefix_chain(prefix_key, inode_ptr, tree_depth_type{0},
                                     chain_consumed);
    } else {
      self.root = inode_ptr;
    }
  }
}

}  // namespace unodb::detail
/// \endcond

UNODB_DETAIL_RESTORE_MSVC_WARNINGS()
UNODB_DETAIL_RESTORE_MSVC_WARNINGS()

#endif  // UNODB_DETAIL_ART_BULK_LOAD_DETAIL_HPP
