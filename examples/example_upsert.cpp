// Copyright 2025 UnoDB contributors

// Example: CAS upsert operations — insert-or-resolve, update, and erase.
// Demonstrates the upsert API's three actions on an OLC concurrent ART.

#include "global.hpp"

#include <cstdint>
#include <iostream>
#include <string_view>

#include "art_common.hpp"
#include "olc_art.hpp"
#include "qsbr.hpp"

namespace {

using Db = unodb::olc_db<std::uint64_t, unodb::value_view>;

[[nodiscard, gnu::pure]] unodb::value_view from_sv(std::string_view sv) {
  return {reinterpret_cast<const std::byte*>(sv.data()), sv.length()};
}

}  // namespace

int main() {
  Db db;

  // --- Insert-or-resolve: key absent → inserts, returns true ---
  {
    unodb::quiescent_state_on_scope_exit qstate{};
    const bool inserted = db.upsert(1, from_sv("hello"), [](auto& /*v*/) {
      return unodb::upsert_action::keep;  // not called — key absent
    });
    std::cout << "upsert(1, \"hello\"): inserted=" << inserted << "\n";
  }

  // --- Keep: key present, lambda sees value, returns keep → no change ---
  {
    unodb::quiescent_state_on_scope_exit qstate{};
    const bool inserted = db.upsert(1, from_sv("world"), [](auto& /*v*/) {
      return unodb::upsert_action::keep;
    });
    std::cout << "upsert(1, keep): inserted=" << inserted << "\n";
    // Value is still "hello".
  }

  // --- Erase: key present, lambda returns erase → CAS remove ---
  {
    unodb::quiescent_state_on_scope_exit qstate{};
    const bool inserted = db.upsert(1, from_sv("unused"), [](auto& /*v*/) {
      return unodb::upsert_action::erase;
    });
    std::cout << "upsert(1, erase): inserted=" << inserted << "\n";
    // Key 1 is now removed from the tree.
  }

  // --- Verify key is gone ---
  {
    unodb::quiescent_state_on_scope_exit qstate{};
    const auto result = db.get(1);
    std::cout << "get(1) after erase: found=" << result.has_value() << "\n";
  }

  // --- Re-insert via upsert (key absent again) ---
  {
    unodb::quiescent_state_on_scope_exit qstate{};
    const bool inserted = db.upsert(1, from_sv("back"), [](auto& /*v*/) {
      return unodb::upsert_action::keep;
    });
    std::cout << "upsert(1, \"back\"): inserted=" << inserted << "\n";
  }
}
