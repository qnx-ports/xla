/* Copyright 2026 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef XLA_HLO_TRANSFORMS_COLLECTIVES_ASYNC_COLLECTIVE_TYPE_FIXER_H_
#define XLA_HLO_TRANSFORMS_COLLECTIVES_ASYNC_COLLECTIVE_TYPE_FIXER_H_

#include <functional>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/shape.h"

namespace xla {

// The translation from StableHLO to HLO does depend on the backend (e.g., CPU
// vs GPU vs TPU). It behaves the same regardless of the backend. However, the
// return types of HLO async collectives do vary based on the backend.
//
// For example, consider an all-to-all run on a 2x2 array across two devices. On
// GPU, the HLO `async_start` returns a
//
//  ((f32[2,2]{1,0}), f32[2,2]{1,0})
//
// but on TPU, it returns a
//
//  ((f32[2,2]{1,0}), f32[2,2]{1,0}, u32[]{:S(2)}, u32[]{:S(2)})
//
// Thus, the an async StableHLO collective might be translated to an async HLO
// collective with the wrong return type. AsyncCollectiveTypeFixer fixes these
// types by appending the appropriate backend specific types, also called
// "context shapes".
class AsyncCollectiveTypeFixer : public HloModulePass {
 public:
  // Function to query the shape of the "context" for collectives that use
  // HLO async-start/async-done. See async_collective_creator.h for more
  // details.
  using ContextShapeQuery =
      std::function<std::vector<Shape>(const HloInstruction*)>;

  struct Config {
    ContextShapeQuery get_context_shapes = [](const HloInstruction*) {
      return std::vector<Shape>{};
    };
  };

  explicit AsyncCollectiveTypeFixer(Config config)
      : config_(std::move(config)) {}

  absl::string_view name() const override {
    return "async-collective-type-fixer";
  }

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  Config config_;
};

}  // namespace xla

#endif  // XLA_HLO_TRANSFORMS_COLLECTIVES_ASYNC_COLLECTIVE_TYPE_FIXER_H_
