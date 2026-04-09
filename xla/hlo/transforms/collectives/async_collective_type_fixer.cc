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

#include "xla/hlo/transforms/collectives/async_collective_type_fixer.h"

#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/errors.h"

namespace xla {
namespace {

absl::Status UpdateShapeAndReplace(HloInstruction* start,
                                   const std::vector<Shape>& context_shapes) {
  const Shape& old_shape = start->shape();
  std::vector<Shape> new_tuple_shapes(old_shape.tuple_shapes().begin(),
                                      old_shape.tuple_shapes().end());
  new_tuple_shapes.insert(new_tuple_shapes.end(), context_shapes.begin(),
                          context_shapes.end());
  Shape new_shape = ShapeUtil::MakeTupleShape(new_tuple_shapes);
  HloComputation* computation = start->parent();
  HloInstruction* new_start =
      computation->AddInstruction(start->CloneWithNewShape(new_shape, ""));
  TF_RETURN_IF_ERROR(
      computation->ReplaceInstructionWithDifferentShape(start, new_start));
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<bool> AsyncCollectiveTypeFixer::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;

  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    std::vector<HloInstruction*> async_starts;
    for (HloInstruction* instruction : computation->instructions()) {
      // These are the collectives where StableHLO might be lowered to an HLO
      // collective with the wrong return type.
      if (instruction->opcode() == HloOpcode::kAsyncStart ||
          instruction->opcode() == HloOpcode::kCollectivePermuteStart) {
        async_starts.push_back(instruction);
      }
    }

    for (HloInstruction* start : async_starts) {
      if (start->opcode() == HloOpcode::kAsyncStart) {
        HloComputation* called_computation = start->called_computations()[0];
        HloInstruction* root = called_computation->root_instruction();
        if (root->opcode() != HloOpcode::kAllToAll &&
            root->opcode() != HloOpcode::kReduceScatter &&
            root->opcode() != HloOpcode::kCollectiveBroadcast) {
          continue;
        }
        std::vector<Shape> context_shapes = config_.get_context_shapes(root);
        if (context_shapes.empty()) {
          continue;
        }
        TF_RETURN_IF_ERROR(UpdateShapeAndReplace(start, context_shapes));
        changed = true;
      } else if (start->opcode() == HloOpcode::kCollectivePermuteStart) {
        std::vector<Shape> context_shapes = config_.get_context_shapes(start);
        if (context_shapes.empty()) {
          continue;
        }
        TF_RETURN_IF_ERROR(UpdateShapeAndReplace(start, context_shapes));
        changed = true;
      }
    }
  }
  return changed;
}

}  // namespace xla
