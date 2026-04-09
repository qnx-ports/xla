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

#include <memory>
#include <vector>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace {

using AsyncCollectiveTypeFixerTest = HloHardwareIndependentTestBase;

TEST_F(AsyncCollectiveTypeFixerTest, FixesAllToAllType) {
  // Parse the module.
  constexpr absl::string_view hlo_string = R"(
  HloModule test

  async_op {
    p0 = f32[8,16] parameter(0)
    ROOT ata = f32[8,16] all-to-all(p0), dimensions={0}, replica_groups={{0,1,2,3,4,5,6,7}}
  }

  ENTRY entry {
    p0 = f32[8,16] parameter(0)
    start = ((f32[8,16]), f32[8,16]) async-start(p0), calls=async_op
    ROOT done = f32[8,16] async-done(start)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(hlo_string));

  // Fix the types.
  AsyncCollectiveTypeFixer::ContextShapeQuery get_context_shapes =
      [](const HloInstruction* inst) {
        return std::vector<Shape>{ShapeUtil::MakeShape(U32, {})};
      };
  AsyncCollectiveTypeFixer fixer({get_context_shapes});
  TF_ASSERT_OK_AND_ASSIGN(bool changed, fixer.Run(hlo_module.get()));
  ASSERT_TRUE(changed);

  // Validate the new program.
  HloComputation* computation = hlo_module->entry_computation();
  HloInstruction* done = computation->root_instruction();
  EXPECT_EQ(done->opcode(), HloOpcode::kAsyncDone);
  HloInstruction* start = done->mutable_operand(0);
  EXPECT_EQ(start->opcode(), HloOpcode::kAsyncStart);

  const Shape& shape = start->shape();
  ASSERT_TRUE(shape.IsTuple());
  ASSERT_EQ(shape.tuple_shapes_size(), 3);
  EXPECT_EQ(shape.tuple_shapes(2).element_type(), U32);
}

TEST_F(AsyncCollectiveTypeFixerTest, FixesCollectivePermuteType) {
  // Parse the module.
  constexpr absl::string_view hlo_string = R"(
  HloModule test

  ENTRY entry {
    p0 = f32[8,16]{1,0} parameter(0)
    start = (f32[8,16]{1,0}, f32[8,16]{1,0}, u32[], u32[]) collective-permute-start(p0), source_target_pairs={{0,1},{1,0}}
    ROOT done = f32[8,16]{1,0} collective-permute-done(start)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(hlo_string));

  // Fix the types.
  AsyncCollectiveTypeFixer::ContextShapeQuery get_context_shapes =
      [](const HloInstruction* inst) {
        return std::vector<Shape>{ShapeUtil::MakeShape(U32, {})};
      };
  AsyncCollectiveTypeFixer fixer({get_context_shapes});
  TF_ASSERT_OK_AND_ASSIGN(bool changed, fixer.Run(hlo_module.get()));
  ASSERT_TRUE(changed);

  // Validate the new program.
  HloComputation* computation = hlo_module->entry_computation();
  HloInstruction* done = computation->root_instruction();
  EXPECT_EQ(done->opcode(), HloOpcode::kCollectivePermuteDone);
  HloInstruction* start = done->mutable_operand(0);
  EXPECT_EQ(start->opcode(), HloOpcode::kCollectivePermuteStart);

  const Shape& shape = start->shape();
  ASSERT_TRUE(shape.IsTuple());
  ASSERT_EQ(shape.tuple_shapes_size(), 5);
  EXPECT_EQ(shape.tuple_shapes(4).element_type(), U32);
}

}  // namespace
}  // namespace xla
