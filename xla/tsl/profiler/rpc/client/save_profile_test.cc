/* Copyright 2026 The TensorFlow Authors All Rights Reserved.

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

#include "xla/tsl/profiler/rpc/client/save_profile.h"

#include <iostream>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/profiler/utils/file_system_utils.h"
#include "tsl/profiler/protobuf/profiler_service.pb.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace tsl {
namespace profiler {
namespace {

using ::tsl::testing::TmpDir;

TEST(SaveProfileTest, SaveXSpaceCollision) {
  std::string repository_root = TmpDir();
  std::string run = absl::StrCat("test_run_", absl::ToUnixMicros(absl::Now()));
  std::string host = "my_host";
  tensorflow::profiler::XSpace xspace;

  // 1. Save first profile
  absl::Status status = SaveXSpace(repository_root, run, host, xspace);
  ASSERT_TRUE(status.ok());

  std::string log_dir = ProfilerJoinPath(repository_root, run);
  EXPECT_TRUE(Env::Default()
                  ->FileExists(ProfilerJoinPath(log_dir, "my_host.xplane.pb"))
                  .ok());

  // 2. Save second profile with same host
  status = SaveXSpace(repository_root, run, host, xspace);
  ASSERT_TRUE(status.ok());

  std::vector<std::string> files;
  ASSERT_TRUE(Env::Default()->GetChildren(log_dir, &files).ok());
  int matching_files = 0;
  for (const auto& file : files) {
    if (absl::StartsWith(file, "my_host-") &&
        absl::EndsWith(file, ".xplane.pb")) {
      matching_files++;
    }
  }
  EXPECT_EQ(matching_files, 1);
}

TEST(SaveProfileTest, SaveProfileCollision) {
  std::string repository_root = TmpDir();
  std::string run =
      absl::StrCat("test_run_profile_", absl::ToUnixMicros(absl::Now()));
  std::string host = "my_host";
  tensorflow::ProfileResponse response;
  auto* tool_data = response.add_tool_data();
  tool_data->set_name("trace.json");
  tool_data->set_data("test data");

  std::string log_dir = ProfilerJoinPath(repository_root, run);
  ASSERT_TRUE(Env::Default()->RecursivelyCreateDir(log_dir).ok());
  ASSERT_TRUE(WriteStringToFile(Env::Default(),
                                ProfilerJoinPath(log_dir, "my_host.xplane.pb"),
                                "")
                  .ok());

  absl::Status status =
      SaveProfile(repository_root, run, host, response, &std::cout);
  ASSERT_TRUE(status.ok());

  std::vector<std::string> files;
  ASSERT_TRUE(Env::Default()->GetChildren(log_dir, &files).ok());
  bool found = false;
  for (const auto& file : files) {
    if (absl::StartsWith(file, "my_host-") &&
        absl::EndsWith(file, ".trace.json")) {
      found = true;
      break;
    }
  }
  EXPECT_TRUE(found);
}

TEST(SaveProfileTest, EmptyHost) {
  std::string repository_root = TmpDir();
  std::string run =
      absl::StrCat("test_run_empty_host_", absl::ToUnixMicros(absl::Now()));
  std::string host = "";
  tensorflow::profiler::XSpace xspace;

  absl::Status status = SaveXSpace(repository_root, run, host, xspace);
  ASSERT_TRUE(status.ok());

  std::string log_dir = ProfilerJoinPath(repository_root, run);
  EXPECT_TRUE(
      Env::Default()->FileExists(ProfilerJoinPath(log_dir, ".xplane.pb")).ok());
}

}  // namespace
}  // namespace profiler
}  // namespace tsl
