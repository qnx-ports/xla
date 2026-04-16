# Copyright 2026 The OpenXLA Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

load("@local_config_cuda//cuda:build_defs.bzl", "cuda_library")
load("//xla:xla.default.bzl", "xla_cc_test")
load("//xla/tests:build_defs.bzl", "xla_test")
load("//xla/tsl/platform:build_config.bzl", "tf_proto_library")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

tf_proto_library(
    name = "scratch_space_lookup_table_proto",
    srcs = ["scratch_space_lookup_table.proto"],
)

cc_library(
    name = "cub_scratch_size_deviceless_lookup",
    srcs = ["cub_scratch_size_deviceless_lookup.cc"],
    hdrs = ["cub_scratch_size_deviceless_lookup.h"],
    deps = [
        ":cub_sort_utils",
        ":scratch_space_lookup_table_proto_cc",
        "//xla/stream_executor:semantic_version",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "cub_sort_utils",
    hdrs = ["cub_sort_utils.h"],
    visibility = ["//xla:internal"],
    deps = [],
)

xla_cc_test(
    name = "cub_scratch_size_deviceless_lookup_test",
    srcs = ["cub_scratch_size_deviceless_lookup_test.cc"],
    deps = [
        ":cub_scratch_size_deviceless_lookup",
        ":scratch_space_lookup_table_proto_cc",
        "//xla/stream_executor:semantic_version",
        "//xla/tsl/util/proto:parse_text_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

cuda_library(
    name = "generate_cub_scratch_size_cuda_test",
    testonly = True,
    srcs = ["generate_cub_scratch_size_test.cu.cc"],
    deps = [
        ":scratch_space_lookup_table_proto_cc",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
        "@com_google_protobuf//:protobuf",
        "@local_config_cuda//cuda:cub_headers",  # buildcleaner: keep
        "@tsl//tsl/platform:path",
    ],
    alwayslink = True,
)

xla_test(
    name = "generate_cub_scratch_size_test",
    # The actual logic is in generate_cub_scratch_size_test.cu.cc, but we can't list that file as the srcs.
    # So we need to add a dummy .cc file for xla_test to work.
    srcs = ["generate_cub_scratch_size_dummy.cc"],
    backend_tags = {
        "h100": ["full"],
        "b200": ["full"],
        "gb200": ["full"],
        "gb300": ["full"],
    },
    backends = ["gpu"],
    tags = [
        "manual",
        "notap",
    ],
    deps = [
        ":generate_cub_scratch_size_cuda_test",
        "@com_google_googletest//:gtest_main",
    ],
)
