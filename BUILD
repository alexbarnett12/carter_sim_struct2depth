"""
Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

load("//engine/build:isaac.bzl", "isaac_app", "isaac_cc_binary", "isaac_cc_library")

isaac_cc_library(
    name = "alice",
    visibility = ["//visibility:public"],
    deps = [
        ":impl",
        "//engine/alice/behaviors",
        "//engine/alice/tools:gather_component_info",
    ],
)

isaac_cc_library(
    name = "impl",
    srcs = glob([
        "backend/*.cpp",
        "components/deprecated/*.cpp",
        "components/*.cpp",
        "hooks/*.cpp",
        "*.cpp",
    ]),
    hdrs = glob([
        "backend/*.hpp",
        "components/deprecated/*.hpp",
        "components/*.hpp",
        "hooks/*.hpp",
        "*.hpp",
    ]),
    linkopts = [
        "-lstdc++fs",
        "-ldl",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//engine/core",
        "//engine/core/buffers",
        "//engine/core/math",
        "//engine/gems/algorithm:string_utils",
        "//engine/gems/cask",
        "//engine/gems/geometry",
        "//engine/gems/math",
        "//engine/gems/pose_tree",
        "//engine/gems/scheduler",
        "//engine/gems/serialization",
        "//engine/gems/serialization:capnp",
        "//engine/gems/serialization:json",
        "//engine/gems/sight",
        "//engine/gems/system:filesystem",
        "//engine/gems/uuid",
        "//messages:alice",
        "@asio",
        "@breakpad",
        "@capnproto//:capnp_lite",
        "@lmdb",
        "@com_google_absl//absl/strings",
    ],
)


isaac_app(
    name = "carter_sim",
    app_json_file = "carter_sim.app.json",
    data = [
        "carter.config.json",
        "carter.graph.json",
        "navigation.config.json",
        "navigation.graph.json",
        "//apps/assets/maps",
        "//packages/map:libmap_module.so",
    ],
    modules = [
        "navigation",
        "perception",
        "planner",
        "viewers",
        "flatsim",
    ],
)

py_binary(
    name = "train",
    srcs = [
        "differential_base_state.py",
        "pinhole_to_tensor.py",
        "train.py",
    ],
    data = [
        ":carter_sim.app.json",
        ":carter_server.config.json",
        ":carter.graph.json",
        ":navigation.config.json",
        ":navigation.graph.json",
        ":base_control.graph.json",
        "//apps/assets/maps",
        "//packages/map:libmap_module.so",
        "//packages/flatsim:libflatsim_module.so",
        "//packages/ml:libml_module.so",
        "//packages/navigation:libnavigation_module.so",
        "//packages/perception:libperception_module.so",
        "//packages/planner:libplanner_module.so",
        "//packages/viewers:libviewers_module.so",
        "//apps:py_init",
        "//messages:core_messages",
    ],
    deps = [
        "//engine/pyalice",
        "//packages/ml:pyml",
    ],
)

py_binary(
    name = "live_inference",
    srcs = [
        "live_inference.py",
        "monocular_depth_map.py",
    ],
    data = [
        ":base_control.graph.json",
        ":carter.config.json",
        ":carter_inference.graph.json",
        ":navigation.config.json",
        ":navigation.graph.json",
        "//apps:py_init",
        "//apps/assets/maps",
        "//messages:core_messages",
        "//packages/flatsim:libflatsim_module.so",
        "//packages/map:libmap_module.so",
        "//packages/ml:libml_module.so",
        "//packages/navigation:libnavigation_module.so",
        "//packages/perception:libperception_module.so",
        "//packages/planner:libplanner_module.so",
        "//packages/viewers:libviewers_module.so",
    ],
    deps = [
        "//engine/pyalice",
        "//packages/ml:pyml",
    ],
)

py_binary(
    name = "optimize_sim",
    srcs = [
        "differential_base_state.py",
        "pinhole_to_tensor.py",
        "optimize_sim.py",
    ],
    data = [
        "pinhole_to_tensor.config.json",
        "pinhole_to_tensor.graph.json",
        ":base_control.graph.json",
        ":carter.config.json",
        ":carter.graph.json",
        ":navigation.config.json",
        ":navigation.graph.json",
        "//apps:py_init",
        "//apps/assets/maps",
        "//messages:core_messages",
        "//packages/flatsim:libflatsim_module.so",
        "//packages/map:libmap_module.so",
        "//packages/ml:libml_module.so",
        "//packages/navigation:libnavigation_module.so",
        "//packages/perception:libperception_module.so",
        "//packages/planner:libplanner_module.so",
        "//packages/viewers:libviewers_module.so",
    ],
    deps = [
        "//engine/pyalice",
        "//packages/ml:pyml",
    ],
)

py_binary(
    name = "save_image_triplets",
    srcs = [
        "differential_base_state.py",
        "save_image_triplets.py",
    ],
    data = [
        ":base_control.graph.json",
        ":carter_save.config.json",
        ":carter_save.graph.json",
        ":navigation.config.json",
        ":navigation.graph.json",
        "//apps/assets/maps",
        "//packages/flatsim:libflatsim_module.so",
        "//packages/map:libmap_module.so",
        "//packages/ml:libml_module.so",
        "//packages/navigation:libnavigation_module.so",
        "//packages/perception:libperception_module.so",
        "//packages/planner:libplanner_module.so",
        "//packages/viewers:libviewers_module.so",
    ],
    deps = [
        "//engine/pyalice",
        "//packages/ml:pyml",
    ],
)

py_binary(
    name = "save_images",
    srcs = ["save_images.py"],
    data = [
        ":base_control.graph.json",
        ":carter_save.config.json",
        ":carter_save.graph.json",
        ":navigation.config.json",
        ":navigation.graph.json",
        "//apps/assets/maps",
        "//packages/flatsim:libflatsim_module.so",
        "//packages/map:libmap_module.so",
        "//packages/ml:libml_module.so",
        "//packages/navigation:libnavigation_module.so",
        "//packages/perception:libperception_module.so",
        "//packages/planner:libplanner_module.so",
        "//packages/viewers:libviewers_module.so",
    ],
    deps = [
        "//engine/pyalice",
        "//packages/ml:pyml",
    ],
)
isaac_app(
    name = "carter_sim_joystick",
    app_json_file = "carter_sim_joystick.app.json",
    data = [
        "carter.config.json",
        "carter.graph.json",
        "joystick.config.json",
        "joystick.graph.json",
        "navigation.config.json",
        "navigation.graph.json",
        "//apps/assets/maps",
        "//packages/map:libmap_module.so",
    ],
    modules = [
        "navigation",
        "perception",
        "planner",
        "viewers",
        "flatsim",
        "sensors:joystick",
    ],
)

isaac_app(
    name = "carter_sim_mapping",
    app_json_file = "carter_sim_mapping.app.json",
    data = [
        "carter.config.json",
        "carter.graph.json",
        "carter.lua",
        "joystick.config.json",
        "joystick.graph.json",
        "//packages/map:libmap_module.so",
    ],
    modules = [
        "perception",
        "planner",
        "viewers",
        "flatsim",
        "navigation",
        "navigation:cartographer",
        "sensors:joystick",
    ],
)
