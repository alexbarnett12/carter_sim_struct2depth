"""
Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

load("//engine/build:isaac.bzl", "isaac_app")

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
        "train.py",
    ],
    data = [
        ":carter_sim.app.json",
        ":carter.config.json",
        ":carter.graph.json",
        ":navigation.config.json",
        ":navigation.graph.json",
        "//apps/assets/maps",
        "//packages/map:libmap_module.so",
        "//packages/flatsim:libflatsim_module.so",
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
    name = "train_server",
    main = "train.py",
    srcs = [
        "differential_base_state.py",
        "train.py"
    ],
    data = [
        ":carter_sim_server.app.json",
        ":carter_server.config.json",
        ":carter.graph.json",
        ":navigation.config.json",
        ":navigation.graph.json",
        "//apps/assets/maps",
        "//packages/map:libmap_module.so",
        "//packages/flatsim:libflatsim_module.so",
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
    name = "train_server_2",
    main = "train.py",
    srcs = [
        "differential_base_state.py",
        "train.py"
    ],
    data = [
        ":carter_sim_server_2.app.json",
        ":carter_server_2.config.json",
        ":carter.graph.json",
        ":navigation.config.json",
        ":navigation.graph.json",
        "//apps/assets/maps",
        "//packages/map:libmap_module.so",
        "//packages/flatsim:libflatsim_module.so",
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
    name = "live_inference",
    srcs = [
        "live_inference.py",
        "differential_base_state.py",
        "monocular_depth_map.py",
    ],
    data = [
        ":carter_sim.app.json",
        ":carter.config.json",
        ":carter.graph.json",
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
    name = "optimize_sim",
    srcs = [
        "differential_base_state.py",
        "optimize_sim.py",
    ],
    data = [
        ":carter_sim.app.json",
        ":carter.config.json",
        ":carter.graph.json",
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
    name = "save_image_triplets",
    srcs = [
        "differential_base_state.py",
        "save_image_triplets.py",
    ],
    data = [
        ":carter_sim.app.json",
        ":carter.config.json",
        ":carter.graph.json",
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
        ":carter_sim.app.json",
        ":carter.config.json",
        ":carter.graph.json",
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
