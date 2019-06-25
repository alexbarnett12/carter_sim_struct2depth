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
        "//packages/ml:ml",
        "//packages/ml:tensorflow",
    ],
)

py_binary(
    name = "train",
    srcs = ["train.py",
            "pinhole_to_tensor.py"],
    data = [
        "pinhole_to_tensor.config.json",
        "pinhole_to_tensor.graph.json",
        ":base_control.graph.json",
        ":carter.config.json",
        ":carter.graph.json",
        ":navigation.config.json",
        ":navigation.graph.json",
        "//apps:py_init",
        "//messages:core_messages",
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
    srcs = ["save_image_triplets.py"],
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

py_binary(
    name = "pinhole_to_tensor",
    srcs = [
        "__init__.py",
        "pinhole_to_tensor.py",
    ],
    data = [
        "pinhole_to_tensor.config.json",
        "pinhole_to_tensor.graph.json",
        ":carter.config.json",
        ":carter.graph.json",
        "//apps:py_init",
        "//messages:core_messages",
        "//packages/ml:libml_module.so",
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
