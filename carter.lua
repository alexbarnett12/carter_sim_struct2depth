-- Copyright 2016 The Cartographer Authors
--
-- Licensed under the Apache License, Version 2.0 (the "License");
-- you may not use this file except in compliance with the License.
-- You may obtain a copy of the License at
--
--      http://www.apache.org/licenses/LICENSE-2.0
--
-- Unless required by applicable law or agreed to in writing, software
-- distributed under the License is distributed on an "AS IS" BASIS,
-- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
-- See the License for the specific language governing permissions and
-- limitations under the License.

include "map_builder.lua"
include "trajectory_builder.lua"

options = {
  map_builder = MAP_BUILDER,
  trajectory_builder = TRAJECTORY_BUILDER,
}

MAP_BUILDER.num_background_threads = 8
MAP_BUILDER.use_trajectory_builder_2d = true
MAP_BUILDER.use_trajectory_builder_3d = false

TRAJECTORY_BUILDER_2D.ceres_scan_matcher.occupied_space_weight = 10
TRAJECTORY_BUILDER_2D.ceres_scan_matcher.rotation_weight = 40
TRAJECTORY_BUILDER_2D.ceres_scan_matcher.ceres_solver_options.num_threads = 2
TRAJECTORY_BUILDER_2D.max_range = 25
TRAJECTORY_BUILDER_2D.min_range = 0.4
TRAJECTORY_BUILDER_2D.motion_filter.max_angle_radians = math.rad(0.2)
TRAJECTORY_BUILDER_2D.motion_filter.max_distance_meters = 0.1
TRAJECTORY_BUILDER_2D.num_accumulated_range_data = 1
TRAJECTORY_BUILDER_2D.submaps.grid_options_2d.resolution = 0.1
TRAJECTORY_BUILDER_2D.submaps.num_range_data = 120
TRAJECTORY_BUILDER_2D.submaps.range_data_inserter.probability_grid_range_data_inserter.hit_probability = 0.55
TRAJECTORY_BUILDER_2D.submaps.range_data_inserter.probability_grid_range_data_inserter.miss_probability = 0.49
TRAJECTORY_BUILDER_2D.use_imu_data = false
TRAJECTORY_BUILDER_2D.use_online_correlative_scan_matching = true

POSE_GRAPH.constraint_builder.global_localization_min_score = 0.7
POSE_GRAPH.constraint_builder.loop_closure_rotation_weight = 1e5
POSE_GRAPH.constraint_builder.loop_closure_translation_weight = 1.1e4
POSE_GRAPH.constraint_builder.max_constraint_distance = 15
POSE_GRAPH.constraint_builder.min_score=0.8
POSE_GRAPH.constraint_builder.sampling_ratio = 0.3
POSE_GRAPH.matcher_rotation_weight = 1e2
POSE_GRAPH.matcher_translation_weight = 1e2
POSE_GRAPH.optimization_problem.huber_scale = 1e2
POSE_GRAPH.optimization_problem.local_slam_pose_rotation_weight = 1e2
POSE_GRAPH.optimization_problem.local_slam_pose_translation_weight = 1e3
POSE_GRAPH.optimization_problem.log_solver_summary = true
POSE_GRAPH.optimization_problem.odometry_rotation_weight = 1e2
POSE_GRAPH.optimization_problem.odometry_translation_weight = 1e3
POSE_GRAPH.optimize_every_n_nodes = 120


return options