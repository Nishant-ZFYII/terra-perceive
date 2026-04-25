cmake_minimum_required(VERSION 3.16)
project(construction_perception VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(nav2_costmap_2d REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(
    include
    third_party/stb
    third_party/tinycolormap
    ${EIGEN3_INCLUDE_DIRS}
)

# --- Phase 1: Core perception libraries (from scratch, Eigen only) ---

add_library(ransac_ground_seg src/ransac_ground_seg.cpp)
target_link_libraries(ransac_ground_seg Eigen3::Eigen)

add_library(traversability src/traversability.cpp)
target_link_libraries(traversability Eigen3::Eigen)

add_library(cam_lidar_projection src/cam_lidar_projection.cpp)
target_link_libraries(cam_lidar_projection Eigen3::Eigen)

add_library(safety_supervisor src/safety_supervisor.cpp)
target_link_libraries(safety_supervisor Eigen3::Eigen)

add_library(point_cloud_loader src/point_cloud_loader.cpp)
target_link_libraries(point_cloud_loader Eigen3::Eigen)

add_executable(pipeline_runner src/pipeline_runner.cpp)
target_link_libraries(pipeline_runner point_cloud_loader ransac_ground_seg traversability Eigen3::Eigen)

add_executable(safety_runner src/safety_runner.cpp)
target_link_libraries(safety_runner safety_supervisor Eigen3::Eigen)

# --- Phase 2: Tracking libraries (stubs for now) ---

add_library(kalman_filter src/kalman_filter.cpp)
target_link_libraries(kalman_filter Eigen3::Eigen)

add_library(hungarian src/hungarian.cpp)
target_link_libraries(hungarian Eigen3::Eigen)

add_library(sort_tracker src/sort_tracker.cpp)
target_link_libraries(sort_tracker kalman_filter hungarian Eigen3::Eigen)

# --- Phase 2-M2: SLAM libraries (from scratch) ---

add_library(so3 src/so3.cpp)
target_link_libraries(so3 Eigen3::Eigen)

add_library(imu_preintegration src/imu_preintegration.cpp)
target_link_libraries(imu_preintegration so3 Eigen3::Eigen)

add_library(scan_context src/scan_context.cpp)
target_link_libraries(scan_context Eigen3::Eigen)

add_library(pose_graph_slam src/pose_graph_slam.cpp)
# g2o_core provides SparseOptimizer::computeMarginals used by computeMarginalsG2O (P2-M3).
# The include path is also needed since g2o headers live under /usr/local/include.
target_link_libraries(pose_graph_slam so3 imu_preintegration Eigen3::Eigen
    g2o_core g2o_types_slam3d g2o_solver_eigen g2o_stuff)
target_include_directories(pose_graph_slam PUBLIC /usr/local/include)

# --- Phase 2-M3: WorldGrid (accumulated BEV world map) ---
add_library(world_grid src/world_grid.cpp)
target_link_libraries(world_grid traversability pose_graph_slam Eigen3::Eigen)
# stb_image_write + tinycolormap are header-only; include paths already set globally.

add_executable(slam_runner src/slam_runner.cpp)
target_link_libraries(slam_runner pose_graph_slam scan_context imu_preintegration so3 point_cloud_loader g2o_wrapper Eigen3::Eigen)

# --- P2-M3 accumulator runner ---
add_executable(accumulator_runner src/accumulator_runner.cpp)
target_link_libraries(accumulator_runner
    world_grid traversability pose_graph_slam
    point_cloud_loader ransac_ground_seg
    so3 imu_preintegration Eigen3::Eigen)

add_library(g2o_wrapper src/g2o_wrapper.cpp)
target_link_libraries(g2o_wrapper pose_graph_slam Eigen3::Eigen
    g2o_core g2o_types_slam3d g2o_solver_eigen g2o_stuff fmt)
target_include_directories(g2o_wrapper PUBLIC /usr/local/include)

option(BUILD_ROS2_NODES "Build ROS2 nodes" OFF)
# --- ROS2 nodes (wrappers around core libs) ---
if(BUILD_ROS2_NODES)
    add_executable(ground_seg_node ros2_nodes/ground_seg_node.cpp)
    ament_target_dependencies(ground_seg_node rclcpp sensor_msgs)
    target_link_libraries(ground_seg_node ransac_ground_seg)

    add_executable(traversability_node ros2_nodes/traversability_node.cpp)
    ament_target_dependencies(traversability_node rclcpp sensor_msgs nav_msgs)
    target_link_libraries(traversability_node traversability ransac_ground_seg)

    add_executable(projection_node ros2_nodes/projection_node.cpp)
    ament_target_dependencies(projection_node rclcpp sensor_msgs)
    target_link_libraries(projection_node cam_lidar_projection)

    add_executable(tracker_node ros2_nodes/tracker_node.cpp)
    ament_target_dependencies(tracker_node rclcpp visualization_msgs geometry_msgs)
    target_link_libraries(tracker_node sort_tracker)

    add_executable(safety_node ros2_nodes/safety_node.cpp)
    ament_target_dependencies(safety_node rclcpp geometry_msgs visualization_msgs nav_msgs)
    target_link_libraries(safety_node safety_supervisor)

    # --- Phase 3: Costmap layer plugin ---

    add_library(traversability_costmap_layer SHARED ros2_nodes/traversability_costmap_layer.cpp)
    ament_target_dependencies(traversability_costmap_layer rclcpp nav2_costmap_2d nav_msgs)
endif()
# --- Tests ---

if(BUILD_TESTING)
    find_package(ament_cmake_gtest REQUIRED)

    # Phase 1 tests
    ament_add_gtest(test_ransac tests/cpp/test_ransac.cpp)
    target_link_libraries(test_ransac ransac_ground_seg)

    ament_add_gtest(test_traversability tests/cpp/test_traversability.cpp)
    target_link_libraries(test_traversability traversability)

    ament_add_gtest(test_projection tests/cpp/test_projection.cpp)
    target_link_libraries(test_projection cam_lidar_projection)

    ament_add_gtest(test_safety tests/cpp/test_safety.cpp)
    target_link_libraries(test_safety safety_supervisor)

    # Phase 2 tests
    ament_add_gtest(test_kalman tests/cpp/test_kalman.cpp)
    target_link_libraries(test_kalman kalman_filter)

    ament_add_gtest(test_hungarian tests/cpp/test_hungarian.cpp)
    target_link_libraries(test_hungarian hungarian)

    ament_add_gtest(test_sort_tracker tests/cpp/test_sort_tracker.cpp)
    target_link_libraries(test_sort_tracker sort_tracker)

    ament_add_gtest(test_loader tests/cpp/test_loader.cpp WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    target_link_libraries(test_loader point_cloud_loader)

    # Phase 2-M2 tests
    ament_add_gtest(test_so3 tests/cpp/test_so3.cpp)
    target_link_libraries(test_so3 so3)

    ament_add_gtest(test_imu_preintegration tests/cpp/test_imu_preintegration.cpp)
    target_link_libraries(test_imu_preintegration imu_preintegration so3)

    ament_add_gtest(test_scan_context tests/cpp/test_scan_context.cpp)
    target_link_libraries(test_scan_context scan_context)

    ament_add_gtest(test_pose_graph_slam tests/cpp/test_pose_graph_slam.cpp)
    target_link_libraries(test_pose_graph_slam pose_graph_slam so3 imu_preintegration)

    # --- Phase 2-M3 tests ---
    ament_add_gtest(test_world_grid tests/cpp/test_world_grid.cpp)
    target_link_libraries(test_world_grid world_grid traversability pose_graph_slam so3 imu_preintegration)

    # Runs with cwd = repo root so the test can find tests/data/*.csv by relative path.
    ament_add_gtest(test_load_poses tests/cpp/test_load_poses.cpp
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    target_link_libraries(test_load_poses pose_graph_slam so3 imu_preintegration)

    # Pose covariance extraction — requires g2o linkage for computeMarginalsG2O.
    ament_add_gtest(test_pose_covariance tests/cpp/test_pose_covariance.cpp)
    target_link_libraries(test_pose_covariance pose_graph_slam g2o_wrapper so3 imu_preintegration
        g2o_core g2o_types_slam3d g2o_solver_eigen g2o_stuff)
endif()

if(BUILD_ROS2_NODES)
    # --- Install ---

    install(TARGETS
        ground_seg_node
        traversability_node
        projection_node
        tracker_node
        safety_node
        DESTINATION lib/${PROJECT_NAME}
    )

    install(TARGETS traversability_costmap_layer
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
    )

endif()

install(DIRECTORY launch/ DESTINATION share/${PROJECT_NAME}/launch)
install(DIRECTORY config/ DESTINATION share/${PROJECT_NAME}/config)

ament_package()
