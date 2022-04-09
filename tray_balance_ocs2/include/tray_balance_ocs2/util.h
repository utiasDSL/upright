#pragma once

#include <string>

#include <Eigen/Eigen>
#include <ros/package.h>

#include <ocs2_core/misc/LoadData.h>

namespace ocs2 {
namespace mobile_manipulator {

inline std::tuple<std::string, std::string> load_urdf_paths(
    const std::string& taskFile) {
    boost::property_tree::ptree pt;
    boost::property_tree::read_info(taskFile, pt);

    std::string urdf_ros_package_name;
    std::string robot_urdf_rel_path;
    std::string obstacle_urdf_rel_path;

    loadData::loadPtreeValue(pt, urdf_ros_package_name, "urdfSettings.package",
                             true);
    loadData::loadPtreeValue(pt, robot_urdf_rel_path,
                             "urdfSettings.robotUrdfPath", true);
    loadData::loadPtreeValue(pt, obstacle_urdf_rel_path,
                             "urdfSettings.obstacleUrdfPath", true);

    const std::string urdf_ros_package_path =
        ros::package::getPath(urdf_ros_package_name);
    const std::string robot_urdf_path =
        urdf_ros_package_path + robot_urdf_rel_path;
    const std::string obstacle_urdf_path =
        urdf_ros_package_path + obstacle_urdf_rel_path;

    return std::tuple<std::string, std::string>(robot_urdf_path,
                                                obstacle_urdf_path);
}

template <typename Scalar>
Eigen::Matrix<Scalar, 2, 2> base_rotation_matrix(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& state) {
    // clang-format off
    const auto theta = state(2);
    Eigen::Matrix<Scalar, 2, 2> C_wb;
    C_wb << cos(theta), -sin(theta),
            sin(theta),  cos(theta);
    // clang-format on
    return C_wb;
}

}  // namespace mobile_manipulator
}  // namespace ocs2
